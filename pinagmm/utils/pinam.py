import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .setup_variables import group_definitions


# ---------------------------------------------------------------------------
# Sub-networks (one per input feature)
# ---------------------------------------------------------------------------
class MonotonicLinear(nn.Linear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, F.softplus(self.weight), self.bias)


class SubNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_layers, dropout=0.0, is_monotonic=False):
        super().__init__()

        LinearType = MonotonicLinear if is_monotonic else nn.Linear
        layers = []
        curr_dim = in_dim
        for h_dim in hidden_layers:
            layers.append(LinearType(curr_dim, h_dim))
            layers.append(nn.Tanh())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            curr_dim = h_dim

        layers.append(LinearType(curr_dim, out_dim, bias=False))
        self.deep_path = nn.Sequential(*layers)

        if not is_monotonic:
            self.skip = LinearType(in_dim, out_dim, bias=False)

        self.is_monotonic = is_monotonic

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.deep_path(x)
        if not self.is_monotonic:
            out = out + self.skip(x)
        return out


# ---------------------------------------------------------------------------
# Pure Additive Network
# ---------------------------------------------------------------------------
class PureAdditiveNetwork(nn.Module):
    def __init__(
        self,
        mono_in_dim,
        free_in_dim,
        out_dim,
        hidden_layers,
        dropout=0.0,
        interactions=None,
    ):
        super().__init__()

        self.mono_nets = nn.ModuleList([
            SubNetwork(1, out_dim, hidden_layers, dropout, is_monotonic=True)
            for _ in range(mono_in_dim)
        ])

        self.free_nets = nn.ModuleList([
            SubNetwork(1, out_dim, hidden_layers, dropout, is_monotonic=False)
            for _ in range(free_in_dim)
        ])

        self.interactions = interactions or []
        self.interaction_nets = nn.ModuleList()
        for inter in self.interactions:
            in_dim = len(inter["features"])
            is_mono = inter.get("monotonic", False)
            self.interaction_nets.append(
                SubNetwork(
                    in_dim,
                    out_dim,
                    hidden_layers,
                    dropout,
                    is_monotonic=is_mono,
                )
            )

        self.global_bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x_mono, x_free, x_inter=None) -> torch.Tensor:
        out = self.global_bias
        if x_mono is not None:
            out = out + sum(
                net(x_mono[:, i : i + 1]) for i, net in enumerate(self.mono_nets)
            )
        if x_free is not None:
            out = out + sum(
                net(x_free[:, i : i + 1]) for i, net in enumerate(self.free_nets)
            )
        if x_inter is not None:
            out = out + sum(
                net(x_inter[i]) for i, net in enumerate(self.interaction_nets)
            )
        return out


# ---------------------------------------------------------------------------
# GroupArchitecture
# ---------------------------------------------------------------------------
class GroupArchitecture(nn.Module):
    def __init__(self, input_dim, out_idx, mono_in, signs, config, device="cpu"):
        super().__init__()

        self.input_dim = input_dim
        self.out_idx = out_idx
        self.device = device
        self.config = config

        # Config overrides
        mono_in = config.get("mono_features", mono_in)
        signs = config.get("mono_signs", signs)
        interactions = config.get("interactions", [])

        all_in = set(range(input_dim))
        free_in = config.get("free_features", sorted(all_in - set(mono_in)))

        # Register index tensors as buffers so they follow device moves
        self.register_buffer("mono_idx", torch.tensor(mono_in, dtype=torch.long))
        self.register_buffer("signs_t", torch.tensor(signs, dtype=torch.float32))
        self.register_buffer("free_idx", torch.tensor(free_in, dtype=torch.long))

        self.interactions = interactions
        for i, inter in enumerate(self.interactions):
            self.register_buffer(
                f"inter_{i}_idx", torch.tensor(inter["features"], dtype=torch.long)
            )
            if inter.get("monotonic", False) and "signs" in inter:
                self.register_buffer(
                    f"inter_{i}_signs",
                    torch.tensor(inter["signs"], dtype=torch.float32),
                )

        self.network = PureAdditiveNetwork(
            mono_in_dim=len(mono_in),
            free_in_dim=len(free_in),
            out_dim=len(out_idx),
            hidden_layers=config["hidden_layers"],
            dropout=config.get("dropout", 0.0),
            interactions=self.interactions,
        )

        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_mono = (
            (x[:, self.mono_idx] * self.signs_t) if len(self.mono_idx) > 0 else None
        )
        x_free = x[:, self.free_idx] if len(self.free_idx) > 0 else None

        x_inter = [] if self.interactions else None
        if self.interactions:
            for i, inter in enumerate(self.interactions):
                idx = getattr(self, f"inter_{i}_idx")
                x_i = x[:, idx]
                if inter.get("monotonic", False) and "signs" in inter:
                    x_i = x_i * getattr(self, f"inter_{i}_signs")
                x_inter.append(x_i)

        return self.network(x_mono, x_free, x_inter)

    def compute_l2_penalty(self) -> torch.Tensor:
        """Calculates one-sided L2 penalty to prevent dead units and vanishing gradients."""
        l2_reg = torch.tensor(0.0, device=self.device)
        for name, module in self.named_modules():
            if isinstance(module, (MonotonicLinear, nn.Linear)) and "skip" not in name:
                if hasattr(module, "weight") and module.weight.requires_grad:
                    if isinstance(module, MonotonicLinear):
                        # One-Sided L2: Only penalize positive raw weights to prevent dead units
                        l2_reg = l2_reg + torch.sum(F.relu(module.weight) ** 2)
                    else:
                        l2_reg = l2_reg + torch.sum(module.weight**2)
        return l2_reg


# ---------------------------------------------------------------------------
# SingleGroupGMM — pure-Python estimator wrapper (no nn.Module inheritance)
# ---------------------------------------------------------------------------
class SingleGroupGMM:
    def __init__(self, input_dim, out_idx, mono_in, signs, config, device="cpu"):
        self.input_dim = input_dim
        self.out_idx = out_idx
        self.mono_in = mono_in
        self.signs = signs
        self.config = config
        self.device = device
        self._arch = None

        self.train_loss_history = []
        self.val_loss_history = []

    def fit(self, X, y, X_val=None, y_val=None):
        """Train the model using full-batch L-BFGS."""
        self._arch = GroupArchitecture(
            input_dim=self.input_dim,
            out_idx=self.out_idx,
            mono_in=self.mono_in,
            signs=self.signs,
            config=self.config,
            device=self.device,
        )

        self.train_loss_history = []
        self.val_loss_history = []

        # Convert data to tensors
        X_t = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.as_tensor(y, dtype=torch.float32, device=self.device)

        # L2 regularisation
        wd_lambda = self.config.get("weight_decay", 0.0)

        # Optimiser
        optimizer = optim.LBFGS(
            self._arch.parameters(),
            lr=self.config.get("lr", 1.0),
            max_iter=self.config.get("lbfgs_max_iter", 20),
            history_size=self.config.get("lbfgs_history", 100),
            line_search_fn="strong_wolfe",
        )

        mse_loss = nn.MSELoss()

        # Validation / early-stopping state
        has_val = X_val is not None and y_val is not None
        best_state = None
        best_val = float("inf")
        patience_ctr = 0
        if has_val:
            X_val_t = torch.as_tensor(X_val, dtype=torch.float32, device=self.device)
            y_val_t = torch.as_tensor(y_val, dtype=torch.float32, device=self.device)
            patience = self.config.get("patience", 5)
            best_state = {
                k: v.cpu().clone() for k, v in self._arch.state_dict().items()
            }

        # Training loop
        epochs = self.config.get("epochs", 5)
        train_tol = self.config.get("train_tol", 1e-4)
        prev_loss = float("inf")

        for _ in range(epochs):
            self._arch.train()

            def closure() -> torch.Tensor:
                optimizer.zero_grad()
                loss = mse_loss(self._arch(X_t), y_t)

                if wd_lambda > 0:
                    loss = loss + 0.5 * wd_lambda * self._arch.compute_l2_penalty()

                loss.backward()
                return loss

            loss_value = optimizer.step(closure)
            current_loss = loss_value.item()
            self.train_loss_history.append(current_loss)

            if has_val:
                self._arch.eval()
                with torch.no_grad():
                    val_loss = mse_loss(self._arch(X_val_t), y_val_t).item()
                self.val_loss_history.append(val_loss)

                if val_loss < best_val:
                    best_val = val_loss
                    best_state = {
                        k: v.cpu().clone() for k, v in self._arch.state_dict().items()
                    }
                    patience_ctr = 0
                else:
                    patience_ctr += 1
                    if patience_ctr >= patience:
                        break
            else:
                # Early stopping based on training loss when validation is not available
                if abs(prev_loss - current_loss) < train_tol:
                    break

            prev_loss = current_loss

        # Restore best-validation-loss weights
        if has_val and best_state is not None:
            self._arch.load_state_dict(best_state)

        return self

    def predict(self, X):
        """Run inference and return predictions."""
        if self._arch is None:
            raise RuntimeError("Model has not been fitted yet.")
        self._arch.eval()
        with torch.no_grad():
            X_t = torch.as_tensor(X, dtype=torch.float32, device=self.device)
            return self._arch(X_t).cpu().numpy()

    def __repr__(self):
        return f"{self.__class__.__name__}(input_dim={self.input_dim}, out_idx={self.out_idx})"


# ---------------------------------------------------------------------------
# EnsembleGMM
# ---------------------------------------------------------------------------
class EnsembleGMM:
    def __init__(self, input_dim, output_dim, configs, device="cpu"):
        self.output_dim = output_dim
        self.device = device
        self.groups = group_definitions

        self.models = {
            grp["name"]: SingleGroupGMM(
                input_dim=input_dim,
                out_idx=grp["out_idx"],
                mono_in=grp["mono_in"],
                signs=grp["signs"],
                config=configs[grp["name"]],
                device=device,
            )
            for grp in self.groups
        }

    def fit(self, X, y, X_val=None, y_val=None):
        """Fit each group estimator on its slice of the target columns."""
        for grp in self.groups:
            name = grp["name"]
            idx = grp["out_idx"]
            y_sliced = y[:, idx]
            y_val_slc = y_val[:, idx] if y_val is not None else None
            self.models[name].fit(X, y_sliced, X_val, y_val_slc)
        return self

    def predict(self, X):
        """Predict all targets, reassembling slices from each group estimator."""
        preds = np.zeros((X.shape[0], self.output_dim))
        for grp in self.groups:
            name = grp["name"]
            idx = grp["out_idx"]
            preds[:, idx] = self.models[name].predict(X)
        return preds

    @property
    def train_loss_history(self):
        """Returns the training loss history for each group model."""
        return {name: model.train_loss_history for name, model in self.models.items()}

    @property
    def val_loss_history(self):
        """Returns the validation loss history for each group model."""
        return {name: model.val_loss_history for name, model in self.models.items()}

    @property
    def configs(self):
        """Returns the hyperparameter configurations for each group model."""
        return {name: model.config for name, model in self.models.items()}

    def __repr__(self):
        group_names = list(self.models.keys())
        return (
            f"{self.__class__.__name__}("
            f"output_dim={self.output_dim}, "
            f"groups={group_names}, "
            f"device='{self.device}')"
        )
