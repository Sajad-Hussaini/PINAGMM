import numpy as np
import matplotlib.pyplot as plt
from pinagmm import PINAGMM


def main():
    print("Loading PINAGMM...")
    gmm = PINAGMM()

    print("\n--- Generating Stochastic Ground Motions ---")
    # Simulate ground motions using the median parameter prediction (n_samples=0)
    # We generate 5 realizations (n_simulations=5) to see the phase aleatory variability
    ts_m_med, ts_i_med, ts_v_med = gmm.simulate(
        Mw=6.5,
        Ztor=1.0,
        Rrup=25.0,
        Vs30=560.0,
        Fm="0",
        dt=0.005,
        n_samples=0,  # Use strictly the median GMM prediction
        n_simulations=5,  # Generate 5 stochastic realizations
    )

    # -------------------------------------------------------------------------
    # Visualization 1: Time Series (Acceleration, Velocity, Displacement)
    # -------------------------------------------------------------------------
    print("\n--- Visualizing 3-Component Time Series ---")

    # Scale factor to convert acceleration from 'g' to 'cm/s^2'
    # (The underlying sgsim package natively integrates acceleration to get vel/disp in these scaled units)
    scale = 980.665

    fig, axes = plt.subplots(
        3, 3, sharex="col", sharey="row", figsize=(12, 6), constrained_layout=True
    )

    comps = [
        ("Major", ts_m_med, "tab:blue"),
        ("Intermediate", ts_i_med, "tab:orange"),
        ("Vertical", ts_v_med, "tab:green"),
    ]

    metrics = [
        ("ac", "Acceleration\n(g)", 1.0),
        ("vel", "Velocity\n(cm/s)", scale),
        ("disp", "Displacement\n(cm)", scale),
    ]

    for col, (comp_name, sim, color) in enumerate(comps):
        axes[0, col].set_title(f"{comp_name} Component")

        for row, (metric, ylabel, scale_factor) in enumerate(metrics):
            ax = axes[row, col]

            # Extract the raw simulation array.
            # Note: sim.ac, sim.vel, sim.disp are typically shaped (n_simulations, npts)
            # We transpose it to (npts, n_simulations) for easy plotting against the time vector.
            data = getattr(sim, metric)
            if getattr(data, "ndim", 1) > 1:
                data = data.T

            # Plot a faint ensemble of all 5 realizations
            ax.plot(
                sim.t,
                data * scale_factor,
                color=color,
                alpha=0.3,
                linewidth=0.5,
            )

            # Highlight one representative realization in black
            ax.plot(sim.t, data[:, 0] * scale_factor, color="k", linewidth=0.8)

            if col == 0:
                ax.set_ylabel(ylabel)
            if row == 2:
                ax.set_xlabel("Time (s)")

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.minorticks_on()
            ax.grid(axis="both", which="minor", linewidth=0.1, alpha=0.2)
            ax.grid(axis="both", which="major", linewidth=0.2, alpha=0.3)

    plt.savefig("simulated_3_component_traces.png", dpi=300, bbox_inches="tight")
    print("Saved 'simulated_3_component_traces.png'")

    # -------------------------------------------------------------------------
    # Visualization 2: Target Conditioning (Conditional Mean Spectra)
    # -------------------------------------------------------------------------
    print("\n--- Visualizing Conditional Hazard Targeting ---")

    # Let's say we want a high-hazard scenario where the Major Component's
    # PGA (M_PGV) or Spectral Acceleration at 1.0s is specifically 0.9g.
    target_conditions = {"M_Sa_1": 0.9}

    # Generate conditioned samples. The ML model automatically adjusts all other
    # intensity measures and physical parameters to physically justify this specific target.
    ts_m_cond, ts_i_cond, ts_v_cond = gmm.simulate(
        Mw=6.5,
        Ztor=1.0,
        Rrup=25.0,
        Vs30=560.0,
        Fm="0",
        dt=0.005,
        conditions=target_conditions,
        n_samples=5,  # 5 conditioned parameter sets from the GMM
        n_simulations=1,  # 1 stochastic realization per set
    )

    # Calculate Response Spectra for the 5 conditioned simulations
    periods_val = np.logspace(-2, 1, 50)

    # (ts_m_cond is a list of GroundMotion objects because n_samples > 1)
    sa_cond_m = np.zeros((5, len(periods_val)))
    for i, sim_realization in enumerate(ts_m_cond):
        _, _, sa = sim_realization.response_spectra(periods_val)
        sa_cond_m[i] = sa.flatten()

    plt.figure(figsize=(7, 5))

    # Plot the full ensemble cloud
    plt.loglog(periods_val, sa_cond_m.T, color="tab:blue", lw=0.2, alpha=0.2)

    # Plot the median of the conditioned cloud
    sim_p50 = np.percentile(sa_cond_m, 50, axis=0)
    plt.loglog(
        periods_val, sim_p50, color="k", linewidth=1.5, label="Median Conditional Sa"
    )

    # Plot the specific condition we demanded
    plt.plot(
        [1.0],
        [0.9],
        marker="*",
        color="red",
        markersize=15,
        label="User Target (0.9g at 1.0s)",
    )

    plt.xlabel("Period (s)")
    plt.ylabel("Spectral Acceleration (g)")
    plt.title("Conditional Mean Spectra (Targeting 0.9g at 1.0s) (No Exact Match)")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.savefig("conditional_spectra_target.png", dpi=300, bbox_inches="tight")
    print("Saved 'conditional_spectra_target.png'")


if __name__ == "__main__":
    main()
