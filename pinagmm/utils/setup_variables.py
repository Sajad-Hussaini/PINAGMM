# A single configuration file to centralize variable definitions across the project

xraw = ["Earthquake Magnitude", "Depth to Top Of Fault Rupture Model"]
xlog = ["ClstD (km)", "Vs30 (m/s) selected for analysis"]
xcat = ["Mechanism Based on Rake Angle"]
xnumeric = xraw + xlog
xvars = xnumeric + xcat

gvars = ["Earthquake Name", "Station Name"]
yvars = [
    # ---- Temporal Parameters ----
    "M_q_duration",
    "I_q_duration",
    "V_q_duration",
    "M_q_centroid",
    "I_q_centroid",
    "V_q_centroid",
    "M_q_spread",
    "I_q_spread",
    "V_q_spread",
    # ---- SCALE PARAMETERS (EXTENSIVE) ----
    "M_q_energy",
    "I_q_energy",
    "V_q_energy",
    # ---- FREQUENCY BANDWIDTH ----
    "M_wu_value",
    "I_wu_value",
    "V_wu_value",
    "M_wl_value",
    "I_wl_value",
    "V_wl_value",
    # ---- INTENSITY MEASURES (BROADBAND) ----
    "M_PGV",
    "I_PGV",
    "V_PGV",
    # ---- SPECTRAL ACCELERATIONS (PERIOD-SPECIFIC BLOCKS) ----
    "M_Sa_0",
    "I_Sa_0",
    "V_Sa_0",
    "M_Sa_0.03",
    "I_Sa_0.03",
    "V_Sa_0.03",
    "M_Sa_0.05",
    "I_Sa_0.05",
    "V_Sa_0.05",
    "M_Sa_0.08",
    "I_Sa_0.08",
    "V_Sa_0.08",
    "M_Sa_0.1",
    "I_Sa_0.1",
    "V_Sa_0.1",
    "M_Sa_0.16",
    "I_Sa_0.16",
    "V_Sa_0.16",
    "M_Sa_0.2",
    "I_Sa_0.2",
    "V_Sa_0.2",
    "M_Sa_0.3",
    "I_Sa_0.3",
    "V_Sa_0.3",
    "M_Sa_0.4",
    "I_Sa_0.4",
    "V_Sa_0.4",
    "M_Sa_0.5",
    "I_Sa_0.5",
    "V_Sa_0.5",
    "M_Sa_0.75",
    "I_Sa_0.75",
    "V_Sa_0.75",
    "M_Sa_1",
    "I_Sa_1",
    "V_Sa_1",
    "M_Sa_1.5",
    "I_Sa_1.5",
    "V_Sa_1.5",
    "M_Sa_2",
    "I_Sa_2",
    "V_Sa_2",
    "M_Sa_3",
    "I_Sa_3",
    "V_Sa_3",
    "M_Sa_4",
    "I_Sa_4",
    "V_Sa_4",
]

y_energy = [yvars.index(v) for v in yvars if "energy" in v]
y_duration = [
    yvars.index(v)
    for v in yvars
    if any(sub in v for sub in ["duration", "centroid", "spread"])
]
y_freq = [yvars.index(v) for v in yvars if any(sub in v for sub in ["wl_", "wu_"])]
y_ims = [yvars.index(v) for v in yvars if any(sub in v for sub in ["PGV", "Sa"])]
y_ims_major = [
    yvars.index(v) for v in yvars if any(sub in v for sub in ["M_PGV", "M_Sa"])
]
y_ims_in = [yvars.index(v) for v in yvars if any(sub in v for sub in ["I_PGV", "I_Sa"])]
y_ims_ve = [yvars.index(v) for v in yvars if any(sub in v for sub in ["V_PGV", "V_Sa"])]
y_model_major = [
    yvars.index(v)
    for v in yvars
    if any(
        sub in v
        for sub in [
            "M_q_energy",
            "M_q_duration",
            "M_q_centroid",
            "M_q_spread",
            "M_wu_value",
            "M_wl_value",
        ]
    )
]
y_model_in = [
    yvars.index(v)
    for v in yvars
    if any(
        sub in v
        for sub in [
            "I_q_energy",
            "I_q_duration",
            "I_q_centroid",
            "I_q_spread",
            "I_wu_value",
            "I_wl_value",
        ]
    )
]
y_model_ve = [
    yvars.index(v)
    for v in yvars
    if any(
        sub in v
        for sub in [
            "V_q_energy",
            "V_q_duration",
            "V_q_centroid",
            "V_q_spread",
            "V_wu_value",
            "V_wl_value",
        ]
    )
]


# Indices in the specified order: Energy (E), fU, fL, Duration (D), Centroid (C), Spread (S)
def _ordered_model_indices(prefix: str):
    patterns = [
        f"{prefix}_q_energy",
        f"{prefix}_wu_value",
        f"{prefix}_wl_value",
        f"{prefix}_q_duration",
        f"{prefix}_q_centroid",
        f"{prefix}_q_spread",
    ]
    out = []
    for p in patterns:
        for i, v in enumerate(yvars):
            if p in v and i not in out:
                out.append(i)
    return out


y_model_major = _ordered_model_indices("M")
y_model_in = _ordered_model_indices("I")
y_model_ve = _ordered_model_indices("V")

# --- Scientific labels for plotting ---
ylabels = [
    # Temporal Parameters
    r"$D_{M}$",
    r"$D_{I}$",
    r"$D_{V}$",
    r"$C_{M}$",
    r"$C_{I}$",
    r"$C_{V}$",
    r"$S_{M}$",
    r"$S_{I}$",
    r"$S_{V}$",
    # Scale Parameters (Energy)
    r"$E_{M}$",
    r"$E_{I}$",
    r"$E_{V}$",
    # Frequency Bandwidth
    r"$f_{U, M}$",
    r"$f_{U, I}$",
    r"$f_{U, V}$",
    r"$f_{L, M}$",
    r"$f_{L, I}$",
    r"$f_{L, V}$",
    # Intensity Measures (Broadband)
    r"$PGV_{M}$",
    r"$PGV_{I}$",
    r"$PGV_{V}$",
    # Spectral Accelerations (Period-Specific Blocks)
    r"$PGA_{M}$",
    r"$PGA_{I}$",
    r"$PGA_{V}$",
    r"$SA_{0.03s, M}$",
    r"$SA_{0.03s, I}$",
    r"$SA_{0.03s, V}$",
    r"$SA_{0.05s, M}$",
    r"$SA_{0.05s, I}$",
    r"$SA_{0.05s, V}$",
    r"$SA_{0.08s, M}$",
    r"$SA_{0.08s, I}$",
    r"$SA_{0.08s, V}$",
    r"$SA_{0.1s, M}$",
    r"$SA_{0.1s, I}$",
    r"$SA_{0.1s, V}$",
    r"$SA_{0.16s, M}$",
    r"$SA_{0.16s, I}$",
    r"$SA_{0.16s, V}$",
    r"$SA_{0.2s, M}$",
    r"$SA_{0.2s, I}$",
    r"$SA_{0.2s, V}$",
    r"$SA_{0.3s, M}$",
    r"$SA_{0.3s, I}$",
    r"$SA_{0.3s, V}$",
    r"$SA_{0.4s, M}$",
    r"$SA_{0.4s, I}$",
    r"$SA_{0.4s, V}$",
    r"$SA_{0.5s, M}$",
    r"$SA_{0.5s, I}$",
    r"$SA_{0.5s, V}$",
    r"$SA_{0.75s, M}$",
    r"$SA_{0.75s, I}$",
    r"$SA_{0.75s, V}$",
    r"$SA_{1.0s, M}$",
    r"$SA_{1.0s, I}$",
    r"$SA_{1.0s, V}$",
    r"$SA_{1.5s, M}$",
    r"$SA_{1.5s, I}$",
    r"$SA_{1.5s, V}$",
    r"$SA_{2.0s, M}$",
    r"$SA_{2.0s, I}$",
    r"$SA_{2.0s, V}$",
    r"$SA_{3.0s, M}$",
    r"$SA_{3.0s, I}$",
    r"$SA_{3.0s, V}$",
    r"$SA_{4.0s, M}$",
    r"$SA_{4.0s, I}$",
    r"$SA_{4.0s, V}$",
]

periods = [
    0.03,
    0.05,
    0.08,
    0.1,
    0.16,
    0.2,
    0.3,
    0.4,
    0.5,
    0.75,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
]

dtype_mapping = {
    "Earthquake Name": "str",
    "Station Name": "str",
    "Mechanism Based on Rake Angle": "str",
    **{var: "float" for var in xnumeric},
}

col_ids = {
    "Mw": 0,
    "Ztor": 1,
    "Rrup": 2,
    "Vs30": 3,
    "im_indices": [
        yvars.index(v) for v in yvars if any(sub in v for sub in ["PGV", "Sa"])
    ],
    "energy_indices": [yvars.index(v) for v in yvars if "energy" in v],
    "duration_indices": [yvars.index(v) for v in yvars if "duration" in v],
    "centroid_indices": [yvars.index(v) for v in yvars if "centroid" in v],
    "spread_indices": [yvars.index(v) for v in yvars if "spread" in v],
    "wl_indices": [yvars.index(v) for v in yvars if "wl_" in v],
    "wu_indices": [yvars.index(v) for v in yvars if "wu_" in v],
}

group_definitions = [
    {
        "name": "IM_Energy",
        "out_idx": col_ids["im_indices"] + col_ids["energy_indices"],
        "mono_in": [col_ids["Mw"], col_ids["Rrup"]],
        "signs": [1.0, -1.0],
        "interactions": [
            {
                "features": [col_ids["Mw"], col_ids["Rrup"]],
                "monotonic": True,
                "signs": [1.0, -1.0],
            }
        ],
    },
    {
        "name": "Duration",
        "out_idx": col_ids["duration_indices"]
        + col_ids["centroid_indices"]
        + col_ids["spread_indices"],
        "mono_in": [col_ids["Mw"], col_ids["Rrup"]],
        "signs": [1.0, 1.0],
        "interactions": [
            {
                "features": [col_ids["Mw"], col_ids["Rrup"]],
                "monotonic": True,
                "signs": [1.0, 1.0],
            }
        ],
    },
    {
        "name": "Freq_Lower",
        "out_idx": col_ids["wl_indices"],
        "mono_in": [col_ids["Mw"], col_ids["Rrup"]],
        "signs": [-1.0, -1.0],
        "interactions": [
            {
                "features": [col_ids["Mw"], col_ids["Rrup"]],
                "monotonic": True,
                "signs": [-1.0, -1.0],
            }
        ],
    },
    {
        "name": "Freq_Upper",
        "out_idx": col_ids["wu_indices"],
        "mono_in": [col_ids["Mw"], col_ids["Rrup"]],
        "signs": [-1.0, -1.0],
        "interactions": [
            {
                "features": [col_ids["Mw"], col_ids["Rrup"]],
                "monotonic": True,
                "signs": [-1.0, -1.0],
            }
        ],
    },
]
