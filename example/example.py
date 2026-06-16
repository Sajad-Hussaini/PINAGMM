import numpy as np
import matplotlib.pyplot as plt
from pinagmm import PINAGMM

def main():
    # 1. Initialize the Model
    print("Loading PINAGMM...")
    gmm = PINAGMM()

    # -------------------------------------------------------------------------
    # Scenario 1: Single Prediction
    # -------------------------------------------------------------------------
    print("\n--- Scenario 1: Single Prediction ---")
    prediction = gmm.predict(Mw=6.5, Ztor=3.0, Rrup=15.0, Vs30=800.0, Fm="0")
    
    print("Core Intensity Measures:")
    print(prediction[["M_Sa_0", "M_Sa_0.2", "M_Sa_1", "M_PGV"]])

    prediction.to_csv("single_scenario_results.csv", index=False)
    print("Saved 'single_scenario_results.csv'")

    # -------------------------------------------------------------------------
    # Scenario 2: Vectorized Prediction (Attenuation Curve)
    # -------------------------------------------------------------------------
    print("\n--- Scenario 2: Attenuation Curve Plotting ---")
    distances = np.linspace(1.0, 100.0, 50)
    
    attenuation_df = gmm.predict(Mw=7.0, Ztor=3.0, Rrup=distances, Vs30=800.0, Fm="0")

    plt.figure(figsize=(8, 6))
    plt.plot(
        distances,
        attenuation_df["M_Sa_0"],
        marker=".",
        linestyle="-",
        color="b",
        label="Mw=7.0, Normal",
    )
    plt.yscale("log")
    plt.xlabel("Rupture Distance ($R_{rup}$) [km]")
    plt.ylabel("Peak Ground Acceleration (PGA) [g]")
    plt.title("PINAGMM Attenuation Curve")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.savefig("attenuation_curve.png", dpi=300, bbox_inches="tight")
    print("Saved 'attenuation_curve.png'")

    # -------------------------------------------------------------------------
    # Scenario 3: Stochastic Ground Motion Simulation
    # -------------------------------------------------------------------------
    print("\n--- Scenario 3: Time Series Simulation ---")
    
    # Simulates ground motion for the median prediction
    ts_m, ts_i, ts_v = gmm.simulate(
        Mw=6.5, Ztor=3.0, Rrup=15.0, Vs30=800.0, Fm="0", dt=0.005,
        n_samples=0, n_simulations=1
    )

    plt.figure(figsize=(10, 4))
    plt.plot(ts_m.t, ts_m.ac.squeeze(), color="k", linewidth=0.5)
    plt.xlabel("Time [s]")
    plt.ylabel("Acceleration [g]")
    plt.title("Simulated Ground Motion (Major Component)")
    plt.grid(True, alpha=0.5)
    plt.savefig("simulated_ts_major.png", dpi=300, bbox_inches="tight")
    print("Saved 'simulated_ts_major.png'")

if __name__ == "__main__":
    main()
