"""
Sensitivity sweep: Drake ACSR ampacity vs wind speed and wind angle (IEEE 738-2012).
Static rating is computed here from stated conservative assumptions, not hardcoded.
"""
import numpy as np
import matplotlib.pyplot as plt
from dlr.config import DRAKE, STATIC as STATIC_CASE
from dlr.ieee738 import calculate_ampacity
from dlr.pipeline import ROOT_DIR

# Conductor and weather assumptions (Drake ACSR)
D_M = DRAKE.diameter_m
T_C = DRAKE.t_max_c
R_75 = DRAKE.r_at_tmax
EMISS = DRAKE.emissivity
ABSORP = DRAKE.absorptivity

# Static (conservative planning) case
STATIC = dict(t_a=STATIC_CASE.t_a, v_wind=STATIC_CASE.v_wind,
              phi_deg=STATIC_CASE.phi_deg, solar_rad=STATIC_CASE.solar_rad)


def ampacity(t_a, v_wind, phi_deg, solar_rad):
    return calculate_ampacity(
        t_c=T_C, t_a=t_a, v_wind=v_wind, phi_deg=phi_deg, solar_rad=solar_rad,
        d_m=D_M, r_tc=R_75, emissivity=EMISS, absorptivity=ABSORP,
    )


def run_sensitivity_sweep():
    static_rating = ampacity(**STATIC)
    print(f"Computed static rating: {static_rating:.1f} A")

    wind_speeds = np.linspace(0.1, 8.0, 80)
    angles = [90.0, 60.0, 30.0, 15.0]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    for phi in angles:
        values = [ampacity(STATIC["t_a"], v, phi, STATIC["solar_rad"]) for v in wind_speeds]
        ax.plot(wind_speeds, values, linewidth=2, label=rf"Wind angle $\phi = {int(phi)}^\circ$")

    ax.axhline(static_rating, color="red", linestyle="--", linewidth=1.5,
               label=f"Computed static rating ({static_rating:.0f} A)")

    # Annotation 1: natural-convection floor (calm air), values computed not hardcoded
    floor = ampacity(STATIC["t_a"], 0.1, 90.0, STATIC["solar_rad"])
    ax.annotate(
        f"Natural-convection floor ({floor:.0f} A):\nbelow static rating in calm air",
        xy=(0.15, floor), xytext=(1.2, 300),
        arrowprops=dict(arrowstyle="->", lw=0.8), fontsize=9,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.9),
    )

    # Annotation 2: headroom at 4 m/s, ratio computed from the model
    a4 = ampacity(STATIC["t_a"], 4.0, 90.0, STATIC["solar_rad"])
    ax.annotate(
        f"{a4 / static_rating:.2f}x static rating at 4 m/s\nperpendicular wind ({a4:.0f} A)",
        xy=(4.0, a4), xytext=(4.6, 450),
        arrowprops=dict(arrowstyle="->", lw=0.8, color="darkblue"),
        fontsize=9, color="darkblue",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.9),
    )

    ax.set_title("Drake ACSR ampacity vs wind speed and angle (IEEE 738-2012)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Wind speed (m/s)", fontsize=11)
    ax.set_ylabel("Ampacity (A)", fontsize=11)
    ax.set_xlim(0.0, 8.0)
    ax.set_ylim(0, 2200)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", frameon=True)
    fig.text(0.01, 0.005,
             f"Conductor limit {T_C:.0f} degC; sweep ambient {STATIC['t_a']:.0f} degC, "
             f"solar {STATIC['solar_rad']:.0f} W/m2. Static case: "
             f"{STATIC['v_wind']} m/s at {STATIC['phi_deg']:.0f} deg.",
             fontsize=7, color="gray")
    fig.tight_layout(rect=(0, 0.02, 1, 1))

    out_dir = ROOT_DIR / "figures"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / "dlr_sensitivity_sweep.png"
    fig.savefig(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    run_sensitivity_sweep()