"""Cross-check dlr.ieee738 against the independent open-source `linerate` package.

Compares each heat-balance term (forced convection, natural convection, radiative
cooling) and the resulting ampacity over a grid of weather. Solar gain is not
compared: linerate derives it from sun position, while this project passes a
radiation value in directly. Solar gain is a one-line product here, so it is
checked by hand in test_ieee738.py.

Run from the project folder:
    pip install linerate
    python scripts/validate_ieee738.py

Exit code is 0 if every term agrees within TOL, 1 otherwise.
"""

import itertools
import math
import sys

import numpy as np

from dlr import config
from dlr.ieee738 import (
    calculate_ampacity,
    forced_convective_cooling,
    natural_convective_cooling,
    radiative_cooling,
    thermal_terms,
)

try:
    from linerate.equations import dimensionless, radiative_cooling as lr_rad
    from linerate.equations.ieee738 import convective_cooling as lr_cc
except ImportError:
    sys.exit("linerate is not installed. Run: pip install linerate")

TOL = 0.01  # 1 % relative; the two codes should agree far better than this
D = config.DRAKE.diameter_m
R = config.DRAKE.r_at_tmax
EMISSIVITY = config.DRAKE.emissivity


def linerate_terms(t_c, t_a, v, phi_deg, elevation=0.0):
    """Forced, natural convection and radiative cooling from linerate (W/m)."""
    t_film = lr_cc.compute_air_temperature_at_boundary_layer(t_c, t_a)
    rho = lr_cc.compute_air_density(t_film, elevation)
    mu = lr_cc.compute_dynamic_viscosity_of_air(t_film)
    nu = lr_cc.compute_kinematic_viscosity_of_air(mu, rho)
    k_air = lr_cc.compute_thermal_conductivity_of_air(t_film)
    re = dimensionless.compute_reynolds_number(v, D, nu)
    k_phi = lr_cc.compute_wind_direction_factor(math.radians(phi_deg))
    forced = lr_cc.compute_forced_convection(k_phi, re, k_air, t_c, t_a)
    natural = lr_cc.compute_natural_convection(rho, D, t_c, t_a)
    q_c = lr_cc.compute_convective_cooling(forced, natural)
    q_r = lr_rad.compute_radiative_cooling(t_c, t_a, D, EMISSIVITY)
    return float(forced), float(natural), float(q_c), float(q_r)


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-12)


def main():
    t_c_vals = (50.0, 75.0)
    t_a_vals = (-5.0, 10.0, 25.0, 35.0)
    v_vals = (0.1, 0.5, 2.0, 6.0, 12.0)
    phi_vals = (15.0, 45.0, 90.0)

    worst = {"forced": 0.0, "natural": 0.0, "q_c": 0.0, "q_r": 0.0, "ampacity": 0.0}
    worst_case = {}
    n = 0
    for t_c, t_a, v, phi in itertools.product(t_c_vals, t_a_vals, v_vals, phi_vals):
        n += 1
        f_lr, n_lr, qc_lr, qr_lr = linerate_terms(t_c, t_a, v, phi)

        f_me = forced_convective_cooling(t_c, t_a, v, phi, D)
        n_me = natural_convective_cooling(t_c, t_a, D)
        terms = thermal_terms(t_c, t_a, v, phi, 0.0, D, EMISSIVITY, 0.0)
        qr_me = radiative_cooling(t_c, t_a, D, EMISSIVITY)

        # Ampacity with zero solar gain, from linerate's terms: I = sqrt((qc+qr)/R)
        i_lr = math.sqrt((qc_lr + qr_lr) / R)
        i_me = calculate_ampacity(t_c, t_a, v, phi, 0.0, D, R, EMISSIVITY, 0.0)

        for key, a, b in (
            ("forced", f_me, f_lr),
            ("natural", n_me, n_lr),
            ("q_c", terms["q_c"], qc_lr),
            ("q_r", qr_me, qr_lr),
            ("ampacity", i_me, i_lr),
        ):
            e = rel(a, b)
            if e > worst[key]:
                worst[key] = e
                worst_case[key] = (t_c, t_a, v, phi, a, b)

    print(f"Compared {n} weather cases against linerate {_linerate_version()}\n")
    print(f"{'term':10s} {'max rel. diff':>14s}   worst case (Tc, Ta, v, phi | mine, linerate)")
    ok = True
    for key, e in worst.items():
        flag = "OK " if e <= TOL else "FAIL"
        ok &= e <= TOL
        c = worst_case.get(key)
        detail = f"{c[:4]} | {c[4]:.4g}, {c[5]:.4g}" if c else ""
        print(f"{key:10s} {e:14.3e}   {flag} {detail}")

    print("\nPASS: dlr.ieee738 agrees with linerate within "
          f"{TOL:.0%}." if ok else "\nFAIL: see rows above.")
    return 0 if ok else 1


def _linerate_version():
    try:
        import linerate
        return getattr(linerate, "__version__", "")
    except Exception:
        return ""


if __name__ == "__main__":
    sys.exit(main())
    