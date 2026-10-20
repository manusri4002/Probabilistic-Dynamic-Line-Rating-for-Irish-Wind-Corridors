"""
IEEE Std 738-2012 steady-state thermal solver (scalar, pure Python).

Pure `math` and scalar arguments on purpose, so the same file runs unchanged
under MicroPython on an ESP32. Units: metres, degC, m/s, W/m, W/m2, ohm/m.
"""

import math

STEFAN_BOLTZMANN = 5.670374419e-8  # W/(m2 K4)


def calculate_air_properties(t_film, elevation=0.0):
    """Air density (kg/m3), dynamic viscosity (Pa s) and thermal conductivity (W/m/K)
    at film temperature t_film (degC) and elevation (m), per IEEE 738-2012."""
    density = (1.293 - 1.525e-4 * elevation + 6.379e-9 * elevation ** 2) / (1.0 + 0.00367 * t_film)
    viscosity = (1.458e-6 * (t_film + 273.15) ** 1.5) / (t_film + 273.15 + 110.4)
    k_air = 2.424e-2 + 7.477e-5 * t_film - 4.407e-9 * t_film ** 2
    return density, viscosity, k_air


def forced_convective_cooling(t_c, t_a, v_wind, phi_deg, d_m, elevation=0.0):
    """Forced convective heat loss q_c (W/m). The larger of the low- and high-wind
    expressions is used, as the standard requires. Zero if there is no wind or the
    conductor is not warmer than the air."""
    dt = t_c - t_a
    if v_wind <= 0.0 or dt <= 0.0:
        return 0.0

    density, viscosity, k_air = calculate_air_properties((t_c + t_a) / 2.0, elevation)

    phi = math.radians(phi_deg)
    k_angle = 1.194 - math.cos(phi) + 0.194 * math.cos(2.0 * phi) + 0.368 * math.sin(2.0 * phi)
    reynolds = d_m * v_wind * density / viscosity

    qc1 = k_angle * (1.01 + 1.35 * reynolds ** 0.52) * k_air * dt
    qc2 = k_angle * 0.754 * reynolds ** 0.6 * k_air * dt
    return max(qc1, qc2)


def natural_convective_cooling(t_c, t_a, d_m, elevation=0.0):
    """Natural (zero-wind) convective heat loss q_cn (W/m). Zero if t_c <= t_a."""
    dt = t_c - t_a
    if dt <= 0.0:
        return 0.0
    density, _, _ = calculate_air_properties((t_c + t_a) / 2.0, elevation)
    return 3.645 * math.sqrt(density) * d_m ** 0.75 * dt ** 1.25


def radiative_cooling(t_c, t_a, d_m, emissivity=0.5):
    """Radiative heat loss q_r (W/m)."""
    return math.pi * d_m * emissivity * STEFAN_BOLTZMANN * ((t_c + 273.15) ** 4 - (t_a + 273.15) ** 4)


def solar_heat_gain(d_m, solar_radiation=1000.0, absorptivity=0.5):
    """Solar heat gain q_s (W/m), taking solar_radiation as W/m2 on the projected area."""
    return absorptivity * solar_radiation * d_m


def thermal_terms(t_c, t_a, v_wind, phi_deg, solar_rad, d_m,
                  emissivity=0.5, absorptivity=0.5, elevation=0.0):
    """All heat-balance terms (W/m) in one place: q_c, q_r, q_s.
    q_c is the larger of forced and natural convection (IEEE 738 rule for low wind)."""
    q_c = max(
        forced_convective_cooling(t_c, t_a, v_wind, phi_deg, d_m, elevation),
        natural_convective_cooling(t_c, t_a, d_m, elevation),
    )
    q_r = radiative_cooling(t_c, t_a, d_m, emissivity)
    q_s = solar_heat_gain(d_m, solar_rad, absorptivity)
    return {"q_c": q_c, "q_r": q_r, "q_s": q_s}


def calculate_ampacity(t_c, t_a, v_wind, phi_deg, solar_rad, d_m, r_tc,
                       emissivity=0.5, absorptivity=0.5, elevation=0.0):
    """Maximum current (A) that holds the conductor at t_c.

    Steady-state balance: I^2 R + q_s = q_c + q_r. Returns 0.0 when the weather
    alone already heats the conductor to t_c (nothing left to carry current)."""
    if d_m <= 0.0 or r_tc <= 0.0:
        raise ValueError("conductor diameter and resistance must be positive")

    t = thermal_terms(t_c, t_a, v_wind, phi_deg, solar_rad, d_m, emissivity, absorptivity, elevation)
    net = t["q_c"] + t["q_r"] - t["q_s"]
    if net <= 0.0:
        return 0.0
    return math.sqrt(net / r_tc)
