"""
IEEE Std 738-2012 Minimal Steady-State Thermal Solver (Corrected)
Pure Python implementation for porting ease (ESP32 MicroPython ready).
"""

import math

def calculate_air_properties(t_film, elevation=0.0):
    """Calculate air density, dynamic viscosity, and thermal conductivity."""
    t_f = t_film
    density = (1.293 - 1.525e-4 * elevation + 6.379e-9 * (elevation ** 2)) / (1.0 + 0.00367 * t_f)
    viscosity = (1.458e-6 * ((t_f + 273.15) ** 1.5)) / (t_f + 273.15 + 110.4)
    k_air = 2.424e-2 + 7.477e-5 * t_f - 4.407e-9 * (t_f ** 2)
    return density, viscosity, k_air

def forced_convective_cooling(t_c, t_a, v_wind, phi_deg, d_m, elevation=0.0):
    """Calculate forced convective heat loss q_c (W/m) per IEEE 738-2012."""
    if v_wind <= 0.0:
        return 0.0

    t_f = (t_c + t_a) / 2.0
    density, viscosity, k_air = calculate_air_properties(t_f, elevation)
    
    phi_rad = math.radians(phi_deg)
    k_angle = 1.194 - math.cos(phi_rad) + 0.194 * math.cos(2 * phi_rad) + 0.368 * math.sin(2 * phi_rad)
    reynolds = (d_m * v_wind * density) / viscosity
    
    qc1 = k_angle * (1.01 + 1.35 * (reynolds ** 0.52)) * k_air * (t_c - t_a)
    qc2 = k_angle * (0.754 * (reynolds ** 0.6)) * k_air * (t_c - t_a)
    return max(qc1, qc2)

def natural_convective_cooling(t_c, t_a, d_m, elevation=0.0):
    """Calculate natural convective heat loss q_cn (W/m) per IEEE 738-2012."""
    t_f = (t_c + t_a) / 2.0
    density, viscosity, _ = calculate_air_properties(t_f, elevation)
    q_cn = 3.645 * (density ** 0.5) * (d_m ** 0.75) * ((t_c - t_a) ** 1.25)
    return q_cn

def radiative_cooling(t_c, t_a, d_m, emissivity=0.5):
    """Calculate radiative heat loss q_r (W/m)."""
    stefan_boltzmann = 5.670374419e-8
    return math.pi * d_m * emissivity * stefan_boltzmann * ((t_c + 273.15)**4 - (t_a + 273.15)**4)

def solar_heat_gain(d_m, solar_radiation=1000.0, absorptivity=0.5):
    """Calculate solar heat gain q_s (W/m)."""
    return absorptivity * solar_radiation * d_m

def calculate_ampacity(t_c, t_a, v_wind, phi_deg, solar_rad, d_m, r_tc, emissivity=0.5, absorptivity=0.5, elevation=0.0):
    """Compute Maximum Current Limit (Ampacity I_max) in Amperes."""
    q_cf = forced_convective_cooling(t_c, t_a, v_wind, phi_deg, d_m, elevation)
    q_cn = natural_convective_cooling(t_c, t_a, d_m, elevation)
    q_c = max(q_cf, q_cn)
    
    q_r = radiative_cooling(t_c, t_a, d_m, emissivity)
    q_s = solar_heat_gain(d_m, solar_rad, absorptivity)
    
    heat_dissipated = q_c + q_r - q_s
    if heat_dissipated <= 0:
        return 0.0
        
    return math.sqrt(heat_dissipated / r_tc)