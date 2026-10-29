"""Solar geometry and the sunshine a horizontal conductor actually absorbs.

Why this module exists: ERA5/Open-Meteo radiation is measured on a HORIZONTAL surface, but a
conductor is a horizontal cylinder whose projected width depends on where the sun is relative to
the line. Treating horizontal irradiance as if it fell on the projected area understates solar
heating in winter (low sun) and gets the line-bearing dependence wrong.

Direct beam:  q ~ I_B * sin(theta), with cos(theta) = cos(H) * cos(Zs - Zl)
              H = solar elevation, Zs = sun azimuth, Zl = line azimuth (all from north).
Ground and sky terms follow CIGRE TB 601 eq. (9) (default) or the isotropic CIGRE TB 207 form.

Sun position is the NOAA low-precision series (about 0.3 deg), enough for heat-gain purposes.
Times are naive UTC, like everything else in this repo.
"""

import numpy as np
import pandas as pd

MIN_SIN_ELEVATION = 0.05   # below ~3 deg the DNI = direct_horizontal / sin(H) division is unstable
MAX_DNI_WM2 = 1200.0       # physical ceiling for DNI at Irish latitudes; clips quantisation noise


def sun_position(times, lat_deg, lon_deg):
    """Solar elevation and azimuth in degrees (azimuth clockwise from north, 0-360).

    times: naive UTC timestamps (the instant the sun is evaluated at, not an hour label)."""
    t = pd.DatetimeIndex(times)
    hours = t.hour.to_numpy() + t.minute.to_numpy() / 60.0 + t.second.to_numpy() / 3600.0
    days_in_year = np.where(t.is_leap_year, 366.0, 365.0)
    gamma = 2.0 * np.pi / days_in_year * (t.dayofyear.to_numpy() - 1 + (hours - 12.0) / 24.0)

    eqtime_min = 229.18 * (0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma)
                           - 0.014615 * np.cos(2 * gamma) - 0.040849 * np.sin(2 * gamma))
    decl = (0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
            - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma)
            - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma))

    solar_time_min = hours * 60.0 + eqtime_min + 4.0 * lon_deg      # UTC, so no time-zone term
    hour_angle = np.deg2rad(solar_time_min / 4.0 - 180.0)
    lat = np.deg2rad(lat_deg)

    sin_elev = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(hour_angle)
    elevation = np.rad2deg(np.arcsin(np.clip(sin_elev, -1.0, 1.0)))
    # atan2 form is measured from south, positive towards west; +180 puts it clockwise from north
    az_from_south = np.arctan2(np.sin(hour_angle),
                               np.cos(hour_angle) * np.sin(lat) - np.tan(decl) * np.cos(lat))
    azimuth = (np.rad2deg(az_from_south) + 180.0) % 360.0
    return elevation, azimuth


def sin_incidence_angle(elevation_deg, azimuth_deg, line_azimuth_deg):
    """sin(theta): theta is the angle between the sun's rays and the conductor axis.

    1 when the rays are perpendicular to the line (full diameter is lit), 0 when they run
    along it. Only the axis matters, so the line azimuth is equivalent to bearing + 180."""
    cos_theta = np.cos(np.deg2rad(elevation_deg)) * np.cos(np.deg2rad(np.asarray(azimuth_deg) - line_azimuth_deg))
    return np.sqrt(np.clip(1.0 - cos_theta ** 2, 0.0, 1.0))


def dni_from_direct_horizontal(direct_horizontal, elevation_deg):
    """Direct normal irradiance (W/m2) from direct irradiance on a horizontal plane.

    Prefer a measured/archived DNI column when you have one (ingest_weather.py requests it).
    This derivation divides by sin(elevation), so the divisor is floored at MIN_SIN_ELEVATION:
    very low sun therefore UNDER-estimates DNI instead of blowing up."""
    sin_h = np.maximum(np.sin(np.deg2rad(elevation_deg)), MIN_SIN_ELEVATION)
    dni = np.asarray(direct_horizontal, dtype=float) / sin_h
    return np.where(np.asarray(elevation_deg) > 0.0, np.clip(dni, 0.0, MAX_DNI_WM2), 0.0)


def effective_irradiance(dni, dhi, elevation_deg, azimuth_deg, line_azimuth_deg,
                         albedo=0.2, diffuse_model="cigre601"):
    """Irradiance (W/m2) to feed the heat balance as 'solar radiation on the projected area'.

    cigre601:   I_T = I_B (sin(theta) + 0.5 pi F sin(H)) + I_d (1 + 0.5 pi F)        (TB 601, eq. 9)
    isotropic:  I_T = I_B (sin(theta) + 0.5 pi F sin(H)) + 0.5 pi I_d (1 + F)        (TB 207)
    The second treats sky light as uniform over the hemisphere, which a horizontal cylinder
    integrates to pi/2 times the horizontal diffuse irradiance; the pair is a modelling band.
    F is the ground albedo. dni and dhi are normal-beam and horizontal-diffuse irradiance."""
    if diffuse_model not in ("cigre601", "isotropic"):
        raise ValueError("diffuse_model must be 'cigre601' or 'isotropic'")
    elev = np.asarray(elevation_deg, dtype=float)
    dni = np.where(elev > 0.0, np.asarray(dni, dtype=float), 0.0)
    dhi = np.asarray(dhi, dtype=float)
    sin_h = np.maximum(np.sin(np.deg2rad(elev)), 0.0)
    sin_theta = sin_incidence_angle(elev, azimuth_deg, line_azimuth_deg)

    direct = dni * (sin_theta + 0.5 * np.pi * albedo * sin_h)
    if diffuse_model == "cigre601":
        diffuse = dhi * (1.0 + 0.5 * np.pi * albedo)
    else:
        diffuse = dhi * 0.5 * np.pi * (1.0 + albedo)
    return direct + diffuse
