import itertools

import pytest

from dlr.ieee738 import (
    calculate_air_properties,
    calculate_ampacity,
    forced_convective_cooling,
    natural_convective_cooling,
    thermal_terms,
)

# Drake ACSR 26/7 at its 75 degC rating (illustrative project conductor)
D, R75 = 0.02814, 8.688e-5
CALM_AIR_REGRESSION = 659.18  # A, v = 0, Ta = 25 degC, 1000 W/m2


def ampacity(t_c=75.0, t_a=25.0, v=1.0, phi=90.0, sol=1000.0, r=R75):
    return calculate_ampacity(t_c, t_a, v, phi, sol, D, r)


# Constants 

def test_air_properties_at_70c():
    """Guards against constant typos (approximate dry-air values at 70 degC)."""
    density, viscosity, k_air = calculate_air_properties(70.0, 0.0)
    assert density == pytest.approx(1.029, abs=0.01)
    assert viscosity == pytest.approx(2.04e-5, rel=0.03)
    assert k_air == pytest.approx(0.0295, rel=0.03)


# The physics must close its own energy balance 

@pytest.mark.parametrize(
    "t_a, v, phi, sol",
    list(itertools.product([-5.0, 10.0, 30.0], [0.0, 0.3, 2.0, 12.0], [0.0, 45.0, 90.0], [0.0, 600.0])),
)
def test_heat_balance_closes(t_a, v, phi, sol):
    """At the returned current, heat in (I^2 R + q_s) must equal heat out (q_c + q_r)."""
    t_c = 75.0
    i = calculate_ampacity(t_c, t_a, v, phi, sol, D, R75)
    t = thermal_terms(t_c, t_a, v, phi, sol, D)
    assert i ** 2 * R75 + t["q_s"] == pytest.approx(t["q_c"] + t["q_r"], rel=1e-9)


# Physical monotonicity (each would fail if a sign or exponent were wrong) 

def test_more_wind_more_ampacity():
    assert ampacity(v=5.0) > ampacity(v=1.0) > ampacity(v=0.3)


def test_colder_air_more_ampacity():
    assert ampacity(t_a=5.0) > ampacity(t_a=25.0) > ampacity(t_a=35.0)


def test_sunshine_reduces_ampacity():
    assert ampacity(sol=0.0) > ampacity(sol=1000.0)


def test_higher_temperature_limit_more_ampacity():
    assert ampacity(t_c=100.0) > ampacity(t_c=75.0)


def test_higher_resistance_less_ampacity():
    assert ampacity(r=1.2e-4) < ampacity(r=R75)


def test_cross_wind_cools_more_than_parallel_wind():
    assert ampacity(phi=90.0) > ampacity(phi=45.0) > ampacity(phi=0.0)


@pytest.mark.parametrize("phi, k_expected", [(0.0, 0.388), (45.0, 0.8549), (90.0, 1.0)])
def test_wind_angle_factor_matches_standard(phi, k_expected):
    """K_angle cancels in the ratio of forced cooling to the perpendicular case, so this
    checks the angle polynomial alone. 0.388 (parallel) and 1.0 (perpendicular) are the
    standard's endpoint values; 45 degrees is the only angle that exercises the sin(2*phi) term."""
    ratio = (forced_convective_cooling(75.0, 25.0, 3.0, phi, D)
             / forced_convective_cooling(75.0, 25.0, 3.0, 90.0, D))
    assert ratio == pytest.approx(k_expected, rel=1e-3)


def test_ampacity_scales_as_inverse_sqrt_resistance():
    assert ampacity(r=4 * R75) == pytest.approx(ampacity() / 2.0, rel=1e-12)


# Calm air and edge cases 

def test_natural_convection_scales_as_dt_to_the_1_25():
    """Same film temperature (so same air properties), different temperature difference:
    q_cn must scale exactly as dT**1.25."""
    a = natural_convective_cooling(60.0, 40.0, D)   # film 50 degC, dT 20
    b = natural_convective_cooling(70.0, 30.0, D)   # film 50 degC, dT 40
    assert b / a == pytest.approx(2.0 ** 1.25, rel=1e-12)


def test_natural_convection_scales_as_d_to_the_0_75():
    assert natural_convective_cooling(75.0, 25.0, 2 * D) / natural_convective_cooling(75.0, 25.0, D) \
        == pytest.approx(2.0 ** 0.75, rel=1e-12)


def test_calm_air_regression_value():
    """REGRESSION ONLY: pins the calm-air (natural convection) rating of the Drake case."""
    assert ampacity(v=0.0) == pytest.approx(CALM_AIR_REGRESSION, abs=0.01)


def test_high_wind_regression_value():
    """REGRESSION ONLY: at 10 m/s the second (high Reynolds number) convection expression
    governs, which the Drake case below never reaches."""
    assert ampacity(v=10.0) == pytest.approx(2008.86, abs=0.01)


def test_zero_wind_falls_back_to_natural_convection():
    assert forced_convective_cooling(75.0, 25.0, 0.0, 90.0, D) == 0.0
    assert natural_convective_cooling(75.0, 25.0, D) > 0.0
    assert ampacity(v=0.0) > 0.0
    # any real wind can only help: the max() rule keeps the curve from dropping below calm air
    assert ampacity(v=0.05) >= ampacity(v=0.0)


@pytest.mark.parametrize("t_c, t_a", [(75.0, 75.0), (75.0, 80.0), (30.0, 40.0)])
def test_ambient_at_or_above_limit_gives_zero_not_complex(t_c, t_a):
    """Regression: (t_c - t_a) ** 1.25 used to return a complex number when t_a > t_c."""
    i = calculate_ampacity(t_c, t_a, 2.0, 90.0, 800.0, D, R75)
    assert isinstance(i, float)
    assert i == 0.0


@pytest.mark.parametrize("d, r", [(0.0, R75), (D, 0.0), (-0.01, R75)])
def test_invalid_conductor_is_rejected(d, r):
    with pytest.raises(ValueError):
        calculate_ampacity(75.0, 25.0, 1.0, 90.0, 0.0, d, r)


# Regression against the current implementation

def test_drake_case_regression_value():
    """REGRESSION ONLY, not an independent benchmark. This pins the current output so an
    accidental change to a constant or exponent is caught. Replace or supplement it with a
    published IEEE 738-2012 / CIGRE TB 601 worked example (see README: Validation)."""
    i = calculate_ampacity(100.0, 40.0, 0.61, 90.0, 1000.0, D, 9.39e-5)
    assert i == pytest.approx(992.47, abs=0.01)
