import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "physics"))
import pytest
from dlr.ieee738 import calculate_ampacity, calculate_air_properties


def test_air_density_regression():
    """Guards against constant typos: ~1.03 kg/m^3 at 70 degC film temperature, sea level."""
    density, viscosity, k_air = calculate_air_properties(70.0, 0.0)
    assert density == pytest.approx(1.029, abs=0.01)
    assert viscosity == pytest.approx(2.04e-5, rel=0.03)
    assert k_air == pytest.approx(0.0295, rel=0.03)


def test_ieee738_drake_benchmark():
    """
    Drake ACSR, Tc=100C, Ta=40C, 0.61 m/s perpendicular, 1000 W/m2, R(100C)=9.39e-5 ohm/m.
    The 985-1015 A window is a placeholder: replace it with the published worked-example
    value from IEEE 738 once checked against the standard.
    """
    i_calc = calculate_ampacity(
        t_c=100.0, t_a=40.0, v_wind=0.61, phi_deg=90.0,
        solar_rad=1000.0, d_m=0.02814, r_tc=9.39e-5,
        emissivity=0.5, absorptivity=0.5, elevation=0.0,
    )
    assert 985.0 <= i_calc <= 1015.0, f"{i_calc:.2f} A outside benchmark window"


def test_more_wind_means_more_ampacity():
    lo = calculate_ampacity(75.0, 25.0, 1.0, 90.0, 1000.0, 0.02814, 8.688e-5)
    hi = calculate_ampacity(75.0, 25.0, 5.0, 90.0, 1000.0, 0.02814, 8.688e-5)
    assert hi > lo
    
