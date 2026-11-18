from dataclasses import replace

import numpy as np
import pytest

from f1a.lapsim import CarParams, aero_from_components, sensitivity, solve_lap


def oval(step):
    # Half-lap symmetry and a pi turn per half close this synthetic oval.
    length = 5000
    s = np.arange(0, length, step)
    shape = np.exp(12*(np.cos(4*np.pi*s/length)-1))
    return s, (2*np.pi/length)*shape/shape.mean()


def test_power_drag_terminal_speed():
    car = CarParams(cla_m2=0, v_max_mps=np.inf)
    s = np.arange(0, 5000, 5.)
    lap = solve_lap(s, np.zeros_like(s), car)
    expected = (car.power_kW*1000/(.5*car.rho_kg_m3*car.cda_m2))**(1/3)
    np.testing.assert_allclose(lap.v, expected, rtol=1e-12)
    assert lap.total == pytest.approx(5000/expected)


@pytest.mark.parametrize("cla", [0., 4.5])
def test_steady_circle(cla):
    radius = 55.
    s = np.linspace(0, 2*np.pi*radius, 101)
    car = CarParams(cla_m2=cla, cda_m2=0)
    lap = solve_lap(s, np.full(100, 1/radius), car)
    expected = np.sqrt(car.mu*9.81/(1/radius-car.mu*.5*car.rho_kg_m3*cla/car.mass_kg))
    np.testing.assert_allclose(lap.v, expected, rtol=1e-12)
    assert lap.total == pytest.approx(s[-1]/expected)
    assert len(lap.t) == len(s) and lap.t[0] == 0 and np.all(np.diff(lap.t)>0)


def test_mirror_seam_and_resolution():
    car = CarParams()
    s, k = oval(5)
    lap = solve_lap(s, k, car)
    np.testing.assert_array_equal(lap.v, solve_lap(s, -k, car).v)
    shifted = solve_lap(s, np.roll(k, 130), car)
    np.testing.assert_allclose(shifted.v, np.roll(lap.v, 130))
    fine = solve_lap(*oval(2.5), car)
    assert abs(lap.total/fine.total-1) < .001


def test_mass_power_and_derivative():
    track, car = oval(5), CarParams()
    for field, step, sign in [('mass_kg', 5., 1), ('power_kW', 10., -1)]:
        value = getattr(car, field)
        times = [solve_lap(*track, replace(car, **{field: value+i*step})).total for i in (-1,0,1)]
        assert np.all(sign*np.diff(times)>0)
        assert sensitivity(car, track, field, step) == pytest.approx((times[2]-times[0])/(2*step))


def test_nonuniform_finish_and_invalid_inputs():
    s = [0, 3, 7, 12]
    lap = solve_lap(s, [0,0,0], CarParams(v_max_mps=30))
    np.testing.assert_allclose(lap.t, np.array(s)/30)
    with pytest.raises(ValueError, match="nonuniform"):
        solve_lap(s[:-1], [0,0,0], CarParams())
    with pytest.raises(ValueError):
        sensitivity(CarParams(), oval(5), 'mass_kg', 0)


def test_component_map():
    base = aero_from_components()
    assert base['cla_m2'] == pytest.approx(2.9)
    assert base['cda_m2'] == pytest.approx(.8)
    assert base['front_share'] == pytest.approx(.43951724137931036)
    assert aero_from_components({'rear': {'installed': False}})['extra_mass_kg'] == -10
    angled = aero_from_components({'rear': {'angle': 18}})
    assert angled['cla_m2'] > base['cla_m2'] and angled['front_share'] < base['front_share']


def test_circle_with_drag_consumes_longitudinal_grip():
    car = CarParams()
    k = 1/55
    drag = .5*car.rho_kg_m3*car.cda_m2/car.mass_kg
    load = .5*car.rho_kg_m3*car.cla_m2/car.mass_kg
    expected = np.sqrt(car.mu*9.81/(np.hypot(k, drag)-car.mu*load))
    lap = solve_lap(np.linspace(0, 2*np.pi/k, 101), np.full(100,k), car)
    np.testing.assert_allclose(lap.v, expected, rtol=1e-9)


def test_straight_power_sensitivity_against_analytic_derivative():
    car = CarParams(cla_m2=0, v_max_mps=np.inf)
    track = (np.arange(0, 5000, 5.), np.zeros(1000))
    top = (1000*car.power_kW/(.5*car.rho_kg_m3*car.cda_m2))**(1/3)
    expected = -(5000/top)/(3*car.power_kW)
    assert sensitivity(car, track, 'power_kW', .5) == pytest.approx(expected, rel=1e-6)


def test_brakes_before_corner_across_finish_line():
    # A speed-limited arc starts at node zero. A forward-only solver would
    # cross the seam at full speed and instantaneously slow down there.
    s = np.arange(0, 1000, 5.)
    k = np.zeros_like(s)
    k[:30] = 1/40
    car = CarParams(cla_m2=0, cda_m2=0, brake_max_mps2=5, v_max_mps=70)
    lap = solve_lap(s, k, car)
    assert lap.v[0] < lap.v[-1] < lap.v[-2] < lap.v[-3]
    # With no drag, braking cannot exceed the mechanical deceleration limit.
    decel = (lap.v**2 - np.roll(lap.v**2, -1))/(2*5)
    assert decel.max() == pytest.approx(car.brake_max_mps2)
