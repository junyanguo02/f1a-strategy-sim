"""Closed-loop point-mass solver derived from external/formula-lab/work/aero-model.js."""
from dataclasses import dataclass, fields, replace

import numpy as np


G = 9.81  # m/s²


@dataclass(frozen=True)
class CarParams:
    mass_kg: float = 800.0         # car + driver + fuel, total at this instant
    power_kW: float = 750.0        # constant effective wheel power
    mu: float = 1.65               # effective tyre friction coefficient
    cla_m2: float = 4.5            # positive downforce coefficient × area
    cda_m2: float = 1.0            # drag coefficient × area
    brake_max_mps2: float = 55.0   # mechanical braking limit, excluding drag
    v_max_mps: float = 110.0       # independent speed ceiling, +inf disables it
    front_share: float = 0.46      # front fraction of aero downforce
    static_front_share: float = 0.46  # front fraction of static weight
    rho_kg_m3: float = 1.2         # air density

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name == "v_max_mps" and value == np.inf:
                continue
            if not np.isfinite(value):
                raise ValueError(f"{field.name} must be finite")
        if min(self.mass_kg, self.power_kW, self.mu, self.brake_max_mps2,
               self.v_max_mps, self.rho_kg_m3) <= 0 or min(self.cla_m2, self.cda_m2) < 0:
            raise ValueError("Mass, power, friction, braking, speed and density must be positive; areas nonnegative")
        if not 0 < self.static_front_share < 1 or not 0 <= self.front_share <= 1:
            raise ValueError("Invalid front load fraction")


@dataclass(frozen=True)
class LapResult:
    v: np.ndarray  # m/s, segment starts (N)
    t: np.ndarray  # s, segment boundaries including finish (N+1)
    total: float   # seconds


def aero_from_components(parts=None):
    """Illustrative JS aeroSetup map; angle in degrees, extra_mass_kg relative to all fitted."""
    parts = {} if parts is None else parts
    components = {
        "front": (.10, .62, 8, 1.06, 8, .028, .095),
        "rear": (.17, .86, 10, -.07, 12, .039, .11),
        "floor": (.08, 1.22, 22, .48, 0, .055, .035),
    }
    if set(parts) - components.keys():
        raise ValueError("Unknown aero component")
    drag, down, moment, mass = .45, .20, .20 * .46, 0
    for name, (base_drag, base_load, kg, front, angle, slope, polar) in components.items():
        part = parts.get(name, {})
        if not part.get("installed", True):
            mass -= kg
            continue
        setting = part.get("angle", angle)
        if not np.isfinite(setting):
            raise ValueError("Component angle must be finite")
        load = max(.05, base_load + slope * (setting - angle))
        drag += base_drag + polar * (load**2 - base_load**2)
        down += load
        moment += load * front
    return dict(cda_m2=drag, cla_m2=down, extra_mass_kg=mass, front_share=moment/down)


def solve_lap(s, curvature, car):
    """Solve periodic flying lap on N nodes. s can contain N+1 edges, or N uniform starts."""
    s = np.asarray(s, dtype=float)
    k = np.asarray(curvature, dtype=float)
    if k.ndim != 1 or s.ndim != 1 or len(k) < 3 or len(s) not in (len(k), len(k)+1):
        raise ValueError("Expected N curvatures and N uniform nodes or N+1 distance boundaries")
    ds = np.diff(s)
    if not np.isfinite(k).all() or not np.isfinite(ds).all() or np.any(ds <= 0):
        raise ValueError("Expected finite curvature and strictly increasing distance")
    if len(s) == len(k):
        if not np.allclose(ds, ds[0], rtol=1e-7, atol=1e-9):
            raise ValueError("For a nonuniform grid supply the finish distance as the last boundary")
        ds = np.append(ds, ds[0])

    load = .5 * car.rho_kg_m3 * car.cla_m2 / car.mass_kg
    drag = .5 * car.rho_kg_m3 * car.cda_m2 / car.mass_kg
    lateral_load = load * min(car.front_share/car.static_front_share,
                              (1-car.front_share)/(1-car.static_front_share))
    power = 1000 * car.power_kW / car.mass_kg
    top = min(car.v_max_mps, np.cbrt(power / drag) if drag else np.inf)
    denom = abs(k) - car.mu*lateral_load
    q = np.full(len(k), np.inf)
    np.divide(car.mu*G, denom, out=q, where=denom > 0)
    q = np.minimum(q, top**2)
    if not np.isfinite(q).all():
        raise ValueError("Unbounded speed envelope; supply finite v_max_mps or positive cda_m2")

    # The force balance is evaluated at the segment midpoint. Work in v² so
    # dx/ds = 2a, where x = v². Signed curvature interpolates through chicanes.
    km = abs((k + np.roll(k, -1)) * .5)

    def available(x):
        used = np.minimum(1, km*x / (car.mu*(G + lateral_load*x)))
        return car.mu*(G + load*x)*np.sqrt((1-used)*(1+used))

    # At fixed midpoint curvature a full-throttle trajectory cannot cross its
    # equilibrium speed. Near lateral saturation the square root is stiff:
    # an unconstrained midpoint predictor can oscillate across that equilibrium.
    # Both tyre utilisation terms under steady drag grow monotonically with v².
    lo = np.zeros_like(q)
    hi = np.full_like(q, q.max())
    for _ in range(48):
        mid = .5*(lo+hi)
        feasible = (km*mid / (car.mu*(G+lateral_load*mid)))**2 + (
            drag*mid / (car.mu*(G+load*mid)))**2 <= 1
        lo = np.where(feasible, mid, lo)
        hi = np.where(feasible, hi, mid)
    equilibrium = .5*(lo+hi)  # top power/speed already included in q.max()

    # Synchronous forward/backward relaxations wrap through the finish. Each
    # iteration propagates constraints by one node in each direction.
    for _ in range(2*len(k)+64):
        end = np.roll(q, -1)
        drive = np.minimum(available(q), power/np.sqrt(q)) - drag*q
        lower, upper = np.minimum(q, equilibrium), np.maximum(q, equilibrium)
        mid_drive = np.clip(q + ds*drive, lower, upper)
        accel = np.minimum(available(mid_drive), power/np.sqrt(np.maximum(mid_drive, 1e-15))) - drag*mid_drive
        driven = np.where(mid_drive == equilibrium, equilibrium,
                          np.clip(q + 2*ds*accel, lower, upper))
        brake = np.minimum(car.brake_max_mps2, available(end)) + drag*end
        mid_brake = end + ds*brake
        decel = np.minimum(car.brake_max_mps2, available(mid_brake)) + drag*mid_brake
        updated = np.minimum(q, np.minimum(np.roll(driven, 1), end + 2*ds*decel))
        if np.any(updated <= 0):
            raise ValueError("Distance step too large for drag integration; refine the grid")
        change = np.max(q-updated)
        q = updated
        if change < 1e-8:
            break
    else:
        raise RuntimeError("Periodic speed constraints did not converge")
    v = np.sqrt(q)
    t = np.r_[0., np.cumsum(2*ds/(v + np.roll(v,-1)))]
    return LapResult(v=v, t=t, total=float(t[-1]))


def sensitivity(car, track, param, delta):
    """Central derivative in seconds per parameter unit. track = (s, curvature)."""
    if param not in CarParams.__dataclass_fields__ or not np.isfinite(delta) or delta <= 0:
        raise ValueError("Expected a CarParams field and a positive finite delta")
    value = getattr(car, param)
    if not np.isfinite(value):
        raise ValueError("Cannot perturb an infinite parameter")
    plus = solve_lap(*track, replace(car, **{param: value + delta}))
    minus = solve_lap(*track, replace(car, **{param: value - delta}))
    return (plus.total - minus.total)/(2*delta)
