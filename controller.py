import numpy as np
from numpy.typing import ArrayLike

from racetrack import RaceTrack

_dt = 0.1

_D_KP = 12.0
_D_KI = 0.2
_D_KD = 0.5

_BASE_LOOKAHEAD = 7.0
_GAIN_LOOKAHEAD = 0.15
_MIN_LOOKAHEAD = 5.0
_MAX_LOOKAHEAD = 15.0

_prev_delta_err = 0.0
_int_delta_err = 0.0

# time-based blending globals (controller-only)
_FADE_TIME = 2.0
_sim_time = 0.0

_track_cache: dict[int, dict[str, np.ndarray]] = {}


def _wrap_angle(a: float) -> float:
    return float(np.arctan2(np.sin(a), np.cos(a)))


def _blend_factor(t: float) -> float:
    if t <= 0.0:
        return 0.0
    if t >= _FADE_TIME:
        return 1.0
    return t / _FADE_TIME


def _resample_path(path: np.ndarray, n: int) -> np.ndarray:
    m = path.shape[0]
    if m == n:
        return path
    idx = np.linspace(0.0, float(m - 1), n)
    base_idx = np.arange(m, dtype=float)
    x = np.interp(idx, base_idx, path[:, 0])
    y = np.interp(idx, base_idx, path[:, 1])
    return np.column_stack((x, y))

def _reference_polyline(rt: RaceTrack, t: float) -> np.ndarray:
    center = np.asarray(rt.centerline, dtype=float)
    base = center[1:-1]

    alpha = _blend_factor(t)

    if hasattr(rt, "raceline") and getattr(rt, "raceline") is not None:
        race = np.asarray(rt.raceline, dtype=float)
        race = _resample_path(race, base.shape[0])
        return (1.0 - alpha) * base + alpha * race

    return base

def _track_data(rt: RaceTrack, t: float) -> dict[str, np.ndarray]:
    key = id(rt)

    # once fade is done, we can safely cache the final blended track
    if t >= _FADE_TIME:
        cached = _track_cache.get(key)
        if cached is not None:
            return cached

    pts = _reference_polyline(rt, t)
    n = pts.shape[0]

    seg = np.diff(pts, axis=0, append=pts[0:1])
    heading = np.arctan2(seg[:, 1], seg[:, 0])

    curv = np.zeros(n)
    for i in range(n):
        ip = (i - 1) % n
        inx = (i + 1) % n
        ds = float(np.linalg.norm(pts[inx] - pts[ip]))
        if ds < 1e-6:
            curv[i] = 0.0
        else:
            dpsi = _wrap_angle(heading[inx] - heading[ip])
            curv[i] = dpsi / ds

    data = {"path": pts, "heading": heading, "curvature": curv}

    if t >= _FADE_TIME:
        _track_cache[key] = data

    return data


def _nearest_index(path: np.ndarray, pos: np.ndarray) -> int:
    d = np.linalg.norm(path - pos[None, :], axis=1)
    return int(np.argmin(d))


def _lookahead_index(path: np.ndarray, start_idx: int, distance: float) -> int:
    n = path.shape[0]
    acc = 0.0
    i = start_idx
    while acc < distance:
        j = (i + 1) % n
        step = float(np.linalg.norm(path[j] - path[i]))
        if step < 1e-6:
            break
        acc += step
        i = j
    return i


def controller(
    state: ArrayLike, parameters: ArrayLike, racetrack: RaceTrack
) -> ArrayLike:
    global _sim_time

    state = np.asarray(state, dtype=float)
    parameters = np.asarray(parameters, dtype=float)

    # time in this controller module only
    t = _sim_time
    _sim_time += _dt

    x, y = state[0], state[1]
    v = state[3]
    phi = state[4]

    trk = _track_data(racetrack, t)
    path = trk["path"]
    curv_path = trk["curvature"]

    pos = np.array([x, y], dtype=float)

    idx_near = _nearest_index(path, pos)

    look_dist = _BASE_LOOKAHEAD + _GAIN_LOOKAHEAD * max(v, 0.0)
    look_dist = float(np.clip(look_dist, _MIN_LOOKAHEAD, _MAX_LOOKAHEAD))

    idx_look = _lookahead_index(path, idx_near, look_dist)
    ref_pt = path[idx_look]

    dx = ref_pt[0] - x
    dy = ref_pt[1] - y

    c = float(np.cos(phi))
    s = float(np.sin(phi))

    x_rel = c * dx + s * dy
    y_rel = -s * dx + c * dy

    if x_rel < 1.0:
        x_rel = 1.0

    Ld = float(np.hypot(x_rel, y_rel))
    wb = float(parameters[0])

    kappa_geom = 2.0 * y_rel / max(Ld * Ld, 1e-4)
    delta_ref = float(np.arctan(wb * kappa_geom))

    delta_ref = float(np.clip(delta_ref, parameters[1], parameters[4]))

    kappa_track = float(curv_path[idx_look])

    v_min = float(parameters[2])
    v_max = float(parameters[5])

    straight_cap = min(v_max, 85.0)

    # --- look ahead for upcoming corners ---
    HORIZON_DIST = 80.0
    a_y_max_local = 12.0
    a_y_max_far = 18.0

    n = path.shape[0]
    acc = 0.0
    i = idx_look
    max_kappa_ahead = abs(kappa_track)

    while acc < HORIZON_DIST:
        j = (i + 1) % n
        step = float(np.linalg.norm(path[j] - path[i]))
        if step < 1e-6:
            break
        acc += step
        i = j
        max_kappa_ahead = max(max_kappa_ahead, abs(float(curv_path[i])))

    if max_kappa_ahead > 1e-4:
        v_far = float(np.sqrt(max(a_y_max_far / max_kappa_ahead, 0.0)))
    else:
        v_far = straight_cap

    if abs(kappa_track) > 1e-4:
        v_local = float(np.sqrt(max(a_y_max_local / abs(kappa_track), 0.0)))
    else:
        v_local = straight_cap

    e_lat = float(y_rel)
    v_local /= (1.0 + 0.1 * abs(e_lat))

    v_curve = min(v_local, v_far, straight_cap)

    v_ref = float(np.clip(v_curve, v_min + 1.0, straight_cap))

    return np.array([delta_ref, v_ref], dtype=float)


def lower_controller(
    state: ArrayLike, desired: ArrayLike, parameters: ArrayLike
) -> ArrayLike:
    global _prev_delta_err, _int_delta_err

    state = np.asarray(state, dtype=float)
    desired = np.asarray(desired, dtype=float)
    parameters = np.asarray(parameters, dtype=float)

    assert desired.shape == (2,)

    delta = float(state[2])
    v = float(state[3])

    delta_ref = float(desired[0])
    v_ref = float(desired[1])

    delta_err = _wrap_angle(delta_ref - delta)
    _int_delta_err += delta_err * _dt
    delta_dot = (delta_err - _prev_delta_err) / _dt
    _prev_delta_err = delta_err

    v_delta = (
        _D_KP * delta_err
        + _D_KI * _int_delta_err
        + _D_KD * delta_dot
    )

    v_delta = float(np.clip(v_delta, parameters[7], parameters[9]))

    v_err = v_ref - v
    a = 15.0 * v_err
    a = float(np.clip(a, parameters[8], parameters[10]))

    return np.array([v_delta, a], dtype=float)
