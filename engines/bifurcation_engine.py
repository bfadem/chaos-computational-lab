from __future__ import annotations

from typing import Callable, Iterable, List, Tuple

import numpy as np
from scipy.integrate import solve_ivp


Array = np.ndarray
RhsFn = Callable[[float, Array, object], Array]
ValueFn = Callable[[Array, Array], Array]


def _set_param(params: object, name: str, value: float) -> None:
    if isinstance(params, dict):
        params[name] = float(value)
    else:
        setattr(params, name, float(value))


def _build_t_eval(t_span: Tuple[float, float], dt: float) -> Array:
    t0, t1 = float(t_span[0]), float(t_span[1])
    if dt <= 0:
        raise ValueError("dt must be positive.")
    n = int(np.floor((t1 - t0) / dt)) + 1
    return t0 + dt * np.arange(n, dtype=float)


def local_maxima(values: Array) -> Array:
    if values.size < 3:
        return np.array([], dtype=float)
    mid = values[1:-1]
    mask = (mid > values[:-2]) & (mid > values[2:])
    return mid[mask]


def bifurcation_sweep(
    rhs: RhsFn,
    params: object,
    param_name: str,
    param_values: Iterable[float],
    y0: Array,
    t_span: Tuple[float, float],
    *,
    dt: float | None = None,
    t_eval: Array | None = None,
    transient_fraction: float = 0.5,
    value_fn: ValueFn | None = None,
    method: str = "BDF",
    rtol: float = 1e-6,
    atol: float = 1e-9,
    max_step: float | None = None,
) -> List[Tuple[float, float]]:
    """
    Generic bifurcation sweep using solve_ivp with continuation.

    - Sweeps `param_name` over `param_values`.
    - Uses final state from each integration as the next initial condition.
    - Discards a transient fraction of samples.
    - Returns a list of (param, value) pairs.

    If `value_fn` is None, returns the first state component samples after transient.
    """
    if t_eval is None:
        if dt is None:
            raise ValueError("Provide dt or t_eval.")
        t_eval = _build_t_eval(t_span, dt)
    else:
        t_eval = np.asarray(t_eval, dtype=float)
        if t_eval.size < 2:
            raise ValueError("t_eval must include at least two time points.")

    if not (0.0 <= transient_fraction < 1.0):
        raise ValueError("transient_fraction must be in [0, 1).")

    if value_fn is None:
        value_fn = lambda t, y: y[0]

    pairs: List[Tuple[float, float]] = []

    y_init = np.asarray(y0, dtype=float)
    cut = int(np.floor(transient_fraction * t_eval.size))
    t_keep = t_eval[cut:]

    for pv in param_values:
        _set_param(params, param_name, pv)

        sol = solve_ivp(
            fun=lambda t, y: rhs(t, y, params),
            t_span=(float(t_span[0]), float(t_span[1])),
            y0=y_init,
            method=method,
            t_eval=t_eval,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
        )
        if not sol.success:
            continue

        y_keep = sol.y[:, cut:]
        values = np.asarray(value_fn(t_keep, y_keep), dtype=float).ravel()

        if values.size:
            pv_f = float(pv)
            pairs.extend((pv_f, float(v)) for v in values.tolist())

        y_init = sol.y[:, -1]

    return pairs
