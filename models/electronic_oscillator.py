from __future__ import annotations

import numpy as np


def _get_param(params, name: str) -> float:
    if isinstance(params, dict):
        return float(params[name])
    return float(getattr(params, name))


def electronic_oscillator(t: float, y: np.ndarray, params) -> np.ndarray:
    """
    Dimensionless electronic oscillator (Tamaševičius et al., Eq. 4).

    State: y = [x, y, z]
      xdot = y
      ydot = a*y - x - z
      eps*zdot = b + y - c*(exp(z) - 1)
    """
    x, yv, z = y

    a = _get_param(params, "a")
    b = _get_param(params, "b")
    c = _get_param(params, "c")
    eps = _get_param(params, "eps")

    z_clip = np.clip(z, -50.0, 50.0)
    ez = np.exp(z_clip)

    xdot = yv
    ydot = a * yv - x - z
    zdot = (b + yv - c * (ez - 1.0)) / eps

    return np.array([xdot, ydot, zdot], dtype=float)
