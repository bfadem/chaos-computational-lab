from __future__ import annotations

# Note: experiment scripts prioritize accuracy over speed (longer t_span, finer dt).

import numpy as np
import matplotlib.pyplot as plt

from engines.bifurcation_engine import compute_bifurcation
from models.electronic_oscillator import electronic_oscillator


def main() -> None:
    a_lo = 0.16695
    a_hi = 0.16708
    a_values = np.linspace(a_lo, a_hi, 400)

    params = {
        "a": float(a_values[0]),
        "b": 30.0,
        "c": 4e-9,
        "eps": 0.13,
    }

    theta_transient = 2000.0
    theta_sample = 2000.0
    t_span = (0.0, theta_transient + theta_sample)
    transient_fraction = theta_transient / (theta_transient + theta_sample)

    pairs = compute_bifurcation(
        rhs=electronic_oscillator,
        params=params,
        param_name="a",
        param_values=a_values,
        y0=np.array([0.1, 0.0, 0.0], dtype=float),
        t_span=t_span,
        dt=0.12,
        transient_fraction=transient_fraction,
        state_index=0,
        method="BDF",
        rtol=1e-6,
        atol=1e-9,
        max_step=2.0,
    )

    if pairs:
        A = np.array([p for p, _ in pairs], dtype=float)
        X = np.array([v for _, v in pairs], dtype=float)
    else:
        A = np.array([], dtype=float)
        X = np.array([], dtype=float)

    plt.figure(figsize=(10, 5))
    plt.plot(A, X, linestyle="None", marker=",")
    plt.xlabel("control parameter a")
    plt.ylabel("local maxima of x (x_m)")
    plt.title(f"Zoom bifurcation: maxima of x vs a  (a in [{a_lo:.6f}, {a_hi:.6f}])")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
