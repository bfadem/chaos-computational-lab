# experiments/single_parameter_inspect.py
# Note: experiment scripts prioritize accuracy over speed (longer t_span, finer dt).

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.electronic_oscillator import electronic_oscillator


def main() -> None:
    # ----- Parameters (match paper defaults) -----
    a = 0.170
    params = {
        "a": a,
        "b": 30.0,
        "c": 4e-9,
        "eps": 0.13,
    }

    # ----- Initial condition and time grid -----
    y0 = np.array([0.1, 0.0, 0.0])
    t_span = (0.0, 200.0)
    dt = 0.05

    # Explicit time sampling for solve_ivp
    t_eval = np.arange(t_span[0], t_span[1] + dt, dt, dtype=float)

    # ----- Integrate the ODE directly -----
    sol = solve_ivp(
        fun=lambda t, y: electronic_oscillator(t, y, params),
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        method="BDF",
        rtol=1e-6,
        atol=1e-9,
    )

    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")

    # ----- Analyze the first state variable -----
    x = sol.y[0]

    # Remove the first 50% of samples as transient
    cut = int(0.5 * x.size)
    t_keep = sol.t[cut:]
    x_keep = x[cut:]

    # Compute local maxima using neighbor comparison
    if x_keep.size < 3:
        peaks = np.array([], dtype=float)
        peak_times = np.array([], dtype=float)
    else:
        left = x_keep[:-2]
        mid = x_keep[1:-1]
        right = x_keep[2:]
        peak_mask = (mid > left) & (mid > right)
        peaks = mid[peak_mask]
        peak_times = t_keep[1:-1][peak_mask]

    # ----- Print peak summary -----
    print(f"Total peaks found: {peaks.size}")
    print("First 10 peaks:", peaks[:10].tolist())

    # ----- Plot time series and mark peaks -----
    plt.figure(figsize=(8, 4))
    plt.plot(t_keep, x_keep, label="x(t) after transient", linewidth=1.0)
    if peaks.size:
        plt.plot(peak_times, peaks, "ro", markersize=3, label="local maxima")
    plt.title(f"Electronic Oscillator at a = {a}")
    plt.xlabel("t")
    plt.ylabel("x")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
