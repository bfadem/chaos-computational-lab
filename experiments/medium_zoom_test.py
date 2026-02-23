# experiments/medium_zoom_test.py
# Note: experiment scripts prioritize accuracy over speed (longer t_span, finer dt).

import numpy as np
import matplotlib.pyplot as plt

from models.electronic_oscillator import electronic_oscillator
from engines.bifurcation_engine import compute_bifurcation


def main():

    param_values = np.linspace(0.16695, 0.16708, 150)

    t_span = (0.0, 120.0)
    dt = 0.05

    params = {
        "a": 0.16695,   # will be swept
        "b": 30.0,
        "c": 4e-9,
        "eps": 0.13,
    }

    print("Running medium zoom test...")

    pairs = compute_bifurcation(
        rhs=electronic_oscillator,
        params=params,
        param_name="a",
        param_values=param_values,
        y0=np.array([0.1, 0.0, 0.0]),
        t_span=t_span,
        dt=dt,
        transient_fraction=0.5,
        method="BDF",
        rtol=1e-6,
        atol=1e-9,
        verbose=True,
    )

    print(f"Generated {len(pairs)} points.")

    if pairs:
        p_vals, x_vals = zip(*pairs)
        plt.figure(figsize=(6,4))
        plt.scatter(p_vals, x_vals, s=2)
        plt.title("Medium Zoom Test")
        plt.show()


if __name__ == "__main__":
    main()
