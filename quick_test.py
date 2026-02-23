import numpy as np
import matplotlib.pyplot as plt

from models.electronic_oscillator import electronic_oscillator
from engines.bifurcation_engine import compute_bifurcation


def main():

    # Small parameter sweep (fast sanity check, not high-accuracy)
    param_values = np.linspace(0.1669, 0.1671, 12)

    # Short integration window for speed
    t_span = (0.0, 80.0)
    dt = 0.05

    base_params = {
        "a": float(param_values[0]),
        "b": 30.0,
        "c": 4e-9,
        "eps": 0.13,
    }

    print("Running quick test...")

    results = compute_bifurcation(
        rhs=electronic_oscillator,
        params=base_params,
        param_name="a",
        param_values=param_values,
        y0=[0.1, 0.0, 0.0],
        t_span=t_span,
        dt=dt,
        transient_fraction=0.5,
    )

    print("Computation complete.")
    print(f"Generated {len(results)} bifurcation points.")

    # Simple scatter
    if results:
        p_vals, x_vals = zip(*results)
        plt.figure(figsize=(6, 4))
        plt.scatter(p_vals, x_vals, s=2)
        plt.xlabel("control parameter a")
        plt.ylabel("local maxima of x (x_m)")
        plt.title("Quick Bifurcation (paper params)")
        plt.show()


if __name__ == "__main__":
    main()
