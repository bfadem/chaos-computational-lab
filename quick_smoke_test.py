from __future__ import annotations

import time

import numpy as np

from engines.bifurcation_engine import compute_bifurcation
from models.electronic_oscillator import electronic_oscillator


def main() -> None:
    # Small parameter sweep for a fast smoke test.
    param_values = np.linspace(0.85, 0.90, 6)

    # Short integration window and coarse step for speed.
    t_span = (0.0, 8.0)
    dt = 0.05

    params = {
        "a": 0.85,  # swept below
        "b": 0.5,
        "c": 0.3,
        "eps": 0.1,
    }

    print("[SMOKE] running quick bifurcation sweep...", flush=True)
    start = time.perf_counter()

    pairs = compute_bifurcation(
        rhs=electronic_oscillator,
        params=params,
        param_name="a",
        param_values=param_values,
        y0=np.array([0.1, 0.0, 0.0]),
        t_span=t_span,
        dt=dt,
        transient_fraction=0.3,
        method="RK45",
        rtol=1e-4,
        atol=1e-7,
        max_step=0.2,
    )

    elapsed = time.perf_counter() - start
    print(f"[SMOKE] done in {elapsed:.3f}s", flush=True)
    print(f"[SMOKE] generated {len(pairs)} points", flush=True)

    if pairs:
        preview = pairs[:5]
        print("[SMOKE] preview:", preview, flush=True)


if __name__ == "__main__":
    main()
