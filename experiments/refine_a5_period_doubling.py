#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Tuple

import numpy as np

# Ensure repo root is on sys.path when running as a script.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from legacy.zoom_only_electronic_oscillator_fast import (  # noqa: E402
    ModelParams,
    simulate_and_collect_maxima_events,
)


@dataclass(frozen=True)
class PeriodSettings:
    b: float = 30.0
    c: float = 4e-9
    eps: float = 0.13
    y0: Tuple[float, float, float] = (0.1, 0.0, 0.0)
    method: str = "BDF"
    rtol: float = 1e-9
    atol: float = 1e-12
    max_step: float | None = 2.0
    # Increase integration time compared to earlier runs.
    theta_transient: float = 6000.0
    theta_sample: float = 6000.0
    # Require at least 200 oscillations before keeping last 64 maxima.
    min_total_maxima: int = 264
    keep_last_maxima: int = 64
    cluster_tol: float = 1e-5


def _cluster_count(values: np.ndarray, tol: float) -> int:
    if values.size == 0:
        return 0
    vals = np.sort(values.astype(float))
    count = 1
    last = vals[0]
    for v in vals[1:]:
        if abs(v - last) > tol:
            count += 1
            last = v
    return count


def compute_period(a: float, settings: PeriodSettings = PeriodSettings()) -> int:
    """
    Integrate long enough to remove transients, detect maxima via events,
    keep the last 64 maxima (after ensuring at least 200 oscillations),
    and return the number of distinct clustered maxima.
    """
    p = ModelParams(a=float(a), b=settings.b, c=settings.c, eps=settings.eps)
    y0 = np.array(settings.y0, dtype=float)

    theta_sample = settings.theta_sample
    maxima: np.ndarray

    for _ in range(4):
        maxima, _ = simulate_and_collect_maxima_events(
            p=p,
            theta_transient=settings.theta_transient,
            theta_sample=theta_sample,
            y0=y0,
            nmax_keep=max(settings.min_total_maxima, settings.keep_last_maxima),
            method=settings.method,
            rtol=settings.rtol,
            atol=settings.atol,
            max_step=settings.max_step,
        )

        if maxima.size >= settings.min_total_maxima:
            break
        theta_sample *= 2.0

    if maxima.size < settings.min_total_maxima:
        raise RuntimeError(
            f"Not enough maxima collected ({maxima.size}); "
            f"increase theta_sample beyond {theta_sample:.1f}."
        )

    maxima = maxima[-settings.keep_last_maxima :]
    return _cluster_count(maxima, tol=settings.cluster_tol)


def main() -> None:
    a_low = 0.193
    a_high = 0.196

    while (a_high - a_low) > 1e-6:
        a_mid = 0.5 * (a_low + a_high)
        period = compute_period(a_mid)
        if period <= 8:
            a_low = a_mid
        else:
            a_high = a_mid

    a5 = 0.5 * (a_low + a_high)
    period_low = compute_period(a_low)
    period_high = compute_period(a_high)

    a3 = 0.1878
    a4 = 0.1930
    delta_4 = (a4 - a3) / (a5 - a4)

    print(f"a_5 ≈ {a5:.10f}")
    print(f"period at a_low  ({a_low:.10f}) = {period_low}")
    print(f"period at a_high ({a_high:.10f}) = {period_high}")
    print(f"delta_4 ≈ {delta_4:.6f}")


if __name__ == "__main__":
    main()
