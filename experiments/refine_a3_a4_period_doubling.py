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
    theta_transient: float = 6000.0
    theta_sample: float = 6000.0
    min_total_maxima: int = 364  # >= 300 oscillations + keep last 64
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
    keep the last 64 maxima (after ensuring >=300 oscillations),
    and return the number of distinct clustered maxima.
    """
    p = ModelParams(a=float(a), b=settings.b, c=settings.c, eps=settings.eps)
    y0 = np.array(settings.y0, dtype=float)

    theta_sample = settings.theta_sample
    maxima: np.ndarray

    for _ in range(5):
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


def bracket_transition(a_min: float, a_max: float, target_period: int) -> Tuple[float, float]:
    a_values = np.linspace(a_min, a_max, 50, dtype=float)
    periods = []
    for a in a_values:
        period = compute_period(a)
        periods.append(period)
        print(f"Scanning a={a:.8f}  period={period}", flush=True)

    for i in range(len(a_values) - 1):
        p0 = periods[i]
        p1 = periods[i + 1]
        if p0 <= target_period and p1 >= 2 * target_period:
            return float(a_values[i]), float(a_values[i + 1])

    raise RuntimeError("Failed to bracket transition in the provided interval.")


def refine_transition(a_low: float, a_high: float, target_period: int) -> float:
    while (a_high - a_low) > 1e-7:
        a_mid = 0.5 * (a_low + a_high)
        period = compute_period(a_mid)
        print(
            f"Refining: a_mid={a_mid:.10f}  period={period}  interval={a_high - a_low:.2e}",
            flush=True,
        )
        if period <= target_period:
            a_low = a_mid
        else:
            a_high = a_mid
    return 0.5 * (a_low + a_high)


def main() -> None:
    print("Bracketing a_3 ...", flush=True)
    a3_low, a3_high = bracket_transition(0.180, 0.190, target_period=4)
    print("Refining a_3 ...", flush=True)
    a3 = refine_transition(a3_low, a3_high, target_period=4)
    print(f"Final a_3 ≈ {a3:.10f}", flush=True)
    print("", flush=True)

    print("Bracketing a_4 ...", flush=True)
    a4_low, a4_high = bracket_transition(0.190, 0.195, target_period=8)
    print("Refining a_4 ...", flush=True)
    a4 = refine_transition(a4_low, a4_high, target_period=8)
    print(f"Final a_4 ≈ {a4:.10f}", flush=True)
    print("", flush=True)

    a2 = 0.167
    delta_3 = (a3 - a2) / (a4 - a3)
    print(f"delta_3 ≈ {delta_3:.6f}", flush=True)


if __name__ == "__main__":
    main()
