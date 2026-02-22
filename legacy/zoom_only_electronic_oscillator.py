#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
zoom_only_electronic_oscillator.py

Zoom-only bifurcation scan for the simple electronic oscillator:

    xdot = y
    ydot = a*y - x - z
    eps*zdot = b + y - c*(exp(z) - 1)

This script:
- DOES NOT run the coarse full scan.
- Runs a dense scan only in [a_lo, a_hi].
- Writes:
    zoom_<a_lo>_<a_hi>_bifurcation.csv
    zoom_<a_lo>_<a_hi>_bifurcation.png
- Optional: overlays the existing coarse bifurcation.csv for context.

Usage examples:
  python zoom_only_electronic_oscillator.py
  python zoom_only_electronic_oscillator.py --a_lo 0.15 --a_hi 0.19 --npts 5000
  python zoom_only_electronic_oscillator.py --no_overlay
"""

from __future__ import annotations

import os
import time
import argparse
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# -----------------------------
# Model + simulation utilities
# -----------------------------

@dataclass
class ModelParams:
    a: float
    b: float
    c: float
    eps: float


def rhs(theta: float, state: np.ndarray, p: ModelParams) -> np.ndarray:
    x, y, z = state
    dx = y
    dy = p.a * y - x - z
    dz = (p.b + y - p.c * (np.exp(z) - 1.0)) / p.eps
    return np.array([dx, dy, dz], dtype=float)


def simulate_and_collect_maxima(
    *,
    p: ModelParams,
    theta_transient: float,
    theta_sample: float,
    dt_sample: float,
    y0: np.ndarray,
    nmax_keep: int = 80,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Integrate through transients, then sample for maxima of x using sign changes in y:
      maxima when y crosses + -> -.

    Returns:
      maxima_x : up to nmax_keep local maxima of x
      y_end    : final state to use as next initial condition (continuation in 'a')
    """
    # 1) burn-in / transient
    sol1 = solve_ivp(
        fun=lambda t, s: rhs(t, s, p),
        t_span=(0.0, float(theta_transient)),
        y0=y0,
        method="RK45",
        rtol=rtol,
        atol=atol,
        dense_output=False,
    )
    yT = sol1.y[:, -1].astype(float)

    # 2) sampling window at uniform times
    nsteps = int(np.floor(theta_sample / dt_sample))
    if nsteps < 5:
        raise ValueError("theta_sample/dt_sample too small to detect maxima reliably.")

    t_eval = np.linspace(0.0, float(theta_sample), nsteps + 1)
    sol2 = solve_ivp(
        fun=lambda t, s: rhs(t, s, p),
        t_span=(0.0, float(theta_sample)),
        y0=yT,
        t_eval=t_eval,
        method="RK45",
        rtol=rtol,
        atol=atol,
        dense_output=False,
    )

    x = sol2.y[0, :]
    y = sol2.y[1, :]
    y_end = sol2.y[:, -1].astype(float)

    # detect maxima: y[i-1] > 0 and y[i] <= 0 (sign change)
    idx = np.where((y[:-1] > 0.0) & (y[1:] <= 0.0))[0]
    if idx.size == 0:
        return np.array([], dtype=float), y_end

    # refine maxima estimate: take the larger of the two x samples around the crossing
    mx = np.maximum(x[idx], x[idx + 1])

    # keep last nmax_keep maxima (late-time behavior)
    if mx.size > nmax_keep:
        mx = mx[-nmax_keep:]

    return mx.astype(float), y_end


def make_dense_a_values(a_lo: float, a_hi: float, npts: int) -> np.ndarray:
    return np.linspace(float(a_lo), float(a_hi), int(npts))


def save_scatter_csv(path: str, A: np.ndarray, X: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("a,x_max\n")
        for a, x in zip(A, X):
            f.write(f"{a:.10f},{x:.10f}\n")


def load_scatter_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data[:, 0].astype(float), data[:, 1].astype(float)


def add_vertical_guides(ax, a_guides: List[float], label_prefix: str = "a=") -> None:
    y0, y1 = ax.get_ylim()
    for a in a_guides:
        ax.axvline(a, linewidth=1)
        ax.text(a, y1, f"{label_prefix}{a:.6f}", rotation=90, va="top", ha="right", fontsize=8)


def annotate_xticks_every(ax, step: float) -> None:
    lo, hi = ax.get_xlim()
    start = np.floor(lo / step) * step
    ticks = np.arange(start, hi + step, step)
    ax.set_xticks(ticks)
    ax.tick_params(axis="x", labelrotation=45)


def run_zoom_scan(
    *,
    a_lo: float,
    a_hi: float,
    npts: int,
    b: float,
    eps: float,
    c: float,
    theta_transient: float,
    theta_sample: float,
    dt_sample: float,
    seed_state: np.ndarray,
    nmax_keep: int,
) -> Tuple[np.ndarray, np.ndarray]:
    a_values = make_dense_a_values(a_lo, a_hi, npts=npts)

    A_list: List[float] = []
    X_list: List[float] = []

    y0 = seed_state.copy()
    t0 = time.time()

    for i, a in enumerate(a_values):
        p = ModelParams(a=float(a), b=float(b), c=float(c), eps=float(eps))
        maxima, y0 = simulate_and_collect_maxima(
            p=p,
            theta_transient=theta_transient,
            theta_sample=theta_sample,
            dt_sample=dt_sample,
            y0=y0,
            nmax_keep=nmax_keep,
        )
        if maxima.size > 0:
            A_list.extend([a] * maxima.size)
            X_list.extend(maxima.tolist())

        if (i + 1) % max(1, npts // 10) == 0:
            elapsed = time.time() - t0
            print(f"[ZOOM] {i+1}/{npts} a={a:.6f} elapsed={elapsed:.1f}s", flush=True)

    return np.array(A_list, dtype=float), np.array(X_list, dtype=float)


def plot_zoom(
    *,
    A_zoom: np.ndarray,
    X_zoom: np.ndarray,
    out_png: str,
    title: str,
    a_guides: List[float] | None,
    xtick_step: float | None,
    overlay_A: np.ndarray | None = None,
    overlay_X: np.ndarray | None = None,
) -> None:
    fig = plt.figure(figsize=(10, 5))
    ax = plt.gca()

    # overlay coarse (optional) behind
    if overlay_A is not None and overlay_X is not None and overlay_A.size > 0:
        ax.plot(overlay_A, overlay_X, linestyle="None", marker=",")  # coarse cloud

    ax.plot(A_zoom, X_zoom, linestyle="None", marker=",")  # zoom cloud
    ax.set_xlabel("control parameter a")
    ax.set_ylabel("local maxima of x (x_m)")
    ax.set_title(title)

    if xtick_step is not None and xtick_step > 0:
        annotate_xticks_every(ax, step=xtick_step)

    fig.tight_layout()

    if a_guides:
        add_vertical_guides(ax, a_guides, label_prefix="a=")

    fig.tight_layout()
    fig.savefig(out_png, dpi=250)
    print(f"[PLOT] wrote {out_png}", flush=True)
    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a_lo", type=float, default=0.15)
    ap.add_argument("--a_hi", type=float, default=0.19)
    ap.add_argument("--npts", type=int, default=5000)

    # model parameters (your current choices)
    ap.add_argument("--b", type=float, default=30.0)
    ap.add_argument("--eps", type=float, default=0.13)
    ap.add_argument("--c", type=float, default=4e-9)

    # sampling controls
    ap.add_argument("--theta_transient", type=float, default=2500.0)
    ap.add_argument("--theta_sample", type=float, default=2500.0)
    ap.add_argument("--dt_sample", type=float, default=0.10)
    ap.add_argument("--nmax_keep", type=int, default=80)

    # plotting
    ap.add_argument("--base_csv", type=str, default="bifurcation.csv")
    ap.add_argument("--no_overlay", action="store_true")
    ap.add_argument("--xtick_step", type=float, default=0.001)
    ap.add_argument("--guides", type=float, nargs="*", default=[0.1668, 0.1862])

    args = ap.parse_args()

    a_lo, a_hi = float(args.a_lo), float(args.a_hi)
    if a_hi <= a_lo:
        raise SystemExit("Need a_hi > a_lo")

    tag = f"{a_lo:.4f}_{a_hi:.4f}".replace(".", "p")
    out_csv = f"zoom_{tag}_bifurcation.csv"
    out_png = f"zoom_{tag}_bifurcation.png"

    # Optional coarse overlay
    overlay_A = overlay_X = None
    if (not args.no_overlay) and os.path.exists(args.base_csv):
        overlay_A, overlay_X = load_scatter_csv(args.base_csv)
        # crop overlay to zoom window for cleanliness
        m = (overlay_A >= a_lo) & (overlay_A <= a_hi)
        overlay_A, overlay_X = overlay_A[m], overlay_X[m]
        print(f"[OVERLAY] loaded {args.base_csv} cropped to {overlay_A.size} pts", flush=True)

    # Cache: if zoom CSV exists, don't re-integrate
    if os.path.exists(out_csv):
        print(f"[CACHE] found {out_csv}; loading (no re-sim).", flush=True)
        A_zoom, X_zoom = load_scatter_csv(out_csv)
    else:
        print(f"[RUN] zoom scan a in [{a_lo}, {a_hi}] with npts={args.npts}", flush=True)
        seed_state = np.array([0.1, 0.0, 0.0], dtype=float)
        A_zoom, X_zoom = run_zoom_scan(
            a_lo=a_lo,
            a_hi=a_hi,
            npts=int(args.npts),
            b=float(args.b),
            eps=float(args.eps),
            c=float(args.c),
            theta_transient=float(args.theta_transient),
            theta_sample=float(args.theta_sample),
            dt_sample=float(args.dt_sample),
            seed_state=seed_state,
            nmax_keep=int(args.nmax_keep),
        )
        save_scatter_csv(out_csv, A_zoom, X_zoom)
        print(f"[DONE] wrote {out_csv} with {A_zoom.size} rows", flush=True)

    title = f"Zoom bifurcation: maxima of x vs a,  a in [{a_lo:.6f}, {a_hi:.6f}]"
    plot_zoom(
        A_zoom=A_zoom,
        X_zoom=X_zoom,
        out_png=out_png,
        title=title,
        a_guides=list(args.guides) if args.guides else None,
        xtick_step=float(args.xtick_step) if args.xtick_step else None,
        overlay_A=overlay_A,
        overlay_X=overlay_X,
    )


if __name__ == "__main__":
    main()
