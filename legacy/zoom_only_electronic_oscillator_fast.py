#!/usr/bin/env python3
"""
zoom_only_electronic_oscillator_fast.py

Zoom-only bifurcation scan for the simple electronic oscillator:

    xdot = y
    ydot = a*y - x - z
    eps*zdot = b + y - c*(exp(z) - 1)

FAST approach:
- Detect maxima using events: y(t)=0 crossing with direction -1 (max of x since xdot=y).
- Avoids huge t_eval arrays (which kill performance).
- Uses stiff-friendly solver by default (Radau).

Outputs:
- zoom_<a_lo>_<a_hi>_bifurcation.csv  with header a,x_max
- zoom_<a_lo>_<a_hi>_bifurcation.png

Optional:
- Overlay an existing coarse bifurcation.csv cropped to [a_lo, a_hi].

Spyder tip:
Use Run -> Configure per file... -> Command line options:
  --a_lo 0.15 --a_hi 0.19 --npts 2000 --print_every 50
"""

from __future__ import annotations

import os
import time
import argparse
from dataclasses import dataclass
from typing import Tuple, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# -----------------------------
# Model
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


# -----------------------------
# IO
# -----------------------------

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


# -----------------------------
# Plot helpers
# -----------------------------

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


def plot_zoom(
    *,
    A_zoom: np.ndarray,
    X_zoom: np.ndarray,
    out_png: str,
    title: str,
    a_lo: float,
    a_hi: float,
    a_guides: Optional[List[float]] = None,
    xtick_step: Optional[float] = None,
    overlay_A: Optional[np.ndarray] = None,
    overlay_X: Optional[np.ndarray] = None,
) -> None:
    fig = plt.figure(figsize=(10, 5))
    ax = plt.gca()

    # overlay coarse (optional)
    if overlay_A is not None and overlay_X is not None and overlay_A.size > 0:
        ax.plot(overlay_A, overlay_X, linestyle="None", marker=",")

    ax.plot(A_zoom, X_zoom, linestyle="None", marker=",")

    ax.set_xlim(a_lo, a_hi)
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


# -----------------------------
# Core: fast maxima finder via events
# -----------------------------

def simulate_and_collect_maxima_events(
    *,
    p: ModelParams,
    theta_transient: float,
    theta_sample: float,
    y0: np.ndarray,
    nmax_keep: int,
    method: str,
    rtol: float,
    atol: float,
    max_step: Optional[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast maxima detection using events:
      maxima of x occur when y(t)=0 crossing with direction -1 (since xdot = y).

    Returns:
      maxima_x : up to nmax_keep local maxima of x (late-time)
      y_end    : final state to use as continuation initial condition
    """

    def y_zero_event(t, s):
        return s[1]  # y

    y_zero_event.direction = -1
    y_zero_event.terminal = False

    # 1) burn transient
    sol1 = solve_ivp(
        fun=lambda t, s: rhs(t, s, p),
        t_span=(0.0, float(theta_transient)),
        y0=y0,
        method=method,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )
    if not sol1.success:
        raise RuntimeError(f"Transient solve failed: {sol1.message}")

    yT = sol1.y[:, -1].astype(float)

    # 2) sample with events
    sol2 = solve_ivp(
        fun=lambda t, s: rhs(t, s, p),
        t_span=(0.0, float(theta_sample)),
        y0=yT,
        method=method,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        events=y_zero_event,
    )
    if not sol2.success:
        raise RuntimeError(f"Sample solve failed: {sol2.message}")

    y_end = sol2.y[:, -1].astype(float)

    if sol2.y_events is None or len(sol2.y_events) == 0:
        return np.array([], dtype=float), y_end

    ev = sol2.y_events[0]
    if ev is None or ev.size == 0:
        return np.array([], dtype=float), y_end

    # Each event state is [x, y, z], and y=0 at the event
    x_events = ev[:, 0].astype(float)

    # Keep last nmax_keep (late-time)
    if x_events.size > nmax_keep:
        x_events = x_events[-nmax_keep:]

    return x_events, y_end


# -----------------------------
# Zoom scan with progress + ETA
# -----------------------------

def make_dense_a_values(a_lo: float, a_hi: float, npts: int) -> np.ndarray:
    return np.linspace(float(a_lo), float(a_hi), int(npts))


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
    seed_state: np.ndarray,
    nmax_keep: int,
    method: str,
    rtol: float,
    atol: float,
    max_step: Optional[float],
    print_every: int,
) -> Tuple[np.ndarray, np.ndarray]:
    a_values = make_dense_a_values(a_lo, a_hi, npts=npts)

    A_list: List[float] = []
    X_list: List[float] = []

    y0 = seed_state.copy()

    t_start = time.time()
    t_last = t_start

    # Always print immediately
    print(f"[ZOOM] starting: a in [{a_lo:.6f}, {a_hi:.6f}], npts={npts}", flush=True)
    print(f"[ZOOM] solver={method} rtol={rtol:g} atol={atol:g} "
          f"theta_transient={theta_transient:g} theta_sample={theta_sample:g} "
          f"nmax_keep={nmax_keep} max_step={max_step}", flush=True)

    for i, a in enumerate(a_values):
        p = ModelParams(a=float(a), b=float(b), c=float(c), eps=float(eps))
        maxima, y0 = simulate_and_collect_maxima_events(
            p=p,
            theta_transient=theta_transient,
            theta_sample=theta_sample,
            y0=y0,
            nmax_keep=nmax_keep,
            method=method,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
        )

        if maxima.size > 0:
            A_list.extend([a] * maxima.size)
            X_list.extend(maxima.tolist())

        # progress report
        if (i == 0) or ((i + 1) % print_every == 0) or (i + 1 == npts):
            now = time.time()
            elapsed = now - t_start
            chunk = now - t_last
            t_last = now

            done = i + 1
            rate = done / max(elapsed, 1e-9)  # a-points per second
            remain = npts - done
            eta = remain / max(rate, 1e-9)

            print(
                f"[ZOOM] {done:5d}/{npts}  a={a:.6f}  "
                f"elapsed={elapsed:7.1f}s  last_chunk={chunk:6.1f}s  "
                f"rate={rate:6.2f} a/s  ETA={eta/60:6.1f} min  "
                f"rows={len(A_list)}",
                flush=True
            )

    return np.array(A_list, dtype=float), np.array(X_list, dtype=float)


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()

    # window + resolution
    ap.add_argument("--a_lo", type=float, default=0.185)
    ap.add_argument("--a_hi", type=float, default=0.194)
    ap.add_argument("--npts", type=int, default=2000)

    # model parameters
    ap.add_argument("--b", type=float, default=30.0)
    ap.add_argument("--eps", type=float, default=0.13)
    ap.add_argument("--c", type=float, default=4e-9)

    # time controls
    ap.add_argument("--theta_transient", type=float, default=2000.0)
    ap.add_argument("--theta_sample", type=float, default=2000.0)
    ap.add_argument("--nmax_keep", type=int, default=80)

    # solver controls
    ap.add_argument("--method", type=str, default="Radau", choices=["RK45", "Radau", "BDF", "DOP853"])
    ap.add_argument("--rtol", type=float, default=1e-7)
    ap.add_argument("--atol", type=float, default=1e-9)
    ap.add_argument("--max_step", type=float, default=0.5)  # keep reasonable; set 0 to disable

    # reporting
    ap.add_argument("--print_every", type=int, default=50)

    # overlay + plotting
    ap.add_argument("--base_csv", type=str, default="bifurcation.csv")
    ap.add_argument("--no_overlay", action="store_true")
    ap.add_argument("--xtick_step", type=float, default=0.001)
    ap.add_argument("--guides", type=float, nargs="*", default=[0.167, 0.188])

    args = ap.parse_args()

    a_lo, a_hi = float(args.a_lo), float(args.a_hi)
    if a_hi <= a_lo:
        raise SystemExit("Need a_hi > a_lo")

    npts = int(args.npts)
    if npts < 5:
        raise SystemExit("npts too small")

    max_step = float(args.max_step)
    if max_step <= 0:
        max_step = None

    # output naming
    tag = f"{a_lo:.6f}_{a_hi:.6f}".replace(".", "p")
    out_csv = f"zoom_{tag}_bifurcation.csv"
    out_png = f"zoom_{tag}_bifurcation.png"

    # overlay coarse scan (optional)
    overlay_A = overlay_X = None
    if (not args.no_overlay) and os.path.exists(args.base_csv):
        overlay_A, overlay_X = load_scatter_csv(args.base_csv)
        m = (overlay_A >= a_lo) & (overlay_A <= a_hi)
        overlay_A, overlay_X = overlay_A[m], overlay_X[m]
        print(f"[OVERLAY] loaded {args.base_csv} cropped to {overlay_A.size} pts", flush=True)

    # cache behavior
    if os.path.exists(out_csv):
        print(f"[CACHE] found {out_csv}; loading (no re-sim).", flush=True)
        A_zoom, X_zoom = load_scatter_csv(out_csv)
    else:
        seed_state = np.array([0.1, 0.0, 0.0], dtype=float)
        A_zoom, X_zoom = run_zoom_scan(
            a_lo=a_lo,
            a_hi=a_hi,
            npts=npts,
            b=float(args.b),
            eps=float(args.eps),
            c=float(args.c),
            theta_transient=float(args.theta_transient),
            theta_sample=float(args.theta_sample),
            seed_state=seed_state,
            nmax_keep=int(args.nmax_keep),
            method=str(args.method),
            rtol=float(args.rtol),
            atol=float(args.atol),
            max_step=max_step,
            print_every=max(1, int(args.print_every)),
        )
        save_scatter_csv(out_csv, A_zoom, X_zoom)
        print(f"[DONE] wrote {out_csv} with {A_zoom.size} rows", flush=True)

    title = f"Zoom bifurcation (events): maxima of x vs a,  a in [{a_lo:.6f}, {a_hi:.6f}]"
    plot_zoom(
        A_zoom=A_zoom,
        X_zoom=X_zoom,
        out_png=out_png,
        title=title,
        a_lo=a_lo,
        a_hi=a_hi,
        a_guides=list(args.guides) if args.guides else None,
        xtick_step=float(args.xtick_step) if args.xtick_step else None,
        overlay_A=overlay_A,
        overlay_X=overlay_X,
    )


if __name__ == "__main__":
    main()
