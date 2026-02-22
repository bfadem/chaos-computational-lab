#!/usr/bin/env python3
"""
zoom_only_electronic_oscillator_refine.py

Zoom + refinement tools for the simple electronic oscillator (Tamaševičius-type model):

    xdot = y
    ydot = a*y - x - z
    eps*zdot = b + y - c*(exp(z) - 1)

FAST bifurcation observable:
- local maxima of x as a function of control parameter a
- maxima detected by events: y(t)=0 crossing with direction -1 (since xdot=y)

What this script can do
-----------------------
(1) ZOOM SCAN (like your current fast script)
    - Scan a in [a_lo, a_hi] with npts points
    - For each a, integrate to remove transient, then collect maxima events
    - Save scatter CSV: a,x_max
    - Plot maxima vs a, with vertical guide lines at your a1..a4 estimates.

(2) REFINE (bisection on the ODE, period-count based)
    - Define "period-count" K(a) = number of distinct maxima levels in late-time data.
      (e.g., K=1,2,4,8,16,...)
    - Use a robust clustering tolerance in x to count distinct levels.
    - For each transition (1->2, 2->4, 4->8, 8->16), bracket and bisection-search a_n.

(3) QUICK BRACKETS FROM AN EXISTING SCAN CSV (no new ODE solves)
    - If you already have zoom_<...>_bifurcation.csv, estimate transition brackets
      by grouping points by a and counting distinct maxima levels per a.

Spyder tip
----------
Run -> Configure per file... -> Command line options, e.g.
  --a_lo 0.185 --a_hi 0.194 --npts 1200 --theta_transient 1500 --theta_sample 1500

Examples
--------
Zoom scan and plot with your current a1..a4:
  --mode scan --a_lo 0.185 --a_hi 0.194 --npts 2000

Refine a3 (4->8) and a4 (8->16) by bisection (ODE solves):
  --mode refine --targets a3 a4 --bracket_a3 0.1875 0.1880 --bracket_a4 0.1927 0.1932

Estimate brackets from an existing scan CSV (no new solves):
  --mode csv_brackets --in_csv zoom_0p185000_0p194000_bifurcation.csv

Notes on robustness near bifurcations
-------------------------------------
Near the boundary, transient times may need to be longer and clustering tol_x smaller.
For bisection refinement, this script defaults to a FIXED seed initial condition for each
a (no continuation), which is usually more robust near bifurcations. You can enable
continuation if you want, but it can introduce hysteresis-like artifacts.
"""

from __future__ import annotations

import os
import time
import argparse
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict

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

def add_vertical_guides(ax, guides: Dict[str, float], *, linewidth: float = 1.8) -> None:
    """Draw labeled vertical lines."""
    y0, y1 = ax.get_ylim()
    for name, a in guides.items():
        ax.axvline(a, linewidth=linewidth)
        ax.text(a, y1, f"{name}={a:.6f}", rotation=90, va="top", ha="right", fontsize=8)


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
    guides: Optional[Dict[str, float]] = None,
    xtick_step: Optional[float] = None,
    overlay_A: Optional[np.ndarray] = None,
    overlay_X: Optional[np.ndarray] = None,
) -> None:
    fig = plt.figure(figsize=(11, 5.5))
    ax = plt.gca()

    # overlay coarse (optional)
    if overlay_A is not None and overlay_X is not None and overlay_A.size > 0:
        ax.plot(overlay_A, overlay_X, linestyle="None", marker=",", alpha=0.55)

    ax.plot(A_zoom, X_zoom, linestyle="None", marker=",", alpha=0.85)

    ax.set_xlim(a_lo, a_hi)
    ax.set_xlabel("control parameter a")
    ax.set_ylabel("local maxima of x (x_m)")
    ax.set_title(title)

    if xtick_step is not None and xtick_step > 0:
        annotate_xticks_every(ax, step=xtick_step)

    # IMPORTANT: set ylim before guides so text goes at the top properly
    fig.tight_layout()
    if guides:
        add_vertical_guides(ax, guides)

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
# Scan
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
    continuation: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    If continuation=True, the final state at each a is used as the initial condition
    for the next a. This is fast and smooth, but can be less robust near bifurcations.
    """
    a_values = make_dense_a_values(a_lo, a_hi, npts=npts)

    A_list: List[float] = []
    X_list: List[float] = []

    y0_cont = seed_state.copy()

    t_start = time.time()
    t_last = t_start

    print(f"[SCAN] a in [{a_lo:.6f}, {a_hi:.6f}], npts={npts}", flush=True)
    print(f"[SCAN] solver={method} rtol={rtol:g} atol={atol:g} "
          f"theta_transient={theta_transient:g} theta_sample={theta_sample:g} "
          f"nmax_keep={nmax_keep} max_step={max_step} continuation={continuation}", flush=True)

    for i, a in enumerate(a_values):
        p = ModelParams(a=float(a), b=float(b), c=float(c), eps=float(eps))
        y0 = y0_cont if continuation else seed_state

        maxima, y_end = simulate_and_collect_maxima_events(
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

        if continuation:
            y0_cont = y_end

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
                f"[SCAN] {done:5d}/{npts}  a={a:.6f}  "
                f"elapsed={elapsed:7.1f}s  last_chunk={chunk:6.1f}s  "
                f"rate={rate:6.2f} a/s  ETA={eta/60:6.1f} min  "
                f"rows={len(A_list)}",
                flush=True
            )

    return np.array(A_list, dtype=float), np.array(X_list, dtype=float)


# -----------------------------
# Refinement: period-count via clustering + bisection
# -----------------------------

def cluster_levels(vals: np.ndarray, tol_x: float) -> Tuple[int, List[float]]:
    """
    Cluster nearby maxima values and return (count, centers).
    Simple 1D agglomeration: a new cluster starts when gap > tol_x.
    """
    v = np.sort(np.asarray(vals, dtype=float))
    if v.size == 0:
        return 0, []
    clusters: List[List[float]] = [[float(v[0])]]
    for x in v[1:]:
        if abs(x - clusters[-1][-1]) > tol_x:
            clusters.append([float(x)])
        else:
            clusters[-1].append(float(x))
    centers = [float(np.mean(c)) for c in clusters]
    return len(clusters), centers


def make_run_one_a_fn(
    *,
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
) :
    """
    Returns a function run_one_a(a)->np.ndarray of maxima_x values (last nmax_keep).
    Uses a FIXED seed_state each call (no continuation) for robustness.
    """
    def run_one_a(a: float) -> np.ndarray:
        p = ModelParams(a=float(a), b=float(b), c=float(c), eps=float(eps))
        maxima, _ = simulate_and_collect_maxima_events(
            p=p,
            theta_transient=theta_transient,
            theta_sample=theta_sample,
            y0=seed_state,
            nmax_keep=nmax_keep,
            method=method,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
        )
        return maxima
    return run_one_a


def period_count_for_a(
    run_one_a_fn,
    a: float,
    *,
    drop_first: int,
    tol_x: float,
) -> Tuple[int, List[float]]:
    """
    Compute K(a)=#distinct maxima levels from late-time maxima_x.
    drop_first removes a few maxima as extra transient insurance.
    """
    xm = np.asarray(run_one_a_fn(a), dtype=float)
    if xm.size == 0:
        return 0, []
    if drop_first > 0 and xm.size > drop_first:
        xm = xm[drop_first:]
    k, centers = cluster_levels(xm, tol_x=tol_x)
    return k, centers


def bisect_transition(
    run_one_a_fn,
    a_lo: float,
    a_hi: float,
    k_lo: int,
    k_hi: int,
    *,
    drop_first: int,
    tol_x: float,
    tol_a: float,
    max_iter: int = 50,
    verbose: bool = True,
) -> Tuple[float, float, float]:
    """
    Bisection for a boundary where K(a) transitions from k_lo to k_hi.

    Assumes:
      K(a_lo)=k_lo
      K(a_hi)=k_hi

    Returns:
      (a_lo_final, a_hi_final, a_mid)
    """
    k0, _ = period_count_for_a(run_one_a_fn, a_lo, drop_first=drop_first, tol_x=tol_x)
    k1, _ = period_count_for_a(run_one_a_fn, a_hi, drop_first=drop_first, tol_x=tol_x)
    if k0 != k_lo or k1 != k_hi:
        raise RuntimeError(f"Bad bracket: K(a_lo)={k0} (wanted {k_lo}), K(a_hi)={k1} (wanted {k_hi})")

    for it in range(max_iter):
        a_mid = 0.5 * (a_lo + a_hi)
        k_mid, _ = period_count_for_a(run_one_a_fn, a_mid, drop_first=drop_first, tol_x=tol_x)

        if verbose:
            print(f"[BISECT] it={it:02d}  a_lo={a_lo:.10f}  a_mid={a_mid:.10f}  a_hi={a_hi:.10f}  K_mid={k_mid}")

        if k_mid == k_lo:
            a_lo = a_mid
        elif k_mid == k_hi:
            a_hi = a_mid
        else:
            # In the messy boundary region, K may transiently look like 3,5,6, etc.
            # Conservative approach: keep the side that still has a guaranteed k_hi.
            a_hi = a_mid

        if (a_hi - a_lo) < tol_a:
            break

    return a_lo, a_hi, 0.5 * (a_lo + a_hi)


# -----------------------------
# CSV-based brackets (no new ODE solves)
# -----------------------------

def period_counts_from_scatter_csv(A: np.ndarray, X: np.ndarray, *, tol_x: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given scatter points (a_i, x_max_i) where each a repeats many times,
    group by a and compute K(a)=#distinct maxima levels.
    Returns:
      a_unique_sorted, K_sorted
    """
    # group by exact a values in the file (they are written with fixed decimals)
    a_vals = np.asarray(A, dtype=float)
    x_vals = np.asarray(X, dtype=float)

    uniq = np.unique(a_vals)
    K = np.zeros_like(uniq, dtype=int)

    for i, a in enumerate(uniq):
        xs = x_vals[a_vals == a]
        k, _ = cluster_levels(xs, tol_x=tol_x)
        K[i] = k

    order = np.argsort(uniq)
    return uniq[order], K[order]


def bracket_from_K_series(a_u: np.ndarray, K: np.ndarray, k_lo: int, k_hi: int) -> Optional[Tuple[float, float]]:
    """
    Find the first adjacent pair where K goes from k_lo to k_hi as a increases.
    Returns (a_left, a_right) or None if not found.
    """
    for i in range(len(a_u) - 1):
        if K[i] == k_lo and K[i + 1] == k_hi:
            return float(a_u[i]), float(a_u[i + 1])
    return None


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--mode", type=str, default="scan", choices=["scan", "refine", "csv_brackets"],
                    help="scan: run zoom scan; refine: bisection refinement via ODE; csv_brackets: estimate brackets from an existing scan CSV")

    # scan window + resolution
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
    ap.add_argument("--max_step", type=float, default=0.5)  # set <=0 to disable

    # scan behavior
    ap.add_argument("--print_every", type=int, default=50)
    ap.add_argument("--continuation", action="store_true",
                    help="In scan mode, use continuation initial conditions across a (faster, but can be less robust near bifurcations)")

    # plotting
    ap.add_argument("--base_csv", type=str, default="bifurcation.csv")
    ap.add_argument("--no_overlay", action="store_true")
    ap.add_argument("--xtick_step", type=float, default=0.001)

    # Your current bifurcation estimates (defaults set to what you wrote)
    ap.add_argument("--a1", type=float, default=0.10885)
    ap.add_argument("--a2", type=float, default=0.167)
    ap.add_argument("--a3", type=float, default=0.188)
    ap.add_argument("--a4", type=float, default=0.193)

    # refinement controls
    ap.add_argument("--targets", type=str, nargs="*", default=["a3", "a4"],
                    choices=["a1", "a2", "a3", "a4"],
                    help="Which bifurcations to refine (a1:1->2, a2:2->4, a3:4->8, a4:8->16)")

    ap.add_argument("--tol_x", type=float, default=1e-3,
                    help="Clustering tolerance in x for counting distinct maxima levels")
    ap.add_argument("--drop_first", type=int, default=20,
                    help="Drop first N maxima (extra transient insurance) before clustering")
    ap.add_argument("--tol_a", type=float, default=1e-6,
                    help="Stop bisection when bracket width < tol_a")
    ap.add_argument("--max_iter", type=int, default=50)

    # explicit brackets (recommended)
    ap.add_argument("--bracket_a1", type=float, nargs=2, default=None)
    ap.add_argument("--bracket_a2", type=float, nargs=2, default=None)
    ap.add_argument("--bracket_a3", type=float, nargs=2, default=None)
    ap.add_argument("--bracket_a4", type=float, nargs=2, default=None)

    # csv_brackets mode
    ap.add_argument("--in_csv", type=str, default=None,
                    help="Input CSV (a,x_max) to estimate brackets without new ODE solves")

    args = ap.parse_args()

    # solver max_step
    max_step = float(args.max_step)
    if max_step <= 0:
        max_step = None

    # common seed
    seed_state = np.array([0.1, 0.0, 0.0], dtype=float)

    # vertical guides (always include your current estimates)
    guides = {"a1": float(args.a1), "a2": float(args.a2), "a3": float(args.a3), "a4": float(args.a4)}

    # -------------------------
    # Mode: csv_brackets
    # -------------------------
    if args.mode == "csv_brackets":
        if not args.in_csv or not os.path.exists(args.in_csv):
            raise SystemExit("csv_brackets mode requires --in_csv <existing_scan.csv>")

        A, X = load_scatter_csv(args.in_csv)
        a_u, K = period_counts_from_scatter_csv(A, X, tol_x=float(args.tol_x))

        print(f"[CSV] unique a points: {a_u.size}")
        # report first few changes
        changes = np.where(np.diff(K) != 0)[0]
        print(f"[CSV] #K changes: {changes.size}")
        for idx in changes[:20]:
            print(f"  a={a_u[idx]:.10f} K={K[idx]}  ->  a={a_u[idx+1]:.10f} K={K[idx+1]}")

        transitions = {
            "a1": (1, 2),
            "a2": (2, 4),
            "a3": (4, 8),
            "a4": (8, 16),
        }
        for name in args.targets:
            k_lo, k_hi = transitions[name]
            br = bracket_from_K_series(a_u, K, k_lo, k_hi)
            if br is None:
                print(f"[CSV] {name}: could not find {k_lo}->{k_hi} transition in this CSV.")
            else:
                alo, ahi = br
                amid = 0.5*(alo+ahi)
                print(f"[CSV] {name} ({k_lo}->{k_hi}) bracket: [{alo:.10f}, {ahi:.10f}]  mid={amid:.10f}  halfwidth={(ahi-alo)/2:.2e}")

        # Optional quick plot of K(a)
        plt.figure(figsize=(10,4))
        plt.plot(a_u, K, linestyle="None", marker=".", markersize=2)
        plt.xlabel("a")
        plt.ylabel("K(a) = #distinct maxima levels (clustered)")
        plt.title("Period-count vs a inferred from scan CSV")
        plt.tight_layout()
        plt.show()
        return

    # -------------------------
    # Mode: scan
    # -------------------------
    if args.mode == "scan":
        a_lo, a_hi = float(args.a_lo), float(args.a_hi)
        if a_hi <= a_lo:
            raise SystemExit("Need a_hi > a_lo")

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

        # cache
        if os.path.exists(out_csv):
            print(f"[CACHE] found {out_csv}; loading (no re-sim).", flush=True)
            A_zoom, X_zoom = load_scatter_csv(out_csv)
        else:
            A_zoom, X_zoom = run_zoom_scan(
                a_lo=a_lo,
                a_hi=a_hi,
                npts=int(args.npts),
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
                continuation=bool(args.continuation),
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
            guides=guides,
            xtick_step=float(args.xtick_step) if args.xtick_step else None,
            overlay_A=overlay_A,
            overlay_X=overlay_X,
        )
        return

    # -------------------------
    # Mode: refine (bisection)
    # -------------------------
    if args.mode == "refine":
        transitions = {
            "a1": (1, 2),
            "a2": (2, 4),
            "a3": (4, 8),
            "a4": (8, 16),
        }

        run_one_a_fn = make_run_one_a_fn(
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
        )

        refined: Dict[str, float] = {}

        # helper to fetch bracket
        def get_bracket(name: str) -> Optional[Tuple[float, float]]:
            opt = getattr(args, f"bracket_{name}", None)
            if opt is None:
                return None
            return float(opt[0]), float(opt[1])

        for name in args.targets:
            k_lo, k_hi = transitions[name]
            br = get_bracket(name)
            if br is None:
                raise SystemExit(f"refine mode needs an explicit bracket for {name}: use --bracket_{name} <a_lo> <a_hi>")
            a_lo, a_hi = br

            print(f"\n[REFINE] {name}: targeting {k_lo}->{k_hi} in bracket [{a_lo:.10f}, {a_hi:.10f}]")
            aL, aH, aM = bisect_transition(
                run_one_a_fn,
                a_lo=a_lo,
                a_hi=a_hi,
                k_lo=k_lo,
                k_hi=k_hi,
                drop_first=int(args.drop_first),
                tol_x=float(args.tol_x),
                tol_a=float(args.tol_a),
                max_iter=int(args.max_iter),
                verbose=True,
            )
            halfw = 0.5*(aH-aL)
            refined[name] = aM
            print(f"[REFINE] {name} refined: a ≈ {aM:.10f}  bracket=[{aL:.10f}, {aH:.10f}]  halfwidth={halfw:.2e}")

        # print a compact summary (and how to paste into guides)
        if refined:
            print("\n=== Refined estimates (paste into your guide lines) ===")
            for name in ["a1", "a2", "a3", "a4"]:
                if name in refined:
                    print(f"{name} = {refined[name]:.10f}")
                else:
                    print(f"{name} = {guides[name]:.10f}   (unchanged)")

        return

    raise SystemExit("Unknown mode")

if __name__ == "__main__":
    main()
