#!/usr/bin/env python3
"""
simple_electronic_oscillator_chaos_03.py
Bifurcation diagram + zoom windows with labeled a-values (simple chaotic oscillator)
(Tamaševičius et al., Eur. J. Phys. 26 (2005) 61–63).

Model (paper eq. 4, nondimensional):
    xdot = y
    ydot = a*y - x - z
    eps*zdot = b + y - c*(exp(z) - 1)

Outputs:
- Coarse scan:
    bifurcation.csv
    bifurcation.png
    bifurcation_partial.csv (autosave)
- Zoom scans (examples):
    zoom_a1_a2_bifurcation.csv / .png
    zoom_a2_a3_bifurcation.csv / .png
    zoom_a3_a4_bifurcation.csv / .png

Zoom plots also support click-to-report: click anywhere -> prints a (x) and x_max (y).
"""

from __future__ import annotations

import time
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List

from scipy.integrate import solve_ivp


# -----------------------------
# Physical-to-dimensionless mapping (optional)
# -----------------------------

@dataclass
class CircuitParams:
    # Paper’s suggested example values:
    L: float = 100e-3      # H
    C: float = 100e-9      # F
    Cstar: float = 15e-9   # F
    R: float = 1e3         # Ohm
    R1: float = 10e3       # Ohm
    R2: float = 4e3        # Ohm (variable)
    R0: float = 20e3       # Ohm
    Vb: float = 20.0       # V (dc bias supply for diode via R0)

    # Thermal voltage at room temperature ~ 25.85 mV
    VT: float = 25.85e-3   # V

    # Diode saturation current Is is not specified by the paper; treat as small.
    Is: float = 1e-13      # A (rough; adjust if you want)

    @property
    def k(self) -> float:
        return self.R2 / self.R1 + 1.0

    @property
    def rho(self) -> float:
        return float(np.sqrt(self.L / self.C))

    @property
    def tau(self) -> float:
        return float(np.sqrt(self.L * self.C))

    @property
    def eps(self) -> float:
        return float(self.Cstar / self.C)

    @property
    def I0(self) -> float:
        return float(self.Vb / self.R0)

    def to_dimensionless(self) -> Tuple[float, float, float, float]:
        """
        Returns (a, b, c, eps) from the component values.
        """
        a = (self.k - 1.0) * self.R / self.rho
        b = (self.rho * self.I0) / self.VT
        c = (self.rho * self.Is) / self.VT
        return float(a), float(b), float(c), float(self.eps)


# -----------------------------
# Dimensionless model (paper eq. 4)
# -----------------------------

@dataclass
class ModelParams:
    a: float
    b: float = 30.0
    c: float = 4e-9
    eps: float = 0.13


def rhs(theta: float, state: np.ndarray, p: ModelParams) -> np.ndarray:
    x, y, z = state

    # Clip z to avoid exp overflow
    z_clip = np.clip(z, -50.0, 50.0)
    ez = np.exp(z_clip)

    xdot = y
    ydot = p.a * y - x - z
    zdot = (p.b + y - p.c * (ez - 1.0)) / p.eps

    return np.array([xdot, ydot, zdot], dtype=float)


# -----------------------------
# Peak picking
# -----------------------------

def local_maxima_of_x(x: np.ndarray) -> np.ndarray:
    """
    Local maxima detection on uniformly sampled x.
    Returns x[i] where x[i-1] < x[i] > x[i+1].
    """
    if x.size < 3:
        return np.array([], dtype=float)
    mid = x[1:-1]
    mask = (mid > x[:-2]) & (mid > x[2:])
    return mid[mask]


# -----------------------------
# Simulate and collect maxima
# -----------------------------

def simulate_and_collect_maxima(
    p: ModelParams,
    theta_transient: float,
    theta_sample: float,
    dt_sample: float,
    y0: np.ndarray,
    solver: str = "BDF",
    rtol: float = 1e-6,
    atol: float = 1e-9,
    max_step: float = 2.0,
    max_maxima: int = 80,
) -> np.ndarray:
    """
    One-shot integration with uniform t_eval.
    Discard transient samples, then extract local maxima of x.
    """
    t_end = theta_transient + theta_sample
    t_eval = np.arange(0.0, t_end + dt_sample, dt_sample)

    sol = solve_ivp(
        fun=lambda t, s: rhs(t, s, p),
        t_span=(0.0, t_end),
        y0=y0,
        method=solver,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )
    if not sol.success:
        raise RuntimeError(f"Integration failed for a={p.a}: {sol.message}")

    x = sol.y[0]

    # Discard transient samples
    cut = int(theta_transient / dt_sample)
    if cut >= x.size - 3:
        return np.array([], dtype=float)

    x2 = x[cut:]
    maxima = local_maxima_of_x(x2)

    # Keep only the last N maxima (focus on attractor)
    if maxima.size > max_maxima:
        maxima = maxima[-max_maxima:]

    return maxima


# -----------------------------
# Coarse bifurcation scan
# -----------------------------

def bifurcation_scan(
    a_values: np.ndarray,
    b: float = 30.0,
    eps: float = 0.13,
    c: float = 4e-9,
    theta_transient: float = 900.0,
    theta_sample: float = 900.0,
    dt_sample: float = 0.25,
    seed_state: np.ndarray | None = None,
    autosave_every: int = 50,
    autosave_path: str = "bifurcation_partial.csv",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns arrays (A, Xmax) suitable for scatter plotting:
      A: repeated a values
      Xmax: corresponding maxima of x

    Prints progress and autosaves partial results.
    """
    if seed_state is None:
        seed_state = np.array([0.1, 0.0, 0.0], dtype=float)

    A_list: List[float] = []
    X_list: List[float] = []

    y0 = seed_state.copy()
    t0 = time.time()

    for i, a in enumerate(a_values):
        p = ModelParams(a=float(a), b=float(b), c=float(c), eps=float(eps))

        try:
            maxima = simulate_and_collect_maxima(
                p=p,
                theta_transient=theta_transient,
                theta_sample=theta_sample,
                dt_sample=dt_sample,
                y0=y0,
                solver="BDF",
                rtol=1e-6,
                atol=1e-9,
                max_step=2.0,
                max_maxima=80,
            )
        except Exception as e:
            print(f"[WARN] i={i:4d} a={a:.6f} failed: {e}", flush=True)
            continue

        A_list.extend([float(a)] * int(maxima.size))
        X_list.extend([float(v) for v in maxima])

        if i % 10 == 0:
            elapsed = time.time() - t0
            print(
                f"[{i:4d}/{len(a_values)}] a={a:.5f} maxima={maxima.size:3d} "
                f"points={len(A_list):7d} elapsed={elapsed:.1f}s",
                flush=True,
            )

        if autosave_every > 0 and (i % autosave_every == 0) and len(A_list) > 0:
            arr = np.column_stack([np.array(A_list), np.array(X_list)])
            np.savetxt(autosave_path, arr, delimiter=",", header="a,x_max", comments="")

        # cheap seed update at same 'a'
        sol_seed = solve_ivp(
            fun=lambda t, s: rhs(t, s, p),
            t_span=(0.0, 40.0),
            y0=y0,
            method="BDF",
            rtol=1e-6,
            atol=1e-9,
            max_step=2.0,
        )
        if sol_seed.success:
            y0 = sol_seed.y[:, -1]

    return np.array(A_list, dtype=float), np.array(X_list, dtype=float)


# ============================================================
# Zoom scan utilities
# ============================================================

def make_dense_a_values(a_lo: float, a_hi: float, npts: int = 1500) -> np.ndarray:
    if a_hi <= a_lo:
        raise ValueError("Need a_hi > a_lo")
    return np.linspace(float(a_lo), float(a_hi), int(npts))

def save_scatter_csv(path: str, A: np.ndarray, X: np.ndarray) -> None:
    arr = np.column_stack([A, X])
    np.savetxt(path, arr, delimiter=",", header="a,x_max", comments="")

def annotate_xticks_every(ax, step: float = 0.002) -> None:
    xmin, xmax = ax.get_xlim()
    start = np.floor(xmin / step) * step
    ticks = np.arange(start, xmax + step, step)
    ax.set_xticks(ticks)

def add_vertical_guides(ax, a_guides: List[float], label_prefix: str = "a=") -> None:
    """
    Vertical guide lines with text labels near the top of the plot.
    """
    y0, y1 = ax.get_ylim()
    text_y = y1 - 0.05 * (y1 - y0)

    for a in a_guides:
        ax.axvline(a, linewidth=1.0)
        ax.text(
            a,
            text_y,
            f"{label_prefix}{a:.6f}",
            rotation=90,
            va="top",
            ha="right",
        )

def enable_click_readout(fig, ax, label: str = "") -> None:
    """
    Click anywhere in the plot: prints x,y to terminal.
    """
    def onclick(event):
        if event.inaxes != ax:
            return
        x = event.xdata
        y = event.ydata
        if x is None or y is None:
            return
        print(f"[CLICK{(' '+label) if label else ''}] a≈{x:.10f}, y≈{y:.10f}", flush=True)

    fig.canvas.mpl_connect("button_press_event", onclick)

def zoom_bifurcation_scan(
    a_lo: float,
    a_hi: float,
    npts: int,
    *,
    b: float,
    eps: float,
    c: float,
    theta_transient: float,
    theta_sample: float,
    dt_sample: float,
    seed_state: np.ndarray | None = None,
    save_prefix: str = "zoom",
    a_guides: List[float] | None = None,
    xtick_step: float | None = None,
    click_readout: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dense scan over [a_lo, a_hi], return (A, Xmax) scatter arrays.
    Saves:
      - {save_prefix}_bifurcation.csv
      - {save_prefix}_bifurcation.png
    """
    if seed_state is None:
        seed_state = np.array([0.1, 0.0, 0.0], dtype=float)

    a_values = make_dense_a_values(a_lo, a_hi, npts=npts)

    A_list: List[float] = []
    X_list: List[float] = []

    y0 = seed_state.copy()

    t0 = time.time()
    for i, a in enumerate(a_values):
        p = ModelParams(a=float(a), b=float(b), c=float(c), eps=float(eps))

        maxima = simulate_and_collect_maxima(
            p=p,
            theta_transient=theta_transient,
            theta_sample=theta_sample,
            dt_sample=dt_sample,
            y0=y0,
            solver="BDF",
            rtol=1e-6,
            atol=1e-9,
            max_step=2.0,
            max_maxima=120,
        )

        A_list.extend([float(a)] * int(maxima.size))
        X_list.extend([float(v) for v in maxima])

        if i % max(1, npts // 10) == 0:
            elapsed = time.time() - t0
            print(
                f"[ZOOM {save_prefix}] {i:4d}/{npts}  a={a:.8f}  maxima={maxima.size:3d}  elapsed={elapsed:.1f}s",
                flush=True
            )

        # cheap continuation seed update
        sol_seed = solve_ivp(
            fun=lambda t, s: rhs(t, s, p),
            t_span=(0.0, 20.0),
            y0=y0,
            method="BDF",
            rtol=1e-6,
            atol=1e-9,
            max_step=2.0,
        )
        if sol_seed.success:
            y0 = sol_seed.y[:, -1]

    A = np.array(A_list, dtype=float)
    X = np.array(X_list, dtype=float)

    csv_path = f"{save_prefix}_bifurcation.csv"
    png_path = f"{save_prefix}_bifurcation.png"
    save_scatter_csv(csv_path, A, X)
    print(f"[ZOOM {save_prefix}] wrote {csv_path} with {A.size} rows", flush=True)

    fig = plt.figure(figsize=(10, 5))
    ax = plt.gca()
    ax.plot(A, X, linestyle="None", marker=",")
    ax.set_xlabel("control parameter a")
    ax.set_ylabel("local maxima of x (x_m)")
    ax.set_title(f"Zoom bifurcation: maxima of x vs a  (a in [{a_lo:.6f}, {a_hi:.6f}])")

    if xtick_step is not None and xtick_step > 0:
        annotate_xticks_every(ax, step=xtick_step)

    # Ensure y-limits are set before adding guide labels
    fig.tight_layout()
    if a_guides:
        add_vertical_guides(ax, a_guides, label_prefix="a=")

    if click_readout:
        enable_click_readout(fig, ax, label=save_prefix)

    fig.tight_layout()
    fig.savefig(png_path, dpi=250)
    print(f"[ZOOM {save_prefix}] wrote {png_path}", flush=True)
    plt.show()

    return A, X


# -----------------------------
# Main
# -----------------------------

def main():
    # Paper-like dimensionless parameters (Fig. 2 uses these)
    b = 30.0
    eps = 0.13
    c = 4e-9

    # ----- Coarse scan settings -----
    a_min, a_max = 0.0, 0.85
    npts = 220
    a_values = np.linspace(a_min, a_max, npts)

    theta_transient = 900.0
    theta_sample = 900.0
    dt_sample = 0.25

    A, X = bifurcation_scan(
        a_values=a_values,
        b=b,
        eps=eps,
        c=c,
        theta_transient=theta_transient,
        theta_sample=theta_sample,
        dt_sample=dt_sample,
        seed_state=np.array([0.1, 0.0, 0.0], dtype=float),
        autosave_every=50,
        autosave_path="bifurcation_partial.csv",
    )

    if A.size > 0:
        np.savetxt(
            "bifurcation.csv",
            np.column_stack([A, X]),
            delimiter=",",
            header="a,x_max",
            comments="",
        )
        print(f"[DONE] wrote bifurcation.csv with {A.size} rows", flush=True)
    else:
        print("[DONE] no data collected (check parameters / solver settings).", flush=True)

    plt.figure(figsize=(10, 5))
    plt.plot(A, X, linestyle="None", marker=",")
    plt.xlabel("control parameter a")
    plt.ylabel("local maxima of x (x_m)")
    plt.title("Coarse bifurcation diagram: maxima of x vs a (paper eq. 4)")
    plt.tight_layout()
    plt.savefig("bifurcation.png", dpi=200)
    print("[DONE] wrote bifurcation.png", flush=True)
    plt.show()

    # ------------------------------------------------------------
    # Your current best estimates (update these as you refine)
    # From your terminal output:
    #   a1 = 0.10867580
    #   a2 = 0.16689498
    #   a3 = 0.18630137
    # Predicted a4 (from Feigenbaum scaling attempt, treat as a guide):
    #   a4_est = 0.19045762
    # ------------------------------------------------------------
    a1 = 0.10867580
    a2 = 0.16689498
    a3 = 0.18630137
    a4_est = 0.19045762

    # ----- Zoom settings (separate windows so plots stay readable) -----
    # These are “good first guesses.” You can widen/narrow as desired.
    # The main control is: window size and npts.
    #
    # Tip: if a window is too busy, shrink it and/or increase theta_transient/sample.

    # Zoom around a1..a2
    zoom_bifurcation_scan(
        a_lo=a1 - 0.010,
        a_hi=a2 + 0.010,
        npts=2500,
        b=b, eps=eps, c=c,
        theta_transient=1500.0,
        theta_sample=1500.0,
        dt_sample=0.15,
        save_prefix="zoom_a1_a2",
        a_guides=[a1, a2],
        xtick_step=0.002,
        click_readout=True,
    )

    # Zoom around a2..a3
    zoom_bifurcation_scan(
        a_lo=a2 - 0.008,
        a_hi=a3 + 0.008,
        npts=2500,
        b=b, eps=eps, c=c,
        theta_transient=2000.0,
        theta_sample=2000.0,
        dt_sample=0.12,
        save_prefix="zoom_a2_a3",
        a_guides=[a2, a3],
        xtick_step=0.0015,
        click_readout=True,
    )

    # Zoom around a3..a4_est
    zoom_bifurcation_scan(
        a_lo=a3 - 0.006,
        a_hi=a4_est + 0.006,
        npts=3000,
        b=b, eps=eps, c=c,
        theta_transient=2500.0,
        theta_sample=2500.0,
        dt_sample=0.10,
        save_prefix="zoom_a3_a4",
        a_guides=[a3, a4_est],
        xtick_step=0.001,
        click_readout=True,
    )

    # Optional: show mapping from component values (sanity check)
    circ = CircuitParams(R2=4e3)
    a_phys, b_phys, c_phys, eps_phys = circ.to_dimensionless()
    print("\n--- Dimensionless parameters from paper-like component values ---")
    print(f"k = {circ.k:.6f}  (with R2={circ.R2} ohm, R1={circ.R1} ohm)")
    print(f"rho = sqrt(L/C) = {circ.rho:.3f} ohm")
    print(f"tau = sqrt(LC)  = {circ.tau:.6e} s")
    print(f"eps = C*/C      = {eps_phys:.6f}")
    print(f"a   = (k-1)R/rho = {a_phys:.6f}")
    print(f"b   = rho*I0/VT   = {b_phys:.6f}   (I0 ~ Vb/R0 = {circ.I0:.6e} A)")
    print(f"c   = rho*Is/VT   = {c_phys:.6e}   (Is guess = {circ.Is:.2e} A)")
    print("Note: the paper’s Fig.2 uses b=30, eps=0.13, c=4e-9 (dimensionless).")


if __name__ == "__main__":
    main()
