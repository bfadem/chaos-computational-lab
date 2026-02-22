#!/usr/bin/env python3
"""
Bifurcation diagram + Feigenbaum-zoom helper for the "simple chaotic oscillator"
(Tamaševičius et al., Eur. J. Phys. 26 (2005) 61–63).

Model (paper eq. 4, nondimensional):
    xdot = y
    ydot = a*y - x - z
    eps*zdot = b + y - c*(exp(z) - 1)

What this script does:
1) Coarse bifurcation scan: local maxima of x vs control parameter a.
2) Optional Feigenbaum zoom: given your coarse "split indices" (e.g. 28,43,48),
   it predicts the next split location using Feigenbaum scaling, brackets it,
   and bisection-searches for the next period-doubling transition.

Notes:
- Uses solve_ivp(method="BDF") for speed (system can be stiff).
- Adds progress printing and partial autosave during the coarse scan.
- Saves bifurcation.csv and bifurcation.png
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

    # Clip z to avoid exp overflow; tighter clip helps speed/stability
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

        # Cheap seed update at same 'a' (keeps y0 from drifting too wildly)
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
# Feigenbaum-zoom helper
# ============================================================

def a_from_index(n: float, a_min: float, a_max: float, npts: int) -> float:
    """0-based index -> a value for np.linspace(a_min, a_max, npts)."""
    if npts < 2:
        raise ValueError("npts must be >= 2")
    da = (a_max - a_min) / (npts - 1)
    return a_min + n * da


def estimate_next_index_by_feigenbaum(
    n1: float, n2: float, n3: float, delta_feig: float = 4.669201609102990
) -> float:
    """
    Given three successive split locations (in index units), estimate the next one:
      Δ1 = n2-n1, Δ2 = n3-n2, predict Δ3 ≈ Δ2/δ, so n4 ≈ n3 + Δ2/δ
    """
    d2 = n3 - n2
    if d2 <= 0:
        raise ValueError("Need increasing split indices: n1 < n2 < n3")
    return n3 + d2 / delta_feig


def _cluster_count(values: np.ndarray, tol: float = 1e-3) -> int:
    """Count distinct values after clustering within tolerance."""
    if values.size == 0:
        return 0
    v = np.sort(values.astype(float))
    clusters = 1
    last = v[0]
    for x in v[1:]:
        if abs(x - last) > tol:
            clusters += 1
            last = x
    return clusters


def period_from_maxima(maxima: np.ndarray, tol: float = 1e-3, max_reasonable: int = 64) -> int:
    """
    Estimated period = number of distinct maxima values after clustering.
    Caps at max_reasonable: if it exceeds this, treat as chaotic/large.
    """
    k = _cluster_count(maxima, tol=tol)
    if k > max_reasonable:
        return max_reasonable + 1
    return k


def period_at_a(
    a: float,
    b: float,
    eps: float,
    c: float,
    theta_transient: float,
    theta_sample: float,
    dt_sample: float,
    seed_state: np.ndarray,
    tol_max_cluster: float = 1e-3,
) -> int:
    """Integrate at a, return estimated period based on clustered maxima."""
    p = ModelParams(a=float(a), b=float(b), c=float(c), eps=float(eps))
    maxima = simulate_and_collect_maxima(
        p=p,
        theta_transient=theta_transient,
        theta_sample=theta_sample,
        dt_sample=dt_sample,
        y0=seed_state,
        solver="BDF",
        rtol=1e-6,
        atol=1e-9,
        max_step=2.0,
        max_maxima=120,
    )
    return period_from_maxima(maxima, tol=tol_max_cluster, max_reasonable=64)


def find_period_doubling_a(
    a_lo: float,
    a_hi: float,
    target_left_period: int,
    b: float,
    eps: float,
    c: float,
    theta_transient: float,
    theta_sample: float,
    dt_sample: float,
    seed_state: np.ndarray,
    tol_max_cluster: float = 1e-3,
    max_iter: int = 30,
) -> float:
    """
    Locate parameter a* where the orbit transitions from target_left_period to larger,
    using bisection.
    """
    p_lo = period_at_a(a_lo, b, eps, c, theta_transient, theta_sample, dt_sample, seed_state, tol_max_cluster)
    p_hi = period_at_a(a_hi, b, eps, c, theta_transient, theta_sample, dt_sample, seed_state, tol_max_cluster)

    print(f"[BRACKET] a_lo={a_lo:.8f} period≈{p_lo}, a_hi={a_hi:.8f} period≈{p_hi}", flush=True)

    if p_lo != target_left_period:
        print(
            f"[WARN] Left period {p_lo} != expected {target_left_period}. "
            f"Try increasing theta_transient/theta_sample or adjusting tol_max_cluster.",
            flush=True,
        )
    if p_hi < 2 * target_left_period:
        print(
            f"[WARN] Right period {p_hi} < expected doubling {2*target_left_period}. "
            f"Widen the bracket window.",
            flush=True,
        )

    lo, hi = a_lo, a_hi
    for it in range(max_iter):
        mid = 0.5 * (lo + hi)
        p_mid = period_at_a(mid, b, eps, c, theta_transient, theta_sample, dt_sample, seed_state, tol_max_cluster)

        if p_mid <= target_left_period:
            lo = mid
        else:
            hi = mid

        if it % 5 == 0:
            print(f"[BISECT {it:02d}] mid={mid:.10f} period≈{p_mid}  interval={hi-lo:.3e}", flush=True)

        if (hi - lo) < 1e-7:
            break

    return 0.5 * (lo + hi)


def feigenbaum_zoom_from_indices(
    split_indices: tuple[int, int, int],
    a_min: float,
    a_max: float,
    npts_coarse: int,
    b: float = 30.0,
    eps: float = 0.13,
    c: float = 4e-9,
    theta_transient: float = 1800.0,
    theta_sample: float = 1800.0,
    dt_sample: float = 0.20,
    seed_state: np.ndarray | None = None,
    tol_max_cluster: float = 2e-3,
    window_halfwidth_in_index_units: float = 2.0,
) -> None:
    """
    Convert coarse indices -> (a1,a2,a3), predict next split via Feigenbaum scaling,
    bracket around it, bisection-search it, and report updated delta estimate.
    """
    if seed_state is None:
        seed_state = np.array([0.1, 0.0, 0.0], dtype=float)

    n1, n2, n3 = [float(x) for x in split_indices]
    da = (a_max - a_min) / (npts_coarse - 1)

    a1 = a_from_index(n1, a_min, a_max, npts_coarse)
    a2 = a_from_index(n2, a_min, a_max, npts_coarse)
    a3 = a_from_index(n3, a_min, a_max, npts_coarse)

    d1 = n2 - n1
    d2 = n3 - n2
    delta_est_3 = d1 / d2

    print("\n=== Feigenbaum zoom helper ===")
    print(f"Coarse grid: a in [{a_min}, {a_max}] with npts={npts_coarse} -> Δa={da:.10f}")
    print(f"Splits (index units): n1={n1:.1f}, n2={n2:.1f}, n3={n3:.1f}")
    print(f"Splits (a units):     a1={a1:.8f}, a2={a2:.8f}, a3={a3:.8f}")
    print(f"Index spacings: Δ1={d1:.3f}, Δ2={d2:.3f}  -> ratio Δ1/Δ2 = {delta_est_3:.6f}")

    # Predict next split index and a value
    n4_pred = estimate_next_index_by_feigenbaum(n1, n2, n3)
    a4_pred = a_from_index(n4_pred, a_min, a_max, npts_coarse)
    print(f"Predicted next split (Feigenbaum): n4≈{n4_pred:.3f} -> a4≈{a4_pred:.8f}")

    # Bracket around predicted n4
    n_lo = n4_pred - window_halfwidth_in_index_units
    n_hi = n4_pred + window_halfwidth_in_index_units
    a_lo = a_from_index(n_lo, a_min, a_max, npts_coarse)
    a_hi = a_from_index(n_hi, a_min, a_max, npts_coarse)

    # Infer likely period just below a3 (then round to nearest power of 2)
    a_test_left = max(a_min, a3 - 3 * da)
    p_left = period_at_a(a_test_left, b, eps, c, theta_transient, theta_sample, dt_sample, seed_state, tol_max_cluster)

    if p_left < 1:
        p_left = 1
    pow2 = 1
    while pow2 < p_left and pow2 < 64:
        pow2 *= 2
    target_left_period = pow2 if abs(pow2 - p_left) <= abs(pow2 // 2 - p_left) else max(1, pow2 // 2)

    print(f"Inferred pre-next-doubling period near a≈{a_test_left:.8f} is ≈{p_left}; using target_left_period={target_left_period}")

    # Bisection locate next doubling
    a4 = find_period_doubling_a(
        a_lo=a_lo,
        a_hi=a_hi,
        target_left_period=target_left_period,
        b=b,
        eps=eps,
        c=c,
        theta_transient=theta_transient,
        theta_sample=theta_sample,
        dt_sample=dt_sample,
        seed_state=seed_state,
        tol_max_cluster=tol_max_cluster,
        max_iter=30,
    )

    # Convert a4 back to index-units, compute updated delta estimate
    n4 = (a4 - a_min) / da
    d3 = n4 - n3
    delta_est_4 = d2 / d3

    print("\n=== Result ===")
    print(f"Estimated next split: a4≈{a4:.10f}  (equivalently n4≈{n4:.5f})")
    print(f"New spacing: Δ3 = n4-n3 ≈ {d3:.6f}")
    print(f"Updated delta estimate: Δ2/Δ3 ≈ {delta_est_4:.6f}  (Feigenbaum δ ≈ 4.6692)")
    print("===============================\n")


# -----------------------------
# Main
# -----------------------------

def main():
    # Parameters matching the paper’s Fig. 2 (dimensionless):
    b = 30.0
    eps = 0.13
    c = 4e-9

    # FAST first-pass scan (coarse)
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

    # Save final data
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

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(A, X, linestyle="None", marker=",")
    plt.xlabel("control parameter a")
    plt.ylabel("local maxima of x (x_m)")
    plt.title("Bifurcation diagram: maxima of x vs a (paper eq. 4)")
    plt.tight_layout()
    plt.savefig("bifurcation.png", dpi=200)
    print("[DONE] wrote bifurcation.png", flush=True)
    plt.show()

    # ---- Feigenbaum zoom (your measured coarse split indices) ----
    feigenbaum_zoom_from_indices(
        split_indices=(28, 43, 48),
        a_min=a_min,
        a_max=a_max,
        npts_coarse=npts,
        b=b,
        eps=eps,
        c=c,
        theta_transient=1800.0,
        theta_sample=1800.0,
        dt_sample=0.20,
        tol_max_cluster=2e-3,
        window_halfwidth_in_index_units=2.0,
    )

    # Optional: show mapping from paper-like component values (sanity check)
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
