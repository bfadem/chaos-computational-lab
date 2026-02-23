# Architecture

This repository is organized to support teaching labs in nonlinear dynamics.
The structure emphasizes clarity and modularity over micro-optimizations.

## Core Model
- `models/electronic_oscillator.py` implements the dimensionless ODE system:
  - `ẋ = y`
  - `ẏ = a y − x − z`
  - `ε ż = b + y − c (exp(z) − 1)`
- Parameters are passed as a dict or object with attributes `a`, `b`, `c`, `eps`.

## Integration and Maxima Detection
- `engines/bifurcation_engine.py` provides a general sweep interface and
  local-maxima extraction from sampled trajectories.
- Event-based maxima detection (preferred for accuracy and performance) is
  implemented in `legacy/zoom_only_electronic_oscillator_fast.py`:
  - Maxima occur when `y(t)` crosses zero with negative direction.

## Experiments vs. Tests
- `quick_test.py` and `run_quick_test.sh` are designed for quick feedback.
- Scripts in `experiments/` favor accuracy and reproducibility:
  - Longer transient removal.
  - Tighter solver tolerances.
  - Finer parameter sweeps.
