# Reproducibility Guide

This guide focuses on reproducible, teaching-friendly workflows.

## Environment
Recommended:
- Python 3.10+ (conda environment `chaos-lab`)
- `numpy`, `scipy`, `matplotlib`

Install:
```bash
pip install numpy scipy matplotlib
```

## Quick Sanity Check
```bash
bash run_quick_test.sh
```

## Reproducing the Paper-Style Bifurcation
Run a medium-zoom scan:
```bash
python experiments/medium_zoom_test.py
```

## Refining Period-Doubling Points
```bash
python experiments/refine_a3_a4_period_doubling.py
python experiments/refine_a5_period_doubling.py
```

These scripts:
- Remove long transients.
- Use event-based maxima detection.
- Cluster maxima to estimate period.

## Determinism Notes
Results should be stable with fixed tolerances and sufficient integration
time. If your environment differs (BLAS/OpenMP variants), use thread limits:
```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export KMP_AFFINITY=disabled
export OMP_WAIT_POLICY=PASSIVE
```
