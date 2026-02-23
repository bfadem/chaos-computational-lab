#!/usr/bin/env bash
set -euo pipefail

# Run quick_test.py with conservative thread settings to avoid OpenMP SHM errors.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export KMP_AFFINITY=disabled
export OMP_WAIT_POLICY=PASSIVE

python quick_test.py
