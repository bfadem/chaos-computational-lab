# Known Issues

## OpenMP SHM Errors on macOS
Some environments fail with:
```
OMP: Error #179: Function Can't open SHM2 failed
```

Workaround:
```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export KMP_AFFINITY=disabled
export OMP_WAIT_POLICY=PASSIVE
```

Then run:
```bash
bash run_quick_test.sh
```

## Slow Runs
Experiment scripts intentionally use long integrations and tight tolerances.
This is expected. For faster checks, use `quick_test.py`.
