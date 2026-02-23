# chaos-computational-lab
Teaching laboratory framework for nonlinear dynamics and chaos. The codebase
is designed to be modular, readable, and easy to extend for coursework and
guided explorations of bifurcations, chaos, and dynamical systems.

## What This Repo Is
- A teaching-focused toolkit for simulating low-dimensional nonlinear systems.
- A set of experiments that illustrate period-doubling routes to chaos.
- A reference implementation of the simple chaotic oscillator from
  Tamaševičius et al. (2005), with scripts to replicate bifurcation behavior.

## Quickstart
1. Create or activate your environment:
   - `use-conda chaos-lab`
2. Install dependencies (if needed):
   - `pip install numpy scipy matplotlib`
3. Run a fast sanity check:
   - `bash run_quick_test.sh`

## Key Scripts
- `quick_test.py`: fast sanity check (speed over accuracy).
- `experiments/medium_zoom_test.py`: accuracy-focused bifurcation scan.
- `experiments/oscillator_zoom_experiment.py`: zoomed bifurcation window.
- `experiments/single_parameter_inspect.py`: single-parameter inspection.
- `experiments/refine_a3_a4_period_doubling.py`: refine a_3 and a_4.
- `experiments/refine_a5_period_doubling.py`: refine a_5.

## Documentation
- `docs/architecture.md`: model and solver structure.
- `docs/experiments.md`: experiment scripts and intended use.
- `docs/repro.md`: reproducibility guide for key results.
- `docs/known_issues.md`: environment quirks and workarounds.

## Citation
See `CITATION.cff` and `CITATION.bib`.
