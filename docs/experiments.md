# Experiments

This folder contains scripts intended for classroom use and guided exploration.
They prioritize accuracy over runtime unless explicitly noted.

## Summary
- `medium_zoom_test.py`
  - Medium-resolution bifurcation scan of `a` using paper defaults.
  - Useful for seeing the period-doubling sequence at moderate cost.
- `oscillator_zoom_experiment.py`
  - High-resolution zoomed window for `a` values of interest.
  - Uses longer integrations to reduce transient effects.
- `single_parameter_inspect.py`
  - Single `a` value inspection; prints peak counts and plots a time series.
- `refine_a3_a4_period_doubling.py`
  - Automatically brackets and refines the 4→8 and 8→16 transitions.
  - Uses event-based maxima and clustering for robust period detection.
- `refine_a5_period_doubling.py`
  - Targeted refinement of the 8→16 transition near a_5 (as instructed).

## Parameter Defaults
All experiment scripts use the paper defaults unless stated otherwise:
- `b = 30`
- `eps = 0.13`
- `c = 4e-9`

These values match the canonical bifurcation diagram in the paper.

## Run Tips
If OpenMP shared memory errors occur, use:
- `bash run_quick_test.sh` for a fast check.
- Or set thread limits in your shell before running an experiment.
