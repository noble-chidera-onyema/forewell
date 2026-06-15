# Combined hydrate detector (faults 8 and 9)

## Framing

Hydrate formation in the production line (fault 8) and the service line
(fault 9) is the same physical phenomenon in a different location, so they
are pooled into one detection target. A window is positive if it lies in
the developing-transient hydrate state, negative if in normal operation.
The transient state precedes confirmation by a long margin, so detecting it
is genuine early warning, not after-the-fact flagging.

Source: src/forewell/hydrate.py (build_detection_dataset_combined),
notebooks/15_hydrate_combined.py.

## Why pooled

Fault 8 alone has only 14 real wells, 4 with usable structure, which proved
too few to train a stable model, and the simulated instances did not
transfer to real wells (a clean sim-to-real gap, documented in earlier
notebooks). Pooling faults 8 and 9 raises the real-well count to 24, enough
for honest grouped cross-validation on real data alone.

## Result

Grouped 5-fold cross-validation by well, scores calibrated within each
fold, every fold tested on wells unseen in training:

- Mean average precision: 0.956, standard deviation 0.045.
- No-skill baseline: 0.554. Lift: 1.73x.
- Per-fold range: 0.869 to 1.000.
- Pooled operating point (threshold 0.5): precision 0.919, recall 0.723 on
  developing hydrates.

The low fold-to-fold variance is the key point: unlike the 4-well version
which swung from 0.33 to 1.0, the 24-well result is stable across every
held-out set, so the score reflects real generalisation.

## Honest caveats

- The pool is unbalanced: fault 8 contributes 12,117 positive windows,
  fault 9 only 585. The detector is therefore proven mainly on
  production-line hydrates; service-line coverage is lighter and would need
  more data to claim equal reliability.
- Real recordings are transient-heavy, so the positive rate (0.55) is higher
  than a live deployment would see. The lift over baseline, not raw
  precision, is the honest measure of skill.

## Status

A stable, leakage-free hydrate detector validated across 24 real wells.
This is the working core the rest of the system (uncertainty, explainability,
live dashboard) is built on.