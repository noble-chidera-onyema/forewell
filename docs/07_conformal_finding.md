# Conformal prediction: calibrated confidence and honest abstention

## Goal

Convert the detector's raw scores into outputs with a coverage guarantee,
so an alarm carries calibrated confidence and the system can abstain when
genuinely uncertain. Three possible outputs per window:

- {developing}: confident alarm.
- {normal}: confident quiet.
- {normal, developing}: uncertain, escalate to a human.

## Naive split-conformal under-covers

A standard train/calibrate/test split by well gave empirical coverage of
only 0.65 against a 0.90 target (notebooks/17). The cause is real and worth
stating: conformal coverage assumes calibration and test data are
exchangeable, but wells are not exchangeable with each other. A threshold
learned on one set of wells does not transfer cleanly to different wells.
This is distribution shift across wells, a genuine limitation of conformal
methods on grouped industrial data.

## Grouped calibration fixes most of it

Calibrating with leave-one-fold-out across the same grouped cross-validation
(notebooks/18) raised pooled coverage to 0.836, with 81% confident single
answers and 17% honest abstentions (mean set size 1.14).

## The honest per-well picture

Coverage is a per-well property, not just an average. The guarantee holds at
or near 1.0 for 20 of 24 wells, but fails on four:

- WELL-00020: 0.00
- WELL-00026: 0.31
- WELL-00042: 0.48
- WELL-00014: 0.56

These wells behave differently enough that the calibrated threshold does not
transfer. In deployment they would be flagged as out-of-distribution and
handled with extra caution rather than trusted blindly. Reporting exactly
which wells fail, and why coverage holds per-well rather than only on
average, is the difference between using conformal prediction and
understanding its limits.

## Status

Calibrated uncertainty with documented coverage, including an explicit map
of where the guarantee holds and where it breaks. This directly addresses
the model's uncertainty in the early developing window by quantifying it
rather than overstating confidence.