# Forecasting hydrate formation ahead of time

## The task

Not detection but forecasting. For each window ending at time T, using only
sensor data at or before T, predict whether a hydrate will be confirmed
within the next 8 hours, with a 30-minute gap between the window and the
forecast region so the model never sees the event it predicts. Validated
with grouped cross-validation by well, so every score is on wells unseen in
training.

Sources: src/forewell/hydrate.py, notebooks/09, 10, 11.

## Finding 1: predictability depends on horizon

A horizon sweep (notebooks/11_horizon_sweep.py) showed the per-window signal
is modest and horizon-dependent. Raw average precision rises with horizon,
but so does the base rate, so lift over the no-skill baseline is the honest
measure. The 8-hour horizon balances real signal with enough positive
examples to train on. See reports/figures/11_horizon_sweep.jpg.

## Finding 2: multi-hour trends carry the signal

Hydrate formation is a slow accumulation, so a single 30-minute snapshot is
the wrong frame. Adding features that describe each sensor's trajectory over
the preceding 1, 2, and 4 hours, plus drift from the well's own early
baseline, lifted mean average precision from 0.56 to 0.76 at the 8-hour
horizon. This matches the physics: sustained cooling and a worsening choke
pressure drop precede hydrate formation.

## Honest performance statement

- Mean average precision: 0.76, against a no-skill baseline of 0.47 (1.6x lift).
- Across folds the score ranges from 0.33 to nearly 1.0. This variance is
  real and is driven by having only 9 wells; one fold testing on two clean
  wells can score near-perfect by chance. The mean with its spread is the
  trustworthy summary, not the best fold.
- At the default threshold the model favours recall (catches ~82% of coming
  hydrates) over precision (~46%). The operating point is chosen properly in
  the event-level evaluation, not left at 0.5.

## Limitations

Nine real wells is a small sample. The result is a credible research-grade
forecaster, not a validated production system. More wells would be needed to
tighten the variance before deployment. This is stated rather than hidden.