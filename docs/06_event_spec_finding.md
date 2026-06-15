# Event-level early-warning: what the detector can and cannot claim

## The gap between window-level and event-level performance

The combined detector scores 0.96 average precision at the window level:
given a window, it reliably separates developing-hydrate from normal. But
event-level early warning is a stricter question: for each well, does an
alarm fire during the pre-confirmation period, early enough to act, without
flooding normal operation with false alarms?

Source: notebooks/16_combined_event_spec.py.
Curve: reports/hydrate_combined_spec.csv.

## Result

Across 11 event wells with a usable pre-event normal stretch:

- At a high-confidence threshold the detector gives 13 to 16 hours of
  warning on the events it does catch.
- But coverage at usable false-alarm rates is limited. At ~19 false alarms
  per well-day it catches about 18% of events; pushing coverage higher
  drives false alarms to 50+ per well-day.
- There is no operating point that catches most events early at a low
  false-alarm cost.

## Why

Hydrate signatures strengthen as the event develops. The model fires
confidently once a hydrate is well-formed, which inflates window-level AP,
but the earliest part of the transient, where warning is most valuable,
resembles normal operation closely. High window-level separation therefore
does not translate into high early-stage event coverage.

## Honest conclusion

The genuine, defensible claim is a strong hydrate-state detector (0.96 AP,
stable across 24 real wells) that provides many hours of warning on clear
cases, with limited early-stage coverage at low false-alarm rates. This is
stated as the system's true operating envelope rather than hidden behind the
headline AP. The next step, calibrated uncertainty via conformal prediction,
directly quantifies the model's uncertainty in the early window instead of
overstating confidence there.