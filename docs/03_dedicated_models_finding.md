# Dedicated per-fault models versus one blurred detector

## The problem with one detector for all faults

A single detector must call physically different faults "developing" with
one decision boundary. A sharp valve closure and a slow choke scaling look
nothing alike, so the strong, common signatures dominate the boundary and
the subtle faults are sacrificed. Adding richer features did not fix this,
because the limitation is structural, not a lack of information.

## What changed

Each fault type now gets its own model, learning that fault's signature
against normal operation, with grouped cross-validation by well so there
is no leakage. Faults with fewer than 500 positive windows are reported as
insufficient data rather than given an unreliable score.

Source: notebooks/07_per_fault_models.py.
Table: reports/per_fault_dedicated.csv.

## Result: average precision, blurred versus dedicated

| Fault | Name | Blurred | Dedicated | Change |
|---|---|---|---|---|
| 1 | BSW increase | 0.145 | 0.692 | strong gain |
| 2 | DHSV spurious closure | 0.981 | 0.954 | already strong |
| 5 | Rapid productivity loss | 0.126 | 0.139 | still hard |
| 6 | Quick choke restriction | n/a | insufficient data | only 86 windows |
| 7 | Choke scaling | 0.105 | 0.278 | clear gain |
| 8 | Hydrate in production line | 0.610 | 0.458 | weaker alone |
| 9 | Hydrate in service line | 0.168 | 0.247 | gain |

## What this shows

- The single blurred detector was the bottleneck for subtle faults, not
  the features. Fault 1 improved nearly fivefold and fault 7 nearly
  threefold once each had its own model.
- Some faults prefer to be pooled. Fault 8 scored higher in the shared
  detector, suggesting it shares useful signal with other faults.
- Some faults stay hard. Fault 5 does not improve, indicating its early
  signature is genuinely weak in these sensors at this window scale. This
  is reported as a limit, not hidden.
- Fault 6 cannot be modelled reliably with 86 positive windows and is
  reported as such.

## Design implication

The production design should run dedicated detectors for faults that have
enough data and a learnable signature, and be explicit about the faults it
cannot yet cover. Honest coverage beats a single inflated number.