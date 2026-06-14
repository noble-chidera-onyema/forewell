# Per-fault detectability and what drives it

## What this measures

The pooled detector scores one number across all faults, which hides
where it works and where it fails. For each fault type, holding wells out
so there is no leakage, this measures average precision and recall for
catching the developing window, placed next to the median early-warning
lead time measured in notebooks/02_lead_time.py.

Source: notebooks/06_per_fault_breakdown.py.
Table: reports/per_fault_performance.csv.

## Result

| Fault | Name | Lead time (min) | Avg precision | Recall @0.5 | Positive windows |
|---|---|---|---|---|---|
| 2 | DHSV spurious closure | 86 | 0.981 | 0.994 | 1394 |
| 8 | Hydrate in production line | 2902 | 0.570 | 0.759 | 2000 |
| 9 | Hydrate in service line | 103 | 0.168 | 0.299 | 2000 |
| 1 | BSW increase | 164 | 0.145 | 0.604 | 1072 |
| 5 | Rapid productivity loss | 153 | 0.126 | 0.054 | 2000 |
| 7 | Choke scaling | 720 | 0.105 | 0.030 | 2000 |
| 6 | Quick choke restriction | 12 | 0.046 | 0.767 | 86 |

## What drives detectability

Detectability is governed by the shape of the early signal, not lead time
alone.

- Sharp, distinctive signatures are easy. Fault 2, a safety valve closing,
  produces an abrupt pressure change and scores near perfect, even though
  its lead time is only moderate.
- Slow, gradual signatures can still be caught when the deviation grows
  large. Fault 8, hydrate formation, develops over many hours and is
  detected reasonably well.
- Subtle, creeping signatures are hard. Fault 7, choke scaling, has long
  lead time but its early signature resembles normal operation, so the
  current windowed features struggle to separate it.

## Honesty notes

- Fault 6 has only 86 positive windows. Its scores are unstable and should
  not be read as reliable. Reported here for completeness, not as a claim.
- Average precision is the fair metric under heavy imbalance; recall at a
  fixed 0.5 threshold can look high while precision is poor, as in fault 6.

## What this motivates

The weak results on subtle faults (7, 9, 5) are the reason for the next
two steps: a per-fault diagnosis stage so the model is not forced to blur
all faults into one class, and richer features that can expose slow,
small deviations a 5-minute window currently misses.