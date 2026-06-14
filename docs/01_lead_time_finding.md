# Early-warning lead time in the 3W dataset

## Question

Before an offshore well fault is confirmed, how much warning is there in
the sensor data? If the answer is "none", then early detection is
impossible and the project has no point. If there is a usable gap, that
gap is the time an operator has to act, and predicting inside it is the
whole goal of Forewell.

## Method

The 3W dataset labels each second of each recording. Label 0 is normal.
A label of N (1 to 9) marks a confirmed fault of that type. A label of
100 + N marks the transient period: the fault is developing but not yet
confirmed. For every real well recording (not simulated), I found the
first second labelled as the transient warning and the first second
labelled as the confirmed fault, then measured the gap between them.
Each row of data is one second, so the gap converts directly to minutes.

Source: notebooks/02_lead_time.py. Per-instance results:
reports/lead_time_by_instance.csv.

## Result

Across 48 real fault instances with a transient window:

- Median lead time: 182 minutes (about 3 hours).
- Middle half of cases: 50 to 1,529 minutes.
- Shortest: 9 minutes. Longest: about 1 week.

Because the spread is very wide and a few instances run into days, the
mean (1,260 minutes) is misleading. The median is the honest summary.

Lead time depends strongly on fault type, and the ordering matches the
physics:

- Fault 6, quick restriction in the production choke: median 12 minutes.
  A mechanical restriction happens fast, so warning is short.
- Fault 8, hydrate formation in the production line: median 2,902 minutes.
  Hydrates accrete slowly, so warning is long.

See reports/figures/02_lead_time.jpg.

## Why this matters

Real faults announce themselves in the sensor data well before they are
confirmed, often by hours. A model that reads the live sensor streams and
flags the transient window gives operators time to intervene. The lead
time also sets a realistic target: for fast faults like type 6, the model
must react within minutes to be useful, which shapes how the detector and
the alarm are designed.