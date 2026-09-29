# How Forewell works, in detail

This explains the full path from a raw sensor reading to an alarm, how the headline numbers were measured, and how the system would fit a real offshore platform. It is written for a reader who wants to check the work.

## The data

One dataset, the public Petrobras 3W dataset (CC BY 4.0). No outside data is mixed in.

Within it, two fault types are pooled into a single hydrate target, because they are the same physical phenomenon in a different location:

- Fault 8: hydrate in the production line
- Fault 9: hydrate in the service line

Pooling them takes the usable sample from a handful of wells up to a workable size. The combined real-well dataset is 22,930 windows from 24 real wells: 15,259 from fault 8 and 7,671 from fault 9.

The 24 wells were not hand-picked. The builder walks every real WELL file in folders 8 and 9 and keeps a well only if it contains both a clean developing period and a clean normal period. Wells that are entirely event, or too short to form a window, produce nothing and drop out. Twenty-four is whatever survived that filter.

## The label scheme

Each second carries a label. 0 is normal. 108 (fault 8) or 109 (fault 9) is the transient, the fault developing but not yet confirmed. 8 or 9 is the confirmed fault.

Forewell targets the transient. A window is positive if at least 90% of its seconds are transient, negative if at least 90% are normal, and dropped otherwise. Targeting the transient is what makes this early warning: the transient comes before the confirmed fault, so catching it buys time to act.

## From sensor to feature

The raw signal is about eight sensors sampled once per second: downhole pressure (P-PDG), tubing pressure and temperature (P-TPT, T-TPT), pressures around the production choke (P-MON-CKP, P-JUS-CKP), temperature after the choke (T-JUS-CKP), gas-lift flow (QGL), and choke opening (ABER-CKP).

A single instant carries no information about a developing fault, so the signal is summarised over a 30-minute window (1,800 rows), sliding forward 5 minutes at a time.

For each window, for each sensor, the following are computed.

Mean (level): the average reading, sum divided by count.

Standard deviation (variability): the square root of the average squared distance from the mean.

Trend (slope): a straight line fitted through the window by least squares, its gradient taken over 10 and 30 minutes. A negative temperature slope means the line is cooling, a hydrate precursor.

Drift from baseline: the window mean minus the well opening baseline (mean of its first 10 minutes). This measures how far the well has moved from its own normal, which travels across wells better than an absolute value.

Choke pressure drop: pressure before the choke minus pressure after it. A growing drop signals a restriction forming.

Each 30-minute window becomes a row of about 42 numbers.

## The model

A gradient-boosted tree ensemble (scikit-learn HistGradientBoostingClassifier) maps those 42 numbers to a probability between 0 and 1 that a hydrate is developing. Gradient boosting builds many shallow decision trees in sequence, each correcting the errors of the ones before it, then combines their outputs. It learns, from the training wells only, which combinations of the 42 features separate developing from normal.

## Validation, and how 0.96 was measured

The score is average precision, not accuracy. Average precision is the area under the precision-recall curve.

For every window in a held-out set the model gives a score. The windows are ranked by score. At each cutoff, precision is the fraction of flagged windows that were truly developing, and recall is the fraction of all developing windows that were caught. Sweeping the cutoff traces the precision-recall curve, and average precision is the area under it.

Validation splits by whole well using grouped 5-fold cross-validation, so no well appears in both training and testing. That is what makes the number trustworthy: the model is scored only on wells it never saw.

The five folds scored 0.959, 1.000, 0.980, 0.869, and 0.971. The mean is 0.956, about 0.96, with a standard deviation of 0.045. The tight spread is what says the result is stable rather than a lucky split.

## Calibration, and what the 90% means

The 90% on the dashboard is not an accuracy figure. It is the target coverage of the conformal prediction layer (alpha 0.10, so 1 minus 0.10 is 90%).

Conformal prediction uses a held-out calibration set to turn a raw score into a prediction set with a coverage guarantee. Each reading returns developing only (confident alarm), normal only (confident quiet), or both (uncertain, referred to a human).

Measured honestly with grouped calibration, real coverage came out at 0.836, a little under the 0.90 target, because wells are not exchangeable, so a threshold learned on one set of wells does not transfer perfectly to others. Coverage holds for 20 of the 24 wells and fails on 4 that behave differently. Those four would be flagged as out-of-distribution in deployment. This is reported, not hidden.

## Explainability, checked not assumed

SHAP was used to find which features drive the model. One stood out by a wide margin: drift in the gas-lift flow. A single dominant feature is a warning sign, so it was tested rather than trusted.

An ablation removed the gas-lift features and retrained. All features scored 0.975 average precision, without gas-lift 0.886, and gas-lift only 0.920. The model does not collapse without gas-lift, so it has real signal across temperature, pressure, and choke features. But gas-lift drift alone reaches 0.920, so it is the single strongest predictor, which fits the physics: a forming hydrate restricts flow, and gas-lift drift is an early sign of that restriction. Removing it also raised variance across wells, so gas-lift drift is stabilising as well as informative.

## How this fits a real offshore platform

Every producing well already streams these readings into a control room through the SCADA system and into a historian database. A technician watches trends across many wells at once. They cannot watch every well every second, and a hydrate builds slowly, so the early drift is easy to miss on a busy shift.

Forewell sits as a layer on top of that existing feed. No new hardware. It reads the same tags the control room already collects. Instead of a person eyeballing dozens of wells, it flags the one well drifting into the hydrate pattern, names the sensor driving it, and does so hours before the flow actually chokes. When the evidence is thin it returns Uncertain and leaves the decision to a person, which matters where a false shutdown is costly and a missed hydrate is dangerous.

A concrete scenario. It is 3am, low staffing. Well 12 tubing temperature has been sliding for two hours and its choke pressure drop is creeping up. A tired technician watching many trends misses it. Forewell raises Well 12 at high confidence, points at the gas-lift drift, and the operator injects a hydrate inhibitor while the line is still flowing. The alternative is a hardened plug by morning, a full intervention, and days of deferred production. That gap is where continuous automated attention earns its place.

## What this is not

A research prototype on 24 wells, not a certified operational system. The early-stage warning window, the most valuable one, is also the hardest to catch, because the very start of a hydrate looks close to normal. Service-line hydrates are under-represented compared with production-line hydrates. These limits are stated so the strong numbers are read in context.
