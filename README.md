# Forewell

Early-warning detection of hydrate formation in offshore oil wells, from real multivariate sensor data.

**Live demo:** https://forewell.streamlit.app/

## What it does

Forewell reads a well's sensor data and raises an alarm while a hydrate is still forming, during the window when an operator can still act. It is trained and tested on real wells from the Petrobras 3W dataset. Each alarm carries a calibrated confidence level and shows the sensor readings behind it.

## Key results

- Combined hydrate detector (production-line and service-line hydrates) reaches about 0.96 average precision across 24 real wells, validated by splitting on whole wells so no well appears in both training and testing.
- Calibrated confidence through conformal prediction, with an honest per-well map of where the guarantee holds and where it does not.
- Explainable alarms: the strongest driver is drift in gas-lift flow, consistent with the physics of a hydrate restricting a line, verified by an ablation test.
- Early-warning framing: the model targets the developing fault, not the confirmed one, because that is the window where warning has value.

Full findings are in the `docs/` folder. Analysis scripts are in `notebooks/`, the reusable pipeline in `src/forewell/`, and the live dashboard in `app/`.

## Data and attribution

Built on the public Petrobras 3W dataset (CC BY 4.0). Citation: Vaz Vargas et al., "A realistic and public dataset with rare undesirable real events in oil wells," Journal of Petroleum Science and Engineering, vol. 181, 2019, 106223. DOI 10.1016/j.petrol.2019.106223.

## Status

Research prototype, not a certified operational system.

Copyright (c) 2026 Noble Chidera Onyema. All rights reserved. See LICENSE.