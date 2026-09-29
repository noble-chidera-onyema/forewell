# Forewell

Early-warning detection of hydrate formation in offshore oil wells, from real multivariate sensor data.

**Live demo:** https://forewell.streamlit.app/

## What it does

Forewell reads a well's sensor data and raises an alarm while a hydrate is still forming, during the window when an operator can still act. It is trained and tested on real wells from the Petrobras 3W dataset. Each alarm carries a calibrated confidence level and shows the sensor readings behind it.

A hydrate is a solid blockage that builds inside a production line. Left undetected it means lost production, an expensive intervention, and a safety risk. Catching it early is the difference between a small correction and a shutdown.

## How it works

The raw data is one sensor reading per second. Forewell slides a window over each well and, for each window, summarises every sensor: its level, its variability, and its trend over the preceding minutes and hours. Hydrate formation is a slow accumulation, so the multi-hour trend features carry most of the signal. It also computes drift from each well's own baseline and the pressure drop across the production choke.

A gradient-boosted classifier is trained on these features. It is validated with grouped cross-validation by well, so no well ever appears in both training and testing. Scores are then calibrated with conformal prediction, which turns a raw score into a prediction set with a coverage guarantee and lets the system report Uncertain instead of guessing.

The reusable pipeline is in `src/forewell/`, the analysis in `notebooks/`, the written findings in `docs/`, and the live dashboard in `app/`.

For the full method, the exact calculations, how the numbers were measured, and how it maps to a real platform, see [docs/how_it_works_in_detail.md](docs/how_it_works_in_detail.md).

## Key results

- Combined hydrate detector (production-line and service-line hydrates) reaches about 0.96 average precision across 24 real wells, validated by splitting on whole wells so no well appears in both training and testing.
- Calibrated confidence through conformal prediction, with an honest per-well map of where the guarantee holds and where it does not.
- Explainable alarms: the strongest single driver is drift in gas-lift flow, consistent with the physics of a hydrate restricting a line, verified by an ablation test.
- Early-warning framing: the model targets the developing fault, not the confirmed one, because that is the window where warning has value.

## Known limitations

- The strong window-level score does not translate into strong early-stage event coverage at low false-alarm rates. The earliest part of a developing hydrate resembles normal operation, so the most valuable warning window is the hardest to catch. This is documented rather than hidden.
- The dataset pools two hydrate faults. Production-line hydrates dominate the sample; service-line coverage is lighter and would need more data to claim equal reliability.
- Twenty-four real wells is a modest sample. The result is a credible research prototype, not a validated production system.
- Conformal coverage holds for 20 of 24 wells but fails on 4 that behave differently from the rest. In deployment those would be flagged as out-of-distribution.

## Run it locally

```bash
git clone https://github.com/noble-chidera-onyema/forewell.git
cd forewell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

The dashboard runs on a saved model and bundled demo data, so it works without the full dataset. To rebuild the model from scratch you also need the Petrobras 3W dataset cloned alongside this repository.

## Data and attribution

Built on the public Petrobras 3W dataset (CC BY 4.0). Citation: Vaz Vargas et al., "A realistic and public dataset with rare undesirable real events in oil wells," Journal of Petroleum Science and Engineering, vol. 181, 2019, 106223. DOI 10.1016/j.petrol.2019.106223.

## Author

Built by Noble Chidera Onyema, MSc Applied Artificial Intelligence and User Experience, Abertay University, with a background in Mechatronics Engineering.

Email: onyemanoble1628@gmail.com
LinkedIn: https://www.linkedin.com/in/noble-chidera-onyema-1a88b53ab/

## Status and licence

Research prototype, not a certified operational system.

Copyright (c) 2026 Noble Chidera Onyema. All rights reserved. See LICENSE.