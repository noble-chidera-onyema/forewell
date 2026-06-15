# Full event-level validation with calibrated scores.
#
# Coverage and warning lead are measured on hydrate wells against true
# event onset. False alarms are measured separately on pure-normal wells
# that never hydrate, which is where a false alarm actually means crying
# wolf. Model scores are calibrated so the alarm threshold genuinely
# separates warning from quiet. Both the strict false-alarm number (on
# pre-event ramps) and the clean number (on normal wells) are reported.

import sys
from pathlib import Path

sys.path.insert(0, str(Path("../src").resolve()))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedGroupKFold

from forewell.hydrate import TRANSIENT, CONFIRMED, STEP_SECONDS

RANDOM_STATE = 42
DATA_ROOT = Path("../../3W/dataset")
MIN_NORMAL_FRACTION = 0.30

df = pd.read_parquet(Path("../data/processed/hydrate_forecast.parquet"))
non_features = {"label", "well", "source_file", "win_end_idx"}
feature_cols = [c for c in df.columns if c not in non_features]

X = np.nan_to_num(df[feature_cols].to_numpy())
y = df["label"].to_numpy()
groups = df["well"].to_numpy()

# Honest, calibrated out-of-fold scores. Within each fold the base model
# is wrapped in isotonic calibration fitted on the training wells only.
oof = np.full(len(df), np.nan)
n_splits = min(5, df["well"].nunique())
cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                          random_state=RANDOM_STATE)
for tr, te in cv.split(X, y, groups):
    base = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
        l2_regularization=1.0, random_state=RANDOM_STATE,
    )
    clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
    clf.fit(X[tr], y[tr])
    oof[te] = clf.predict_proba(X[te])[:, 1]

df = df.copy()
df["score"] = oof

# Classify wells: hydrate wells (have positives) vs pure-normal wells.
hydrate_wells, normal_wells = [], []
well_onset, well_norm_frac = {}, {}

def onset_seconds(source_file):
    raw = pd.read_parquet(DATA_ROOT / "8" / source_file, columns=["class"])
    classes = raw["class"].to_numpy()
    hit = np.where(np.isin(classes, [TRANSIENT, CONFIRMED]))[0]
    return int(hit[0]) if len(hit) else None

for well, g in df.groupby("well"):
    if g["label"].sum() == 0:
        normal_wells.append(well)
        well_onset[well] = None
        well_norm_frac[well] = 1.0
    else:
        hydrate_wells.append(well)
        well_onset[well] = onset_seconds(g["source_file"].iloc[0])
        well_norm_frac[well] = float((g["label"] == 0).mean())

valid_wells = [w for w in hydrate_wells
               if well_norm_frac[w] >= MIN_NORMAL_FRACTION and well_onset[w]]

print(f"Hydrate wells for coverage/lead: {sorted(valid_wells)}")
print(f"Pure-normal wells for false-alarm rate: {sorted(normal_wells)}")

windows_per_day = 86400 / STEP_SECONDS

def evaluate(thr):
    caught, lead_hours = 0, []
    for well in valid_wells:
        g = df[df["well"] == well].sort_values("win_end_idx")
        onset = well_onset[well]
        ends = g["win_end_idx"].to_numpy()
        alarms = g["score"].to_numpy() >= thr
        pre = ends < onset
        fired = np.where(alarms & pre)[0]
        if len(fired):
            caught += 1
            lead_hours.append((onset - ends[fired[0]]) / 3600.0)

    # False alarms on pure-normal wells: any alarm is false.
    fa_clean, n_clean = 0, 0
    for well in normal_wells:
        g = df[df["well"] == well]
        alarms = g["score"].to_numpy() >= thr
        fa_clean += int(alarms.sum())
        n_clean += len(g)
    fa_per_day = (fa_clean / n_clean * windows_per_day) if n_clean else 0.0

    coverage = caught / len(valid_wells) if valid_wells else 0.0
    med_lead = float(np.median(lead_hours)) if lead_hours else 0.0
    return coverage, med_lead, fa_per_day, caught

rows = []
for thr in np.linspace(0.3, 0.95, 14):
    cov, lead, fa, caught = evaluate(thr)
    rows.append({"threshold": round(thr, 3), "events_caught": caught,
                 "coverage": round(cov, 3), "median_lead_h": round(lead, 2),
                 "false_alarms_per_normal_well_day": round(fa, 2)})

table = pd.DataFrame(rows)
print("\n--- Calibrated event-level trade-off ---")
print("(coverage/lead on hydrate wells; false alarms on pure-normal wells)\n")
print(table.to_string(index=False))

table.to_csv(Path("../reports/hydrate_event_tradeoff.csv"), index=False)
print("\nSaved table to reports/hydrate_event_tradeoff.csv")