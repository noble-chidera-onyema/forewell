# Conformal prediction: turn raw model scores into calibrated alarms
# with a coverage guarantee.
#
# A raw score of 0.62 has no guaranteed meaning. Conformal prediction
# uses a held-out calibration set to build prediction sets that satisfy a
# chosen coverage level: across many predictions, the true label lands in
# the set at least (1 - alpha) of the time. For an alarm system this lets
# the model give a confident single answer when it can, and abstain (a set
# containing both classes) when the evidence is ambiguous.
#
# We use split conformal with a class-wise nonconformity score, validated
# on wells never seen during training or calibration.

import sys
from pathlib import Path

sys.path.insert(0, str(Path("../src").resolve()))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

RANDOM_STATE = 42
ALPHA = 0.10          # target: at least 90% coverage
TARGET_FAULT = 1      # strongest dedicated model

rng = np.random.default_rng(RANDOM_STATE)

df = pd.read_parquet(Path("../data/processed/features.parquet"))
non_features = {"label", "fault_type", "well", "source_file"}
feature_cols = [c for c in df.columns if c not in non_features]

# This fault's positives plus all normal windows.
positives = df[(df["label"] == 1) & (df["fault_type"] == TARGET_FAULT)]
normal = df[df["label"] == 0]
sub = pd.concat([positives, normal], ignore_index=True)

# Split by WELL into train / calibrate / test, so coverage is tested on
# wells unseen in both fitting and calibration.
wells = sub["well"].unique().tolist()
rng.shuffle(wells)
n = len(wells)
train_wells = set(wells[: int(0.5 * n)])
calib_wells = set(wells[int(0.5 * n): int(0.75 * n)])
test_wells = set(wells[int(0.75 * n):])

def split(frame, well_set):
    part = frame[frame["well"].isin(well_set)]
    X = np.nan_to_num(part[feature_cols].to_numpy())
    y = part["label"].to_numpy()
    return X, y

X_tr, y_tr = split(sub, train_wells)
X_cal, y_cal = split(sub, calib_wells)
X_te, y_te = split(sub, test_wells)

print(f"Fault {TARGET_FAULT}: train {len(y_tr)}, "
      f"calibrate {len(y_cal)}, test {len(y_te)} windows.")
print(f"Wells -> train {len(train_wells)}, calib {len(calib_wells)}, "
      f"test {len(test_wells)}")

model = HistGradientBoostingClassifier(
    max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
    random_state=RANDOM_STATE,
)
model.fit(X_tr, y_tr)

# Nonconformity score = 1 - probability of the true class.
proba_cal = model.predict_proba(X_cal)
nonconf_cal = 1.0 - proba_cal[np.arange(len(y_cal)), y_cal]

# Class-wise thresholds give more reliable coverage under imbalance.
classes = [0, 1]
thresholds = {}
for c in classes:
    scores_c = nonconf_cal[y_cal == c]
    if len(scores_c) == 0:
        thresholds[c] = 1.0
        continue
    # Conformal quantile with finite-sample correction.
    k = int(np.ceil((len(scores_c) + 1) * (1 - ALPHA)))
    k = min(k, len(scores_c))
    thresholds[c] = np.sort(scores_c)[k - 1]

print(f"\nClass-wise nonconformity thresholds (alpha={ALPHA}):")
print(f"  normal (0):     {thresholds[0]:.3f}")
print(f"  developing (1): {thresholds[1]:.3f}")

# Build prediction sets on the unseen test wells.
proba_te = model.predict_proba(X_te)
in_set = np.zeros((len(y_te), 2), dtype=bool)
for c in classes:
    nonconf_c = 1.0 - proba_te[:, c]
    in_set[:, c] = nonconf_c <= thresholds[c]

set_sizes = in_set.sum(axis=1)
covered = in_set[np.arange(len(y_te)), y_te]

print("\n--- Results on unseen test wells ---")
print(f"Empirical coverage: {covered.mean():.3f}  (target {1 - ALPHA:.2f})")
print(f"Mean prediction-set size: {set_sizes.mean():.3f}")
print("\nSet-size distribution:")
print(f"  size 0 (abstain, no class fits): {(set_sizes == 0).mean():.3f}")
print(f"  size 1 (confident single answer): {(set_sizes == 1).mean():.3f}")
print(f"  size 2 (uncertain, flag for human): {(set_sizes == 2).mean():.3f}")

# Of the confident single-answer cases, how accurate are they?
single = set_sizes == 1
if single.sum() > 0:
    single_pred = in_set[single, 1].astype(int)  # predicted developing?
    single_true = y_te[single]
    acc = (single_pred == single_true).mean()
    print(f"\nAccuracy when the model gives a confident single answer: {acc:.3f}")
    print(f"Fraction of all windows where it is confident: {single.mean():.3f}")