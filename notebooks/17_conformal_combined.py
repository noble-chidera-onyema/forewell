# Conformal prediction on the combined hydrate detector.
#
# The detector is confident on clear cases and uncertain in the early
# developing window. Conformal prediction quantifies that uncertainty with
# a coverage guarantee: for a chosen level, the prediction set contains the
# true label at least that often, validated on wells unseen in both
# training and calibration. Outputs are:
#   {developing}        -> confident alarm
#   {normal}            -> confident quiet
#   {normal,developing} -> genuinely uncertain, escalate to a human
# The ability to abstain honestly is the point for safety-critical use.

import sys
from pathlib import Path

sys.path.insert(0, str(Path("../src").resolve()))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from forewell.hydrate import build_detection_dataset_combined

RANDOM_STATE = 42
ALPHA = 0.10   # target coverage 90%
DATA_ROOT = Path("../../3W/dataset")
rng = np.random.default_rng(RANDOM_STATE)

df = build_detection_dataset_combined(DATA_ROOT, source="real",
                                      fault_folders=(8, 9))
non_features = {"label", "well", "source_file", "win_end_idx", "fault_folder"}
feature_cols = [c for c in df.columns if c not in non_features]

# Three-way split by WELL: train / calibrate / test, all disjoint.
wells = df["well"].unique().tolist()
rng.shuffle(wells)
n = len(wells)
train_w = set(wells[: int(0.5 * n)])
calib_w = set(wells[int(0.5 * n): int(0.75 * n)])
test_w = set(wells[int(0.75 * n):])

def part(well_set):
    p = df[df["well"].isin(well_set)]
    return (np.nan_to_num(p[feature_cols].to_numpy()),
            p["label"].to_numpy())

X_tr, y_tr = part(train_w)
X_cal, y_cal = part(calib_w)
X_te, y_te = part(test_w)

print(f"Train wells {len(train_w)} ({len(y_tr)} windows), "
      f"calib {len(calib_w)} ({len(y_cal)}), test {len(test_w)} ({len(y_te)}).")

model = HistGradientBoostingClassifier(
    max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
    l2_regularization=1.0, random_state=RANDOM_STATE,
)
model.fit(X_tr, y_tr)

# Split-conformal with a single global nonconformity quantile.
proba_cal = model.predict_proba(X_cal)
nonconf_cal = 1.0 - proba_cal[np.arange(len(y_cal)), y_cal]
k = int(np.ceil((len(nonconf_cal) + 1) * (1 - ALPHA)))
k = min(k, len(nonconf_cal))
q_hat = float(np.sort(nonconf_cal)[k - 1])
print(f"\nGlobal nonconformity threshold (alpha={ALPHA}): {q_hat:.3f}")

# Build prediction sets on unseen test wells.
proba_te = model.predict_proba(X_te)
in_set = np.zeros((len(y_te), 2), dtype=bool)
for c in (0, 1):
    in_set[:, c] = (1.0 - proba_te[:, c]) <= q_hat

set_sizes = in_set.sum(axis=1)
covered = in_set[np.arange(len(y_te)), y_te]

print("\n--- Conformal results on unseen test wells ---")
print(f"Empirical coverage: {covered.mean():.3f}  (target {1 - ALPHA:.2f})")
print(f"Mean set size: {set_sizes.mean():.3f}")
print(f"  confident single answer (size 1): {(set_sizes == 1).mean():.3f}")
print(f"  uncertain, escalate (size 2):     {(set_sizes == 2).mean():.3f}")
print(f"  empty (size 0):                   {(set_sizes == 0).mean():.3f}")

single = set_sizes == 1
if single.sum() > 0:
    pred_single = in_set[single, 1].astype(int)
    acc_single = (pred_single == y_te[single]).mean()
    print(f"\nWhen confident (single answer): accuracy {acc_single:.3f}, "
          f"covering {single.mean():.3f} of all windows.")