# Conformal prediction done correctly for grouped (multi-well) data.
#
# Naive split-conformal under-covered here (0.65 vs 0.90 target) because
# wells are not exchangeable: calibration wells differ from test wells, so
# a threshold learned on one set does not transfer. The correct approach
# for grouped data calibrates and evaluates within the same grouped
# cross-validation, so the calibration distribution matches the evaluation
# distribution. We use leave-one-fold-out: for each fold, calibrate the
# conformal threshold on the other folds' out-of-fold scores and apply it
# to this fold, then pool coverage across all folds.

import sys
from pathlib import Path

sys.path.insert(0, str(Path("../src").resolve()))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold

from forewell.hydrate import build_detection_dataset_combined

RANDOM_STATE = 42
ALPHA = 0.10
DATA_ROOT = Path("../../3W/dataset")

df = build_detection_dataset_combined(DATA_ROOT, source="real",
                                      fault_folders=(8, 9)).reset_index(drop=True)
non_features = {"label", "well", "source_file", "win_end_idx", "fault_folder"}
feature_cols = [c for c in df.columns if c not in non_features]

X = np.nan_to_num(df[feature_cols].to_numpy())
y = df["label"].to_numpy()
groups = df["well"].to_numpy()

# Step 1: honest out-of-fold probabilities (each well scored by a model
# that never trained on it).
oof_proba = np.full((len(df), 2), np.nan)
fold_id = np.full(len(df), -1)
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
for fid, (tr, te) in enumerate(cv.split(X, y, groups)):
    m = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
        l2_regularization=1.0, random_state=RANDOM_STATE,
    )
    m.fit(X[tr], y[tr])
    oof_proba[te] = m.predict_proba(X[te])
    fold_id[te] = fid

# Nonconformity for the true class, from out-of-fold scores.
nonconf = 1.0 - oof_proba[np.arange(len(y)), y]

# Step 2: leave-one-fold-out conformal. For each fold, the threshold comes
# from the OTHER folds' nonconformity scores, then is applied to this fold.
in_set = np.zeros((len(y), 2), dtype=bool)
for fid in range(5):
    te = fold_id == fid
    cal = (fold_id != fid)
    cal_scores = nonconf[cal]
    k = int(np.ceil((cal_scores.size + 1) * (1 - ALPHA)))
    k = min(k, cal_scores.size)
    q = float(np.sort(cal_scores)[k - 1])
    for c in (0, 1):
        in_set[te, c] = (1.0 - oof_proba[te, c]) <= q

set_sizes = in_set.sum(axis=1)
covered = in_set[np.arange(len(y)), y]

print("--- Grouped conformal (leave-one-fold-out calibration) ---")
print(f"Empirical coverage: {covered.mean():.3f}  (target {1 - ALPHA:.2f})")
print(f"Mean set size: {set_sizes.mean():.3f}")
print(f"  confident single answer: {(set_sizes == 1).mean():.3f}")
print(f"  uncertain, escalate:     {(set_sizes == 2).mean():.3f}")
print(f"  empty:                   {(set_sizes == 0).mean():.3f}")

# Coverage per well: does the guarantee hold across wells, or only on average?
df = df.copy()
df["covered"] = covered
df["set_size"] = set_sizes
per_well = df.groupby("well")["covered"].mean().round(3)
print("\nPer-well coverage (shows where the guarantee holds or fails):")
print(per_well.to_string())
print(f"\nWells meeting >= 0.85 coverage: "
      f"{(per_well >= 0.85).sum()} of {len(per_well)}")