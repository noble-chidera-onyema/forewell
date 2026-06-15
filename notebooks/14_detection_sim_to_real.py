# Detect a developing hydrate: train on simulated transient + normal data,
# test on real wells. Positive = developing-transient hydrate (class 108),
# negative = normal operation. The transient precedes confirmation by a
# long margin, so detecting it is genuine early warning. The model never
# sees a real hydrate well in training, so real-well performance is an
# honest test of transfer from simulation to reality.

import sys
from pathlib import Path

sys.path.insert(0, str(Path("../src").resolve()))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, classification_report

from forewell.hydrate import build_detection_dataset, build_normal_negatives

RANDOM_STATE = 42
DATA_ROOT = Path("../../3W/dataset")

print("Harvesting simulated transient instances (training)...")
sim = build_detection_dataset(DATA_ROOT, source="sim")
print(f"  simulated windows: {len(sim)} "
      f"(transient {int(sim['label'].sum())}, normal {int((sim['label']==0).sum())})")

print("Harvesting normal-well data (training negatives)...")
norm = build_normal_negatives(DATA_ROOT, max_files=60)
# build_normal_negatives uses the forecasting feature names; align later.
print(f"  normal windows: {len(norm)}")

print("Harvesting real hydrate wells (testing)...")
real = build_detection_dataset(DATA_ROOT, source="real")
print(f"  real windows: {len(real)} from {real['well'].nunique()} wells "
      f"(transient {int(real['label'].sum())}, normal {int((real['label']==0).sum())})")

# Use simulated transient positives + simulated normal + extra normal wells.
# Align features to the detection feature set (intersection of columns).
non_features = {"label", "well", "source_file", "win_end_idx"}
sim_feats = [c for c in sim.columns if c not in non_features]

# Bring normal-well negatives into the detection feature space: keep only
# detection features that also exist there; fill any missing with 0.
train_parts = [sim]
norm2 = norm.copy()
for c in sim_feats:
    if c not in norm2.columns:
        norm2[c] = 0.0
norm2 = norm2[[c for c in sim.columns if c in norm2.columns]]
norm2["label"] = 0
train_parts.append(norm2)

train = pd.concat(train_parts, ignore_index=True)
feature_cols = [c for c in sim_feats if c in train.columns and c in real.columns]

X_train = np.nan_to_num(train[feature_cols].to_numpy())
y_train = train["label"].to_numpy()
X_real = np.nan_to_num(real[feature_cols].to_numpy())
y_real = real["label"].to_numpy()

print(f"\nFeatures used: {len(feature_cols)}")
print(f"Train: {len(X_train)} windows (positive rate {y_train.mean():.3f})")
print(f"Test (real): {len(X_real)} windows (positive rate {y_real.mean():.3f})")

base = HistGradientBoostingClassifier(
    max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
    l2_regularization=1.0, random_state=RANDOM_STATE,
)
clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
clf.fit(X_train, y_train)

real = real.copy()
real["score"] = clf.predict_proba(X_real)[:, 1]

ap = average_precision_score(y_real, real["score"])
base_rate = y_real.mean()
print(f"\n--- Trained on simulation+normal, tested on REAL wells ---")
print(f"Average precision on real: {ap:.3f}")
print(f"No-skill baseline: {base_rate:.3f}")
print(f"Lift: {ap / base_rate:.2f}x")

y_pred = (real["score"] >= 0.5).astype(int)
print("\nReal-well classification report (threshold 0.5):")
print(classification_report(y_real, y_pred, digits=3,
                            target_names=["normal", "developing hydrate"]))

real.to_parquet(Path("../data/processed/hydrate_real_scored.parquet"), index=False)
print("Saved scored real wells.")