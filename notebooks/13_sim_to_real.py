# Train on simulated hydrate instances, validate on real wells.
#
# Real hydrate events are too few (4 usable wells) to train a robust
# model. The dataset provides physics-simulated instances for exactly this
# case. Here the model learns hydrate structure from 81 simulated events,
# then is tested only on real wells it never saw in any form. A model
# trained purely on simulation that still forecasts real hydrates is a
# strong, honest result, and it removes the dependence on the tiny real
# sample for training.

import sys
from pathlib import Path

sys.path.insert(0, str(Path("../src").resolve()))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, classification_report

from forewell.hydrate import (
    build_hydrate_dataset, build_simulated_dataset, build_normal_negatives,
    TRANSIENT, CONFIRMED, STEP_SECONDS,
)

RANDOM_STATE = 42
DATA_ROOT = Path("../../3W/dataset")

print("Harvesting simulated hydrate instances for training...")
sim = build_simulated_dataset(DATA_ROOT)
print(f"  simulated windows: {len(sim)} from {sim['well'].nunique()} "
      f"instances (positives {int(sim['label'].sum())})")

print("Harvesting normal negatives for training context...")
norm = build_normal_negatives(DATA_ROOT, max_files=60)
print(f"  normal windows: {len(norm)}")

print("Loading real hydrate wells for testing...")
real = build_hydrate_dataset(DATA_ROOT)
print(f"  real windows: {len(real)} from {real['well'].nunique()} wells "
      f"(positives {int(real['label'].sum())})")

# Training set: simulated events + normal context. No real wells.
train = pd.concat([sim, norm], ignore_index=True)
non_features = {"label", "well", "source_file", "win_end_idx"}
feature_cols = [c for c in train.columns if c not in non_features]

# Align columns in case real has any the train set lacks or vice versa.
for col in feature_cols:
    if col not in real.columns:
        real[col] = 0.0
feature_cols = [c for c in feature_cols if c in real.columns]

X_train = np.nan_to_num(train[feature_cols].to_numpy())
y_train = train["label"].to_numpy()
X_real = np.nan_to_num(real[feature_cols].to_numpy())
y_real = real["label"].to_numpy()

print(f"\nTraining on {len(X_train)} simulated+normal windows "
      f"(positive rate {y_train.mean():.3f}).")
print(f"Testing on {len(X_real)} real windows "
      f"(positive rate {y_real.mean():.3f}).")

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
print(f"\n--- Trained on simulation, tested on real wells ---")
print(f"Average precision on real: {ap:.3f}")
print(f"No-skill baseline: {base_rate:.3f}")
print(f"Lift: {ap / base_rate:.2f}x")

y_pred = (real["score"] >= 0.5).astype(int)
print("\nClassification report on real windows (threshold 0.5):")
print(classification_report(y_real, y_pred, digits=3,
                            target_names=["no hydrate soon", "hydrate within 8h"]))

# Save real scores for the event-level evaluation next.
real.to_parquet(Path("../data/processed/hydrate_real_scored.parquet"),
                index=False)
print("Saved scored real wells to data/processed/hydrate_real_scored.parquet")