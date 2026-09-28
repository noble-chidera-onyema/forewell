# Train the production hydrate detector once and save it for serving.
#
# A deployed app loads a saved model; it does not retrain on every visit.
# This trains on all real wells, fits the conformal threshold, and saves
# the model, threshold, and feature list as one artifact. It also saves a
# few real wells' sensor traces so the dashboard can stream genuine data
# without the full dataset being online.

import sys
from pathlib import Path

sys.path.insert(0, str(Path("../src").resolve()))

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier

from forewell.hydrate import (
    build_detection_dataset_combined, _build_detection_for_transient,
    SENSOR_COLUMNS,
)

RANDOM_STATE = 42
ALPHA = 0.10
DATA_ROOT = Path("../../3W/dataset")
ARTIFACT_DIR = Path("../app/artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

print("Building combined hydrate dataset...")
df = build_detection_dataset_combined(DATA_ROOT, source="real",
                                      fault_folders=(8, 9)).reset_index(drop=True)
non_features = {"label", "well", "source_file", "win_end_idx", "fault_folder"}
feature_cols = [c for c in df.columns if c not in non_features]

X = np.nan_to_num(df[feature_cols].to_numpy())
y = df["label"].to_numpy()

# Conformal threshold via grouped cross-validation, the method validated
# in notebook 18. A random row split collapses the threshold to zero
# because calibration rows share wells with training rows; grouping by
# well keeps calibration honest.
from sklearn.model_selection import StratifiedGroupKFold

groups = df["well"].to_numpy()
oof_proba = np.full((len(df), 2), np.nan)
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
for tr, te in cv.split(X, y, groups):
    m = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
        l2_regularization=1.0, random_state=RANDOM_STATE,
    )
    m.fit(X[tr], y[tr])
    oof_proba[te] = m.predict_proba(X[te])

nonconf = 1.0 - oof_proba[np.arange(len(y)), y]
k = int(np.ceil((len(nonconf) + 1) * (1 - ALPHA)))
k = min(k, len(nonconf))
q_hat = float(np.sort(nonconf)[k - 1])
print(f"Conformal threshold (grouped, alpha={ALPHA}): {q_hat:.3f}")

# Retrain on ALL data for the final served model (more data = better).
model_full = HistGradientBoostingClassifier(
    max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
    l2_regularization=1.0, random_state=RANDOM_STATE,
)
model_full.fit(X, y)

artifact = {
    "model": model_full,
    "feature_cols": feature_cols,
    "conformal_q": q_hat,
    "alpha": ALPHA,
    "sensor_columns": SENSOR_COLUMNS,
}
joblib.dump(artifact, ARTIFACT_DIR / "forewell_model.joblib")
print(f"Saved model artifact to {ARTIFACT_DIR / 'forewell_model.joblib'}")

# Save a few real wells' windowed feature rows (with labels and time index)
# so the dashboard can stream genuine data. Pick wells with clear events.
demo_wells = ["WELL-00019", "WELL-00026", "WELL-00025", "WELL-00029"]
demo = df[df["well"].isin(demo_wells)].copy()
demo.to_parquet(ARTIFACT_DIR / "demo_wells.parquet", index=False)
print(f"Saved demo data ({len(demo)} windows from "
      f"{demo['well'].nunique()} wells) to {ARTIFACT_DIR / 'demo_wells.parquet'}")

print("\nDone. Artifact and demo data ready for the dashboard.")