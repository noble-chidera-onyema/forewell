# First detector: a logistic-regression baseline for early-warning.
#
# Two principles drive this script:
#   1. Split by WELL, never by row. The model is tested only on wells it
#      never saw in training, so the scores reflect real generalisation
#      and not memorisation of a specific well.
#   2. Judge with precision and recall on the fault class, not accuracy.
#      Recall = of all developing faults, how many did we catch.
#      Precision = when we alarmed, how often was it a real developing fault.

import sys
from pathlib import Path

sys.path.insert(0, str(Path("../src").resolve()))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_curve, average_precision_score,
)
import matplotlib.pyplot as plt

RANDOM_STATE = 42

data_path = Path("../data/processed/features.parquet")
df = pd.read_parquet(data_path)
print(f"Loaded {len(df)} windows from {df['well'].nunique()} wells.")

# Feature columns are everything except labels and bookkeeping.
non_features = {"label", "fault_type", "well", "source_file"}
feature_cols = [c for c in df.columns if c not in non_features]
print(f"Using {len(feature_cols)} features.")

# --- Split by well ---
wells = df["well"].unique()
rng = np.random.default_rng(RANDOM_STATE)
rng.shuffle(wells)

n_test = max(1, int(round(0.3 * len(wells))))
test_wells = set(wells[:n_test])
train_wells = set(wells[n_test:])

train_df = df[df["well"].isin(train_wells)]
test_df = df[df["well"].isin(test_wells)]

print(f"\nTrain wells ({len(train_wells)}): {sorted(train_wells)}")
print(f"Test wells ({len(test_wells)}): {sorted(test_wells)}")
print(f"Train windows: {len(train_df)}, Test windows: {len(test_df)}")

X_train = train_df[feature_cols].to_numpy()
y_train = train_df["label"].to_numpy()
X_test = test_df[feature_cols].to_numpy()
y_test = test_df["label"].to_numpy()

# Replace any leftover NaN with 0 so the model can run.
X_train = np.nan_to_num(X_train)
X_test = np.nan_to_num(X_test)

# --- Baseline model: scaled logistic regression ---
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, class_weight="balanced",
                       random_state=RANDOM_STATE),
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_score = model.predict_proba(X_test)[:, 1]

print("\n--- Baseline results on unseen wells ---")
print(classification_report(y_test, y_pred, digits=3,
                            target_names=["normal", "developing"]))
print("Confusion matrix (rows = true, cols = predicted):")
print(confusion_matrix(y_test, y_pred))

ap = average_precision_score(y_test, y_score)
print(f"\nAverage precision (area under PR curve): {ap:.3f}")

# --- Precision-recall curve figure ---
prec, rec, _ = precision_recall_curve(y_test, y_score)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(rec, prec, linewidth=2)
ax.set_xlabel("recall (fraction of developing faults caught)")
ax.set_ylabel("precision (fraction of alarms that were real)")
ax.set_title(f"Baseline detector: precision-recall (AP = {ap:.3f})")
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
fig.tight_layout()

out_fig = Path("../reports/figures/04_baseline_pr_curve.jpg")
fig.savefig(out_fig, dpi=120, bbox_inches="tight")
print(f"\nSaved figure to {out_fig.resolve()}")