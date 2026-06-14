# Stronger detector with honest validation.
#
# Two upgrades over the baseline:
#   1. Gradient-boosted trees instead of linear logistic regression.
#      Trees capture interactions between sensors that a line cannot.
#   2. Grouped cross-validation by well. Instead of one random split, we
#      rotate which wells are held out across several folds and average
#      the scores. No well ever appears in both train and test within a
#      fold, so there is no leakage, and the averaged score is stable
#      rather than dependent on one lucky draw.

import sys
from pathlib import Path

sys.path.insert(0, str(Path("../src").resolve()))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    average_precision_score, precision_recall_curve,
)
import matplotlib.pyplot as plt

RANDOM_STATE = 42

df = pd.read_parquet(Path("../data/processed/features.parquet"))
print(f"Loaded {len(df)} windows from {df['well'].nunique()} wells.")

non_features = {"label", "fault_type", "well", "source_file"}
feature_cols = [c for c in df.columns if c not in non_features]

X = np.nan_to_num(df[feature_cols].to_numpy())
y = df["label"].to_numpy()
groups = df["well"].to_numpy()

# Grouped, stratified cross-validation: keep wells whole, keep class
# balance roughly even across folds.
n_splits = 5
cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                          random_state=RANDOM_STATE)

fold_ap = []
all_true = []
all_score = []

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), start=1):
    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05,
        max_leaf_nodes=31, random_state=RANDOM_STATE,
    )
    model.fit(X[train_idx], y[train_idx])
    score = model.predict_proba(X[test_idx])[:, 1]
    ap = average_precision_score(y[test_idx], score)
    fold_ap.append(ap)
    all_true.append(y[test_idx])
    all_score.append(score)
    print(f"Fold {fold}: average precision = {ap:.3f} "
          f"(test windows: {len(test_idx)})")

fold_ap = np.array(fold_ap)
print(f"\nMean average precision across {n_splits} folds: "
      f"{fold_ap.mean():.3f} (std {fold_ap.std():.3f})")

# Pool all out-of-fold predictions for a single honest report.
y_true_all = np.concatenate(all_true)
y_score_all = np.concatenate(all_score)
y_pred_all = (y_score_all >= 0.5).astype(int)

print("\n--- Pooled out-of-fold results ---")
print(classification_report(y_true_all, y_pred_all, digits=3,
                            target_names=["normal", "developing"]))
print("Confusion matrix (rows = true, cols = predicted):")
print(confusion_matrix(y_true_all, y_pred_all))

overall_ap = average_precision_score(y_true_all, y_score_all)
prec, rec, _ = precision_recall_curve(y_true_all, y_score_all)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(rec, prec, linewidth=2)
ax.set_xlabel("recall (fraction of developing faults caught)")
ax.set_ylabel("precision (fraction of alarms that were real)")
ax.set_title(f"Gradient boosting: pooled precision-recall (AP = {overall_ap:.3f})")
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
fig.tight_layout()

out_fig = Path("../reports/figures/05_gbm_pr_curve.jpg")
fig.savefig(out_fig, dpi=120, bbox_inches="tight")
print(f"\nSaved figure to {out_fig.resolve()}")