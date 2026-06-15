# Train and validate the hydrate forecaster.
#
# Judged two ways: standard average precision under grouped cross-
# validation by well (no leakage), and the operational question that
# matters, how far ahead of the event the forecaster fires. The dataset
# is small (9 wells), so per-fold variance is reported honestly rather
# than hidden behind a single number.

import sys
from pathlib import Path

sys.path.insert(0, str(Path("../src").resolve()))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score, classification_report
import matplotlib.pyplot as plt

RANDOM_STATE = 42

df = pd.read_parquet(Path("../data/processed/hydrate_forecast.parquet"))
non_features = {"label", "well", "source_file"}
feature_cols = [c for c in df.columns if c not in non_features]

X = np.nan_to_num(df[feature_cols].to_numpy())
y = df["label"].to_numpy()
groups = df["well"].to_numpy()

n_wells = df["well"].nunique()
n_splits = min(5, n_wells)

cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                          random_state=RANDOM_STATE)

oof = np.full(len(df), np.nan)
fold_ap = []

for fold, (tr, te) in enumerate(cv.split(X, y, groups), start=1):
    model = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
        l2_regularization=1.0, random_state=RANDOM_STATE,
    )
    model.fit(X[tr], y[tr])
    score = model.predict_proba(X[te])[:, 1]
    oof[te] = score
    ap = average_precision_score(y[te], score)
    fold_ap.append(ap)
    test_wells = sorted(set(groups[te]))
    print(f"Fold {fold}: AP = {ap:.3f}  test wells = {test_wells}")

fold_ap = np.array(fold_ap)
base_rate = y.mean()
print(f"\nMean AP: {fold_ap.mean():.3f} (std {fold_ap.std():.3f})")
print(f"No-skill baseline (positive rate): {base_rate:.3f}")
print(f"Lift over baseline: {fold_ap.mean() / base_rate:.2f}x")

y_pred = (oof >= 0.5).astype(int)
print("\n--- Pooled out-of-fold report ---")
print(classification_report(y, y_pred, digits=3,
                            target_names=["no hydrate soon", "hydrate within 6h"]))

prec, rec, _ = __import__("sklearn.metrics", fromlist=["precision_recall_curve"]).precision_recall_curve(y, oof)
overall_ap = average_precision_score(y, oof)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(rec, prec, linewidth=2, label=f"forecaster (AP={overall_ap:.3f})")
ax.axhline(base_rate, color="grey", linestyle="--",
           label=f"no-skill ({base_rate:.2f})")
ax.set_xlabel("recall (fraction of coming hydrates caught)")
ax.set_ylabel("precision (fraction of alarms that were real)")
ax.set_title("Hydrate 6-hour forecaster: precision-recall")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
fig.tight_layout()
out_fig = Path("../reports/figures/10_hydrate_pr_curve.jpg")
fig.savefig(out_fig, dpi=120, bbox_inches="tight")
print(f"\nSaved figure to {out_fig.resolve()}")