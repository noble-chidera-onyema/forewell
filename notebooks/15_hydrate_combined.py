# Combined hydrate detector across faults 8 and 9 (production-line and
# service-line hydrate formation), trained and validated on real wells.
# With both faults pooled there are dozens of real wells, enough for
# honest grouped cross-validation without leaning on simulated data.

import sys
from pathlib import Path

sys.path.insert(0, str(Path("../src").resolve()))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score, classification_report
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path("../src").resolve()))
from forewell.hydrate import build_detection_dataset_combined

RANDOM_STATE = 42
DATA_ROOT = Path("../../3W/dataset")

print("Harvesting combined hydrate dataset (faults 8 and 9, real wells)...")
df = build_detection_dataset_combined(DATA_ROOT, source="real",
                                      fault_folders=(8, 9))
print(f"Total: {len(df)} windows from {df['well'].nunique()} real wells.")
print("\nLabel balance (1 = developing hydrate, 0 = normal):")
print(df["label"].value_counts())
print(f"Positive rate: {df['label'].mean():.3f}")
print("\nWindows per fault folder:")
print(df.groupby("fault_folder")["label"].agg(["count", "sum"]))

non_features = {"label", "well", "source_file", "win_end_idx", "fault_folder"}
feature_cols = [c for c in df.columns if c not in non_features]

X = np.nan_to_num(df[feature_cols].to_numpy())
y = df["label"].to_numpy()
groups = df["well"].to_numpy()

oof = np.full(len(df), np.nan)
fold_ap = []
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
for fold, (tr, te) in enumerate(cv.split(X, y, groups), start=1):
    base = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
        l2_regularization=1.0, random_state=RANDOM_STATE,
    )
    clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
    clf.fit(X[tr], y[tr])
    oof[te] = clf.predict_proba(X[te])[:, 1]
    ap = average_precision_score(y[te], oof[te])
    fold_ap.append(ap)
    print(f"Fold {fold}: AP = {ap:.3f}  ({len(set(groups[te]))} test wells)")

fold_ap = np.array(fold_ap)
base_rate = y.mean()
print(f"\nMean AP: {fold_ap.mean():.3f} (std {fold_ap.std():.3f})")
print(f"No-skill baseline: {base_rate:.3f}")
print(f"Lift: {fold_ap.mean() / base_rate:.2f}x")

y_pred = (oof >= 0.5).astype(int)
print("\nPooled out-of-fold report:")
print(classification_report(y, y_pred, digits=3,
                            target_names=["normal", "developing hydrate"]))

df = df.copy()
df["score"] = oof
df.to_parquet(Path("../data/processed/hydrate_combined_scored.parquet"), index=False)

from sklearn.metrics import precision_recall_curve
prec, rec, _ = precision_recall_curve(y, oof)
overall_ap = average_precision_score(y, oof)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(rec, prec, linewidth=2, label=f"detector (AP={overall_ap:.3f})")
ax.axhline(base_rate, color="grey", linestyle="--", label=f"no-skill ({base_rate:.2f})")
ax.set_xlabel("recall (developing hydrates caught)")
ax.set_ylabel("precision (alarms that were real)")
ax.set_title("Combined hydrate detector (faults 8+9): precision-recall")
ax.legend(); ax.grid(True, alpha=0.3); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
fig.tight_layout()
out_fig = Path("../reports/figures/15_combined_pr_curve.jpg")
fig.savefig(out_fig, dpi=120, bbox_inches="tight")
print(f"\nSaved figure to {out_fig.resolve()}")