# Per-fault-type breakdown of the detector.
#
# A single pooled score hides the real story. What matters operationally
# is which faults the model catches early and which it misses. For each
# fault type we hold out its windows, score the model on them, and report
# the result next to the early-warning lead time measured earlier. The
# pattern that emerges links model performance to fault physics: faults
# that develop slowly give long warning and are caught well; fast faults
# give little warning and are hard.

import sys
from pathlib import Path

sys.path.insert(0, str(Path("../src").resolve()))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score, recall_score

RANDOM_STATE = 42

FAULT_NAMES = {
    1: "BSW increase",
    2: "DHSV spurious closure",
    5: "Rapid productivity loss",
    6: "Quick choke restriction",
    7: "Choke scaling",
    8: "Hydrate in production line",
    9: "Hydrate in service line",
}

df = pd.read_parquet(Path("../data/processed/features.parquet"))
non_features = {"label", "fault_type", "well", "source_file"}
feature_cols = [c for c in df.columns if c not in non_features]

X = np.nan_to_num(df[feature_cols].to_numpy())
y = df["label"].to_numpy()
groups = df["well"].to_numpy()
fault_type = df["fault_type"].to_numpy()

# Collect out-of-fold predictions for every window, then slice by fault.
oof_score = np.full(len(df), np.nan)

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
for train_idx, test_idx in cv.split(X, y, groups):
    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05,
        max_leaf_nodes=31, random_state=RANDOM_STATE,
    )
    model.fit(X[train_idx], y[train_idx])
    oof_score[test_idx] = model.predict_proba(X[test_idx])[:, 1]

# Load the lead-time medians measured earlier.
lead_path = Path("../reports/lead_time_by_instance.csv")
lead_median = {}
if lead_path.exists():
    lead_df = pd.read_csv(lead_path)
    lead_median = (
        lead_df.groupby("fault_label")["lead_time_minutes"]
        .median().round(1).to_dict()
    )

rows = []
for fault in sorted(FAULT_NAMES):
    # Positive windows for this fault, against all normal windows.
    mask = (fault_type == fault) | (y == 0)
    yt = y[mask]
    ys = oof_score[mask]
    valid = ~np.isnan(ys)
    yt, ys = yt[valid], ys[valid]

    n_pos = int((yt == 1).sum())
    if n_pos == 0:
        continue

    ap = average_precision_score(yt, ys)
    recall_at_50 = recall_score(yt, (ys >= 0.5).astype(int), zero_division=0)

    rows.append({
        "fault": fault,
        "name": FAULT_NAMES[fault],
        "positive_windows": n_pos,
        "median_lead_min": lead_median.get(fault, np.nan),
        "avg_precision": round(ap, 3),
        "recall_at_0.5": round(recall_at_50, 3),
    })

result = pd.DataFrame(rows).sort_values("median_lead_min")

print("\n--- Per-fault early-warning performance ---")
print("(sorted by median lead time: fast faults at top, slow at bottom)\n")
print(result.to_string(index=False))

out_csv = Path("../reports/per_fault_performance.csv")
result.to_csv(out_csv, index=False)
print(f"\nSaved table to {out_csv.resolve()}")