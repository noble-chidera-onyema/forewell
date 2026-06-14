# One model per fault type, instead of one blurred detector for all.
#
# Asking a single model to call both a sharp valve closure and a slow
# scaling event "developing" forces one decision boundary to cover
# physically different signatures, and the subtle faults lose out. Here
# each fault gets its own model that learns its own signature against
# normal operation. Faults with too few positive windows to train a
# trustworthy model are reported as insufficient data, not given a
# fabricated score.

import sys
from pathlib import Path

sys.path.insert(0, str(Path("../src").resolve()))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score, recall_score

RANDOM_STATE = 42
MIN_POSITIVE = 500   # below this, do not train a per-fault model

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

lead_path = Path("../reports/lead_time_by_instance.csv")
lead_median = {}
if lead_path.exists():
    lead_df = pd.read_csv(lead_path)
    lead_median = (lead_df.groupby("fault_label")["lead_time_minutes"]
                   .median().round(1).to_dict())

normal = df[df["label"] == 0]
rows = []

for fault in sorted(FAULT_NAMES):
    positives = df[(df["label"] == 1) & (df["fault_type"] == fault)]
    n_pos = len(positives)

    if n_pos < MIN_POSITIVE:
        rows.append({
            "fault": fault, "name": FAULT_NAMES[fault],
            "positive_windows": n_pos,
            "median_lead_min": lead_median.get(fault, np.nan),
            "avg_precision": "insufficient data",
            "recall_at_0.5": "insufficient data",
        })
        continue

    # This fault's positives versus all normal windows.
    sub = pd.concat([positives, normal], ignore_index=True)
    X = np.nan_to_num(sub[feature_cols].to_numpy())
    y = sub["label"].to_numpy()
    groups = sub["well"].to_numpy()

    oof = np.full(len(sub), np.nan)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True,
                              random_state=RANDOM_STATE)
    for tr, te in cv.split(X, y, groups):
        model = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05,
            max_leaf_nodes=31, random_state=RANDOM_STATE,
        )
        model.fit(X[tr], y[tr])
        oof[te] = model.predict_proba(X[te])[:, 1]

    valid = ~np.isnan(oof)
    ap = average_precision_score(y[valid], oof[valid])
    rec = recall_score(y[valid], (oof[valid] >= 0.5).astype(int),
                       zero_division=0)

    rows.append({
        "fault": fault, "name": FAULT_NAMES[fault],
        "positive_windows": n_pos,
        "median_lead_min": lead_median.get(fault, np.nan),
        "avg_precision": round(ap, 3),
        "recall_at_0.5": round(rec, 3),
    })

result = pd.DataFrame(rows)
print("\n--- Per-fault dedicated models ---\n")
print(result.to_string(index=False))

out_csv = Path("../reports/per_fault_dedicated.csv")
result.to_csv(out_csv, index=False)
print(f"\nSaved table to {out_csv.resolve()}")