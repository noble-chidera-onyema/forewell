# At what lead time is a hydrate actually predictable from these sensors?
#
# The 6-hour horizon was an arbitrary starting choice. This sweep rebuilds
# the forecasting dataset at several horizons and measures the model's lift
# over the no-skill baseline at each. The horizon where lift is highest is
# the honest, deployable forecast window for this fault.

import sys
from pathlib import Path

sys.path.insert(0, str(Path("../src").resolve()))

import importlib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt

import forewell.hydrate as H

RANDOM_STATE = 42
DATA_ROOT = Path("../../3W/dataset")
HORIZONS = [1, 2, 3, 4, 6, 8, 12]

results = []

for horizon in HORIZONS:
    # Rebuild the dataset at this horizon by overriding the module constant.
    H.HORIZON_HOURS = horizon
    importlib.reload  # no-op guard; constant is read inside the function
    df = H.build_hydrate_dataset(DATA_ROOT)
    if df.empty or df["label"].sum() < 30:
        results.append({"horizon_h": horizon, "positives": int(df["label"].sum()),
                        "mean_ap": np.nan, "baseline": np.nan, "lift": np.nan})
        continue

    non_features = {"label", "well", "source_file"}
    feature_cols = [c for c in df.columns if c not in non_features]
    X = np.nan_to_num(df[feature_cols].to_numpy())
    y = df["label"].to_numpy()
    groups = df["well"].to_numpy()

    n_splits = min(5, df["well"].nunique())
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                              random_state=RANDOM_STATE)

    aps = []
    for tr, te in cv.split(X, y, groups):
        model = HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
            l2_regularization=1.0, random_state=RANDOM_STATE,
        )
        model.fit(X[tr], y[tr])
        score = model.predict_proba(X[te])[:, 1]
        aps.append(average_precision_score(y[te], score))

    mean_ap = float(np.mean(aps))
    base = float(y.mean())
    results.append({
        "horizon_h": horizon,
        "positives": int(y.sum()),
        "mean_ap": round(mean_ap, 3),
        "baseline": round(base, 3),
        "lift": round(mean_ap / base, 2),
    })
    print(f"Horizon {horizon}h: positives={int(y.sum())}, "
          f"AP={mean_ap:.3f}, baseline={base:.3f}, lift={mean_ap/base:.2f}x")

table = pd.DataFrame(results)
print("\n--- Horizon sweep ---")
print(table.to_string(index=False))

out_csv = Path("../reports/hydrate_horizon_sweep.csv")
table.to_csv(out_csv, index=False)

valid = table.dropna()
if not valid.empty:
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(valid["horizon_h"], valid["lift"], marker="o", linewidth=2)
    ax.axhline(1.0, color="grey", linestyle="--", label="no skill")
    ax.set_xlabel("forecast horizon (hours ahead)")
    ax.set_ylabel("lift over no-skill baseline")
    ax.set_title("Hydrate predictability versus forecast horizon")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_fig = Path("../reports/figures/11_horizon_sweep.jpg")
    fig.savefig(out_fig, dpi=120, bbox_inches="tight")
    print(f"\nSaved figure to {out_fig.resolve()}")

print(f"Saved table to {out_csv.resolve()}")