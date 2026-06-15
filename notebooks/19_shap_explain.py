# Explain the hydrate detector with SHAP.
#
# Two outputs:
#   1. Global feature importance: which sensors drive the detector across
#      many predictions. For a hydrate model the physics predicts
#      temperature and choke pressure-drop features should dominate; if
#      they do, the model learned real physics rather than noise.
#   2. A single-alarm explanation: for one real developing-hydrate window,
#      the top features that pushed the score toward "developing". This is
#      the operator-facing "why did it alarm" output.

import sys
from pathlib import Path

sys.path.insert(0, str(Path("../src").resolve()))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold

from forewell.hydrate import build_detection_dataset_combined

RANDOM_STATE = 42
DATA_ROOT = Path("../../3W/dataset")

df = build_detection_dataset_combined(DATA_ROOT, source="real",
                                      fault_folders=(8, 9)).reset_index(drop=True)
non_features = {"label", "well", "source_file", "win_end_idx", "fault_folder"}
feature_cols = [c for c in df.columns if c not in non_features]

X = np.nan_to_num(df[feature_cols].to_numpy())
y = df["label"].to_numpy()
groups = df["well"].to_numpy()

# Train on most wells, explain on a held-out fold (so explanations are on
# data the model did not train on).
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
train_idx, test_idx = next(cv.split(X, y, groups))

model = HistGradientBoostingClassifier(
    max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
    l2_regularization=1.0, random_state=RANDOM_STATE,
)
model.fit(X[train_idx], y[train_idx])

# SHAP on a sample of the held-out set (sampling keeps it fast).
X_test = X[test_idx]
y_test = y[test_idx]
rng = np.random.default_rng(RANDOM_STATE)
sample_n = min(800, len(X_test))
sample = rng.choice(len(X_test), sample_n, replace=False)
X_sample = X_test[sample]

print("Computing SHAP values (this takes a moment)...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)
# For binary HistGB, shap_values may be a single array for the positive
# class margin; handle both shapes.
if isinstance(shap_values, list):
    sv = shap_values[1]
else:
    sv = shap_values

# --- Global importance: mean absolute SHAP per feature ---
mean_abs = np.abs(sv).mean(axis=0)
importance = (pd.Series(mean_abs, index=feature_cols)
              .sort_values(ascending=False))

print("\n--- Top 15 features driving the detector ---")
print(importance.head(15).round(4).to_string())

fig, ax = plt.subplots(figsize=(9, 7))
importance.head(15)[::-1].plot(kind="barh", ax=ax, color="tab:blue")
ax.set_xlabel("mean |SHAP value| (impact on detection)")
ax.set_title("What drives the hydrate detector (top 15 features)")
fig.tight_layout()
out_fig = Path("../reports/figures/19_shap_global.jpg")
fig.savefig(out_fig, dpi=120, bbox_inches="tight")
print(f"Saved global importance figure to {out_fig.resolve()}")

# --- Single-alarm explanation: pick a correctly-detected developing window ---
pred = model.predict_proba(X_sample)[:, 1]
true_sample = y_test[sample]
hits = np.where((true_sample == 1) & (pred >= 0.6))[0]
if len(hits):
    i = hits[0]
    contrib = pd.Series(sv[i], index=feature_cols).sort_values()
    print(f"\n--- Explanation for one real developing-hydrate alarm "
          f"(model score {pred[i]:.2f}) ---")
    print("Top features pushing toward DEVELOPING:")
    print(contrib.tail(6)[::-1].round(4).to_string())
    print("\nTop features pushing toward NORMAL:")
    print(contrib.head(4).round(4).to_string())

    top = pd.concat([contrib.head(4), contrib.tail(6)])
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    colors = ["tab:red" if v > 0 else "tab:green" for v in top.values]
    top.plot(kind="barh", ax=ax2, color=colors)
    ax2.set_xlabel("SHAP contribution (red = toward developing, green = toward normal)")
    ax2.set_title("Why this window was flagged as a developing hydrate")
    fig2.tight_layout()
    out_fig2 = Path("../reports/figures/19_shap_single_alarm.jpg")
    fig2.savefig(out_fig2, dpi=120, bbox_inches="tight")
    print(f"Saved single-alarm explanation to {out_fig2.resolve()}")
else:
    print("\nNo high-confidence developing window in the sample to explain.")