# Is the detector resting on one feature (QGL drift), or does it have real
# distributed signal? SHAP showed QGL_drift_base dominating by a wide
# margin. If that feature is doing physics, removing it should hurt only
# moderately. If the model was leaning on a dataset artifact, removing it
# should collapse performance. This ablation decides which.

import sys
from pathlib import Path

sys.path.insert(0, str(Path("../src").resolve()))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score

from forewell.hydrate import build_detection_dataset_combined

RANDOM_STATE = 42
DATA_ROOT = Path("../../3W/dataset")

df = build_detection_dataset_combined(DATA_ROOT, source="real",
                                      fault_folders=(8, 9)).reset_index(drop=True)
non_features = {"label", "well", "source_file", "win_end_idx", "fault_folder"}
all_features = [c for c in df.columns if c not in non_features]

y = df["label"].to_numpy()
groups = df["well"].to_numpy()


def cv_ap(feature_cols, tag):
    X = np.nan_to_num(df[feature_cols].to_numpy())
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    aps = []
    for tr, te in cv.split(X, y, groups):
        m = HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
            l2_regularization=1.0, random_state=RANDOM_STATE,
        )
        m.fit(X[tr], y[tr])
        aps.append(average_precision_score(y[te], m.predict_proba(X[te])[:, 1]))
    aps = np.array(aps)
    print(f"{tag}: mean AP {aps.mean():.3f} (std {aps.std():.3f})  "
          f"[{len(feature_cols)} features]")
    return aps.mean()


print("Baseline (all features):")
ap_all = cv_ap(all_features, "  all")

qgl_feats = [c for c in all_features if c.startswith("QGL")]
no_qgl = [c for c in all_features if not c.startswith("QGL")]
print(f"\nQGL features being removed: {qgl_feats}")
print("\nWithout any QGL features:")
ap_no_qgl = cv_ap(no_qgl, "  no QGL")

only_qgl = qgl_feats
print("\nUsing ONLY QGL features:")
ap_only_qgl = cv_ap(only_qgl, "  only QGL")

print("\n--- Verdict ---")
print(f"All features:   {ap_all:.3f}")
print(f"Without QGL:    {ap_no_qgl:.3f}  (drop of {ap_all - ap_no_qgl:.3f})")
print(f"Only QGL:       {ap_only_qgl:.3f}")
if ap_no_qgl >= ap_all - 0.10:
    print("\nThe model retains strong performance without QGL: signal is "
          "distributed, not resting on one feature.")
else:
    print("\nPerformance drops substantially without QGL: the detector was "
          "leaning heavily on this one feature. Investigate whether QGL drift "
          "reflects real hydrate physics or a dataset artifact.")