# Turn raw one-second sensor streams into windowed feature rows.
#
# A single instant tells a model nothing. What carries the fault signal
# is how each sensor behaves over a short window: its average level, its
# trend (slope), and how much it varies. This module slides a window over
# each well recording and summarises every sensor inside it, then labels
# the window as developing-fault or normal.

from pathlib import Path
import numpy as np
import pandas as pd

# Continuous process variables worth summarising.
SENSOR_COLUMNS = [
    "P-PDG", "P-TPT", "T-TPT", "P-MON-CKP", "T-JUS-CKP",
    "P-JUS-CKP", "QGL", "ABER-CKP",
]

WINDOW_SECONDS = 300   # 5-minute window
STEP_SECONDS = 60      # move forward 1 minute at a time


def _slope(values: np.ndarray) -> float:
    # Fit a straight line through the window and return its gradient.
    n = len(values)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    if np.all(np.isnan(values)):
        return 0.0
    mask = ~np.isnan(values)
    if mask.sum() < 2:
        return 0.0
    return float(np.polyfit(x[mask], values[mask], 1)[0])


def build_features_for_instance(df: pd.DataFrame, fault_label: int) -> pd.DataFrame:
    # Given one recording, return a table of windowed features.
    #
    # Positive (1): window lies inside the transient warning period.
    # Negative (0): window lies in the clearly-normal period before any
    #               warning starts.
    # Excluded:     windows inside the confirmed-fault stretch, and windows
    #               that mix states, are dropped as not part of the
    #               early-warning question.
    transient_label = 100 + fault_label

    present_sensors = [c for c in SENSOR_COLUMNS if c in df.columns]
    classes = df["class"].to_numpy()

    rows = []
    n = len(df)
    start = 0
    while start + WINDOW_SECONDS <= n:
        end = start + WINDOW_SECONDS
        window = df.iloc[start:end]
        window_classes = classes[start:end]

        in_transient = np.isin(window_classes, [transient_label])
        in_normal = np.isin(window_classes, [0])

        frac_transient = in_transient.mean()
        frac_normal = in_normal.mean()

        if frac_transient >= 0.9:
            label = 1            # clearly developing
        elif frac_normal >= 0.9:
            label = 0            # clearly normal
        else:
            start += STEP_SECONDS
            continue             # mixed or confirmed-fault: drop

        feat = {}
        for sensor in present_sensors:
            vals = window[sensor].to_numpy(dtype=float)
            feat[f"{sensor}_mean"] = np.nanmean(vals) if not np.all(np.isnan(vals)) else 0.0
            feat[f"{sensor}_std"] = np.nanstd(vals) if not np.all(np.isnan(vals)) else 0.0
            feat[f"{sensor}_slope"] = _slope(vals)

        feat["label"] = label
        feat["fault_type"] = fault_label
        rows.append(feat)

        start += STEP_SECONDS

    return pd.DataFrame(rows)


def build_dataset(data_root: Path, fault_labels, max_files_per_label=None):
    # Walk the real well recordings for each fault type and build one
    # combined feature table. Tracks which well each row came from so we
    # can later split by well and avoid data leakage.
    all_frames = []

    for label in fault_labels:
        folder = data_root / str(label)
        if not folder.exists():
            continue
        real_files = sorted(folder.glob("WELL-*.parquet"))
        if max_files_per_label is not None:
            real_files = real_files[:max_files_per_label]

        for f in real_files:
            cols = SENSOR_COLUMNS + ["class"]
            df = pd.read_parquet(f, columns=cols)
            feats = build_features_for_instance(df, label)
            if feats.empty:
                continue
            feats["well"] = f.name.split("_")[0]
            feats["source_file"] = f.name
            all_frames.append(feats)

    if not all_frames:
        return pd.DataFrame()
    return pd.concat(all_frames, ignore_index=True)