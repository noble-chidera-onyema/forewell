# Turn raw one-second sensor streams into windowed feature rows.
#
# A single instant tells a model nothing. What carries the fault signal
# is how each sensor behaves over a window: its level, its trend, how
# much it varies, and how fast it is changing. This builder uses two
# window lengths so that both abrupt and slow-developing faults are
# visible, and adds a few cross-sensor features grounded in the physics
# of the production system.

from pathlib import Path
import numpy as np
import pandas as pd

# Continuous process variables worth summarising.
SENSOR_COLUMNS = [
    "P-PDG", "P-TPT", "T-TPT", "P-MON-CKP", "T-JUS-CKP",
    "P-JUS-CKP", "QGL", "ABER-CKP",
]

SHORT_WINDOW = 300     # 5 minutes: catches abrupt change
LONG_WINDOW = 1800     # 30 minutes: catches slow drift
STEP_SECONDS = 60      # move forward 1 minute at a time


def _slope(values: np.ndarray) -> float:
    # Gradient of a straight line through the window. Positive means rising.
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


def _safe_mean(vals: np.ndarray) -> float:
    return float(np.nanmean(vals)) if not np.all(np.isnan(vals)) else 0.0


def _safe_std(vals: np.ndarray) -> float:
    return float(np.nanstd(vals)) if not np.all(np.isnan(vals)) else 0.0


def build_features_for_instance(df: pd.DataFrame, fault_label: int) -> pd.DataFrame:
    # Positive (1): window inside the transient warning period.
    # Negative (0): window in the clearly-normal period before any warning.
    # Excluded: confirmed-fault stretches and mixed windows are dropped.
    transient_label = 100 + fault_label

    present_sensors = [c for c in SENSOR_COLUMNS if c in df.columns]
    classes = df["class"].to_numpy()

    rows = []
    n = len(df)
    start = 0
    while start + LONG_WINDOW <= n:
        long_end = start + LONG_WINDOW
        short_start = long_end - SHORT_WINDOW

        long_win = df.iloc[start:long_end]
        short_win = df.iloc[short_start:long_end]
        win_classes = classes[short_start:long_end]

        in_transient = np.isin(win_classes, [transient_label])
        in_normal = np.isin(win_classes, [0])
        frac_transient = in_transient.mean()
        frac_normal = in_normal.mean()

        if frac_transient >= 0.9:
            label = 1
        elif frac_normal >= 0.9:
            label = 0
        else:
            start += STEP_SECONDS
            continue

        feat = {}
        for sensor in present_sensors:
            short_vals = short_win[sensor].to_numpy(dtype=float)
            long_vals = long_win[sensor].to_numpy(dtype=float)

            s_mean = _safe_mean(short_vals)
            l_mean = _safe_mean(long_vals)

            # Short-window level, spread, and trend.
            feat[f"{sensor}_mean"] = s_mean
            feat[f"{sensor}_std"] = _safe_std(short_vals)
            feat[f"{sensor}_slope"] = _slope(short_vals)

            # Long-window trend and spread: slow drift shows up here.
            feat[f"{sensor}_slope_long"] = _slope(long_vals)
            feat[f"{sensor}_std_long"] = _safe_std(long_vals)

            # Drift: how far the recent short-window mean has moved from
            # the longer-term mean. A creeping fault grows this gap.
            feat[f"{sensor}_drift"] = s_mean - l_mean

        # Cross-sensor physics: pressure drop across the production choke.
        # A developing restriction changes this difference even when each
        # individual pressure still looks plausible.
        if "P-MON-CKP" in present_sensors and "P-JUS-CKP" in present_sensors:
            up = _safe_mean(short_win["P-MON-CKP"].to_numpy(dtype=float))
            down = _safe_mean(short_win["P-JUS-CKP"].to_numpy(dtype=float))
            feat["choke_dp"] = up - down

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