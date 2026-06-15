# Flagship early-warning pipeline for hydrate formation (fault 8).
#
# This is a forecasting task, not detection. For each window ending at
# time T, the label answers: will a hydrate be confirmed within the next
# HORIZON hours after T? Features are drawn only from data at or before T,
# with a deliberate gap before the event, so the model cannot peek at what
# it is forecasting. This is what makes the warning genuine.

from pathlib import Path
import numpy as np
import pandas as pd

FAULT = 8
TRANSIENT = 108
CONFIRMED = 8

SENSOR_COLUMNS = [
    "P-PDG", "P-TPT", "T-TPT", "P-MON-CKP", "T-JUS-CKP",
    "P-JUS-CKP", "QGL", "ABER-CKP",
]

WINDOW_SECONDS = 1800      # 30-minute feature window
STEP_SECONDS = 300         # advance 5 minutes between windows
HORIZON_HOURS = 8          # forecast: hydrate within next 8 hours?
GAP_SECONDS = 1800         # 30-minute gap between window end and horizon start


def _slope(values: np.ndarray) -> float:
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


def _safe_mean(v):
    return float(np.nanmean(v)) if not np.all(np.isnan(v)) else 0.0


def _safe_std(v):
    return float(np.nanstd(v)) if not np.all(np.isnan(v)) else 0.0


def _trend_over(series_vals: np.ndarray, seconds: int, step: int) -> float:
    # Net change of a sensor over the last `seconds`, as a per-hour rate.
    # Positive means rising, negative falling, over that lookback.
    n_points = max(2, seconds // step)
    if len(series_vals) < 2:
        return 0.0
    recent = series_vals[-n_points:]
    mask = ~np.isnan(recent)
    if mask.sum() < 2:
        return 0.0
    idx = np.arange(len(recent))[mask]
    vals = recent[mask]
    slope_per_step = float(np.polyfit(idx, vals, 1)[0])
    return slope_per_step * (3600.0 / step)  # convert to per-hour


def build_hydrate_instance(df: pd.DataFrame) -> pd.DataFrame:
    # Each row ends at time T. Features describe both the recent 30-minute
    # window and the multi-hour trajectory leading up to T, because hydrate
    # formation is a slow accumulation that shows in sustained trends, not
    # in any single snapshot. Label: does a hydrate begin within HORIZON
    # hours after a gap following T?
    present = [c for c in SENSOR_COLUMNS if c in df.columns]
    classes = df["class"].to_numpy()
    n = len(df)
    horizon_secs = HORIZON_HOURS * 3600

    # Per-second arrays for trend lookbacks, with the well's own early
    # baseline (first 30 minutes) as a drift reference.
    sensor_arrays = {s: df[s].to_numpy(dtype=float) for s in present}
    baseline = {}
    for s in present:
        head = sensor_arrays[s][:1800]
        baseline[s] = float(np.nanmean(head)) if not np.all(np.isnan(head)) else 0.0

    LOOKBACKS = [3600, 7200, 14400]  # 1h, 2h, 4h trends
    STEP = 1  # arrays are per-second

    rows = []
    start = 0
    while start + WINDOW_SECONDS <= n:
        win_end = start + WINDOW_SECONDS
        window = df.iloc[start:win_end]

        look_start = win_end + GAP_SECONDS
        if look_start >= n:
            break
        look_end = min(look_start + horizon_secs, n)
        future_classes = classes[look_start:look_end]

        win_classes = classes[start:win_end]
        if np.isin(win_classes, [TRANSIENT, CONFIRMED]).any():
            start += STEP_SECONDS
            continue

        # Require enough history for the longest trend lookback.
        if win_end < max(LOOKBACKS):
            start += STEP_SECONDS
            continue

        label = int(bool(np.isin(future_classes, [TRANSIENT, CONFIRMED]).any()))

        feat = {}
        for s in present:
            vals = window[s].to_numpy(dtype=float)
            feat[f"{s}_mean"] = _safe_mean(vals)
            feat[f"{s}_std"] = _safe_std(vals)

            # Drift from the well's own early baseline.
            feat[f"{s}_drift_base"] = _safe_mean(vals) - baseline[s]

            # Multi-hour trends ending at win_end.
            history = sensor_arrays[s][:win_end]
            for lb in LOOKBACKS:
                feat[f"{s}_trend_{lb//3600}h"] = _trend_over(history, lb, STEP)

        # Physics features for hydrates: how cold, and choke pressure drop.
        if "T-TPT" in present:
            t = window["T-TPT"].to_numpy(dtype=float)
            feat["T-TPT_min"] = float(np.nanmin(t)) if not np.all(np.isnan(t)) else 0.0
        if "P-MON-CKP" in present and "P-JUS-CKP" in present:
            up = _safe_mean(window["P-MON-CKP"].to_numpy(dtype=float))
            down = _safe_mean(window["P-JUS-CKP"].to_numpy(dtype=float))
            feat["choke_dp"] = up - down

        feat["label"] = label
        # Keep the window-end position so we can evaluate per-event later.
        feat["win_end_idx"] = win_end
        rows.append(feat)
        start += STEP_SECONDS

    return pd.DataFrame(rows)


def build_hydrate_dataset(data_root: Path) -> pd.DataFrame:
    folder = data_root / str(FAULT)
    files = sorted(folder.glob("WELL-*.parquet"))
    frames = []
    for f in files:
        cols = SENSOR_COLUMNS + ["class"]
        df = pd.read_parquet(f, columns=cols)
        feats = build_hydrate_instance(df)
        if feats.empty:
            continue
        feats["well"] = f.name.split("_")[0]
        feats["source_file"] = f.name
        frames.append(feats)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)