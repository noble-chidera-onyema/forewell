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
HORIZON_HOURS = 2          # forecast: hydrate within next 2 hours?
GAP_SECONDS = 900          # 15-minute gap between window end and horizon start


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

    LOOKBACKS = [1800, 3600, 7200]  # 30min, 1h, 2h trends
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

def build_normal_negatives(data_root: Path, max_files=12):
    # Build negative (no-hydrate-coming) windows from normal-operation
    # recordings in folder 0. These teach the model what calm looks like,
    # so it stays quiet during normal operation instead of alarming
    # constantly. Same feature layout as the hydrate windows, label 0.
    folder = data_root / "0"
    all_files = sorted(folder.glob("WELL-*.parquet"))
    # Group by well and take a few files from each, for variety across wells.
    by_well = {}
    for f in all_files:
        by_well.setdefault(f.name.split("_")[0], []).append(f)
    files = []
    files_per_well = max(1, max_files // max(1, len(by_well)))
    for well_files in by_well.values():
        files.extend(well_files[:files_per_well])
    frames = []

    for f in files:
        cols = SENSOR_COLUMNS + ["class"]
        df = pd.read_parquet(f, columns=cols)
        classes = df["class"].to_numpy()
        n = len(df)
        if n < max([1800, 3600, 7200]) + WINDOW_SECONDS:
            continue

        present = [c for c in SENSOR_COLUMNS if c in df.columns]
        sensor_arrays = {s: df[s].to_numpy(dtype=float) for s in present}
        baseline = {}
        for s in present:
            head = sensor_arrays[s][:1800]
            baseline[s] = float(np.nanmean(head)) if not np.all(np.isnan(head)) else 0.0

        LOOKBACKS = [3600, 7200, 14400]
        rows = []
        start = 0
        while start + WINDOW_SECONDS <= n:
            win_end = start + WINDOW_SECONDS
            if win_end < max(LOOKBACKS):
                start += STEP_SECONDS
                continue
            window = df.iloc[start:win_end]

            feat = {}
            for s in present:
                vals = window[s].to_numpy(dtype=float)
                feat[f"{s}_mean"] = _safe_mean(vals)
                feat[f"{s}_std"] = _safe_std(vals)
                feat[f"{s}_drift_base"] = _safe_mean(vals) - baseline[s]
                history = sensor_arrays[s][:win_end]
                for lb in LOOKBACKS:
                    feat[f"{s}_trend_{lb//3600}h"] = _trend_over(history, lb, 1)

            if "T-TPT" in present:
                t = window["T-TPT"].to_numpy(dtype=float)
                feat["T-TPT_min"] = float(np.nanmin(t)) if not np.all(np.isnan(t)) else 0.0
            if "P-MON-CKP" in present and "P-JUS-CKP" in present:
                up = _safe_mean(window["P-MON-CKP"].to_numpy(dtype=float))
                down = _safe_mean(window["P-JUS-CKP"].to_numpy(dtype=float))
                feat["choke_dp"] = up - down

            feat["label"] = 0
            feat["win_end_idx"] = win_end
            rows.append(feat)
            start += STEP_SECONDS * 2   # subsample normal data, plenty of it

        if rows:
            fr = pd.DataFrame(rows)
            fr["well"] = f.name.split("_")[0]
            fr["source_file"] = f.name
            frames.append(fr)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

def build_simulated_dataset(data_root: Path) -> pd.DataFrame:
    # Harvest simulated hydrate instances (SIMULATED_*.parquet) into the
    # same windowed feature format as the real wells. These are the
    # physics-simulated events the dataset authors provide for exactly the
    # case where real events are too few to train on. Used for training;
    # real wells are reserved for testing.
    folder = data_root / str(FAULT)
    files = sorted(folder.glob("SIMULATED_*.parquet"))
    frames = []
    for f in files:
        cols = SENSOR_COLUMNS + ["class"]
        try:
            df = pd.read_parquet(f, columns=cols)
        except Exception:
            # Some simulated files may lack a column; load what exists.
            raw = pd.read_parquet(f)
            keep = [c for c in cols if c in raw.columns]
            df = raw[keep]
        feats = build_hydrate_instance(df)
        if feats.empty:
            continue
        feats["well"] = "SIM-" + f.stem.split("_")[-1]
        feats["source_file"] = f.name
        frames.append(feats)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

def build_detection_instance(df: pd.DataFrame) -> pd.DataFrame:
    # Detection framing: positive = developing-transient hydrate (class
    # 108), negative = normal operation (class 0). The confirmed-fault
    # stretch (class 8) is excluded, since by then the fault is already
    # known. Features describe the recent window plus short trend context.
    present = [c for c in SENSOR_COLUMNS if c in df.columns]
    classes = df["class"].to_numpy()
    n = len(df)

    sensor_arrays = {s: df[s].to_numpy(dtype=float) for s in present}
    baseline = {}
    for s in present:
        head = sensor_arrays[s][:600]
        baseline[s] = float(np.nanmean(head)) if not np.all(np.isnan(head)) else 0.0

    LOOKBACKS = [600, 1800]  # 10min, 30min trends; short so all files work
    rows = []
    start = 0
    while start + WINDOW_SECONDS <= n:
        win_end = start + WINDOW_SECONDS
        window_classes = classes[start:win_end]

        frac_transient = np.isin(window_classes, [TRANSIENT]).mean()
        frac_normal = np.isin(window_classes, [0]).mean()

        if frac_transient >= 0.9:
            label = 1
        elif frac_normal >= 0.9:
            label = 0
        else:
            start += STEP_SECONDS
            continue

        feat = {}
        for s in present:
            vals = window[s] if False else df.iloc[start:win_end][s].to_numpy(dtype=float)
            feat[f"{s}_mean"] = _safe_mean(vals)
            feat[f"{s}_std"] = _safe_std(vals)
            feat[f"{s}_drift_base"] = _safe_mean(vals) - baseline[s]
            history = sensor_arrays[s][:win_end]
            for lb in LOOKBACKS:
                if win_end >= lb:
                    feat[f"{s}_trend_{lb//60}m"] = _trend_over(history, lb, 1)
                else:
                    feat[f"{s}_trend_{lb//60}m"] = 0.0

        if "T-TPT" in present:
            t = df.iloc[start:win_end]["T-TPT"].to_numpy(dtype=float)
            feat["T-TPT_min"] = float(np.nanmin(t)) if not np.all(np.isnan(t)) else 0.0
        if "P-MON-CKP" in present and "P-JUS-CKP" in present:
            up = _safe_mean(df.iloc[start:win_end]["P-MON-CKP"].to_numpy(dtype=float))
            down = _safe_mean(df.iloc[start:win_end]["P-JUS-CKP"].to_numpy(dtype=float))
            feat["choke_dp"] = up - down

        feat["label"] = label
        feat["win_end_idx"] = win_end
        rows.append(feat)
        start += STEP_SECONDS

    return pd.DataFrame(rows)


def build_detection_dataset(data_root: Path, source="real") -> pd.DataFrame:
    # source="real" reads WELL-* files; source="sim" reads SIMULATED-*.
    folder = data_root / str(FAULT)
    pattern = "WELL-*.parquet" if source == "real" else "SIMULATED_*.parquet"
    files = sorted(folder.glob(pattern))
    frames = []
    for f in files:
        cols = SENSOR_COLUMNS + ["class"]
        try:
            df = pd.read_parquet(f, columns=cols)
        except Exception:
            raw = pd.read_parquet(f)
            df = raw[[c for c in cols if c in raw.columns]]
        feats = build_detection_instance(df)
        if feats.empty:
            continue
        if source == "real":
            feats["well"] = f.name.split("_")[0]
        else:
            feats["well"] = "SIM-" + f.stem.split("_")[-1]
        feats["source_file"] = f.name
        frames.append(feats)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

def build_detection_dataset_combined(data_root: Path, source="real",
                                     fault_folders=(8, 9)) -> pd.DataFrame:
    # Hydrate formation in either the production line (fault 8) or the
    # service line (fault 9) is the same phenomenon in a different location.
    # This harvests both into one detection dataset. Positive = developing
    # transient (class 108 or 109), negative = normal (class 0).
    pattern = "WELL-*.parquet" if source == "real" else "SIMULATED_*.parquet"
    frames = []

    for fault in fault_folders:
        transient = 100 + fault
        folder = data_root / str(fault)
        files = sorted(folder.glob(pattern))
        for f in files:
            cols = SENSOR_COLUMNS + ["class"]
            try:
                df = pd.read_parquet(f, columns=cols)
            except Exception:
                raw = pd.read_parquet(f)
                df = raw[[c for c in cols if c in raw.columns]]

            feats = _build_detection_for_transient(df, transient)
            if feats.empty:
                continue
            if source == "real":
                feats["well"] = f.name.split("_")[0]
            else:
                feats["well"] = f"SIM{fault}-" + f.stem.split("_")[-1]
            feats["source_file"] = f.name
            feats["fault_folder"] = fault
            frames.append(feats)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _build_detection_for_transient(df: pd.DataFrame, transient_label: int) -> pd.DataFrame:
    # Same as build_detection_instance but for a given transient label,
    # so it works for both fault 8 (108) and fault 9 (109).
    present = [c for c in SENSOR_COLUMNS if c in df.columns]
    classes = df["class"].to_numpy()
    n = len(df)

    sensor_arrays = {s: df[s].to_numpy(dtype=float) for s in present}
    baseline = {}
    for s in present:
        head = sensor_arrays[s][:600]
        baseline[s] = float(np.nanmean(head)) if not np.all(np.isnan(head)) else 0.0

    LOOKBACKS = [600, 1800]
    rows = []
    start = 0
    while start + WINDOW_SECONDS <= n:
        win_end = start + WINDOW_SECONDS
        window_classes = classes[start:win_end]

        frac_transient = np.isin(window_classes, [transient_label]).mean()
        frac_normal = np.isin(window_classes, [0]).mean()

        if frac_transient >= 0.9:
            label = 1
        elif frac_normal >= 0.9:
            label = 0
        else:
            start += STEP_SECONDS
            continue

        seg = df.iloc[start:win_end]
        feat = {}
        for s in present:
            vals = seg[s].to_numpy(dtype=float)
            feat[f"{s}_mean"] = _safe_mean(vals)
            feat[f"{s}_std"] = _safe_std(vals)
            feat[f"{s}_drift_base"] = _safe_mean(vals) - baseline[s]
            history = sensor_arrays[s][:win_end]
            for lb in LOOKBACKS:
                feat[f"{s}_trend_{lb//60}m"] = _trend_over(history, lb, 1) if win_end >= lb else 0.0

        if "T-TPT" in present:
            t = seg["T-TPT"].to_numpy(dtype=float)
            feat["T-TPT_min"] = float(np.nanmin(t)) if not np.all(np.isnan(t)) else 0.0
        if "P-MON-CKP" in present and "P-JUS-CKP" in present:
            up = _safe_mean(seg["P-MON-CKP"].to_numpy(dtype=float))
            down = _safe_mean(seg["P-JUS-CKP"].to_numpy(dtype=float))
            feat["choke_dp"] = up - down

        feat["label"] = label
        feat["win_end_idx"] = win_end
        rows.append(feat)
        start += STEP_SECONDS

    return pd.DataFrame(rows)