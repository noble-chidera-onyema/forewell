# Deployable spec for the combined hydrate detector.
#
# Converts per-window scores into the operational claim: of wells that
# developed a hydrate, how many did we alarm on during the developing
# phase, how many hours before confirmation did the first alarm fire, and
# how often did we alarm during normal operation (false-alarm rate). Uses
# the honest out-of-fold scores already saved, so every well was scored by
# a model that never trained on it. The threshold is swept to trace the
# operating curve.

import sys
from pathlib import Path

sys.path.insert(0, str(Path("../src").resolve()))

import numpy as np
import pandas as pd

from forewell.hydrate import STEP_SECONDS

DATA_ROOT = Path("../../3W/dataset")

df = pd.read_parquet(Path("../data/processed/hydrate_combined_scored.parquet"))
print(f"Loaded {len(df)} scored windows from {df['well'].nunique()} wells.")

# True event onset per well: first transient/confirmed second in its file.
# Fault folder tells us which transient label and which folder to read.
def onset_seconds(source_file, fault_folder):
    path = DATA_ROOT / str(fault_folder) / source_file
    raw = pd.read_parquet(path, columns=["class"])
    classes = raw["class"].to_numpy()
    transient = 100 + int(fault_folder)
    hit = np.where(np.isin(classes, [transient, int(fault_folder)]))[0]
    return int(hit[0]) if len(hit) else None

well_info = {}
for well, g in df.groupby("well"):
    has_event = g["label"].sum() > 0
    folder = int(g["fault_folder"].iloc[0])
    sf = g["source_file"].iloc[0]
    onset = onset_seconds(sf, folder) if has_event else None
    norm_frac = float((g["label"] == 0).mean())
    well_info[well] = {"has_event": has_event, "onset": onset,
                       "norm_frac": norm_frac, "folder": folder}

# Event wells with a usable pre-event normal stretch.
event_wells = [w for w, i in well_info.items()
               if i["has_event"] and i["onset"] is not None
               and i["norm_frac"] >= 0.15]
print(f"Event wells usable for lead-time: {len(event_wells)}")

windows_per_day = 86400 / STEP_SECONDS

def evaluate(thr):
    caught, lead_hours = 0, []
    fa_windows, normal_windows = 0, 0

    for well, g in df.groupby("well"):
        info = well_info[well]
        g = g.sort_values("win_end_idx")
        ends = g["win_end_idx"].to_numpy()
        alarms = g["score"].to_numpy() >= thr
        labels = g["label"].to_numpy()

        if well in event_wells:
            onset = info["onset"]
            pre = ends < onset
            fired = np.where(alarms & pre & (labels == 0) | (alarms & pre & (labels == 1)))[0]
            # first alarm anywhere before confirmed onset counts as warning
            fired_pre = np.where(alarms & pre)[0]
            if len(fired_pre):
                caught += 1
                lead_hours.append((onset - ends[fired_pre[0]]) / 3600.0)
            # false alarms = alarms on normal windows of this event well
            normal_mask = labels == 0
            fa_windows += int(alarms[normal_mask].sum())
            normal_windows += int(normal_mask.sum())
        else:
            normal_mask = labels == 0
            fa_windows += int(alarms[normal_mask].sum())
            normal_windows += int(normal_mask.sum())

    coverage = caught / len(event_wells) if event_wells else 0.0
    med_lead = float(np.median(lead_hours)) if lead_hours else 0.0
    min_lead = float(np.min(lead_hours)) if lead_hours else 0.0
    fa_rate = (fa_windows / normal_windows * windows_per_day) if normal_windows else 0.0
    return coverage, med_lead, min_lead, fa_rate, caught

rows = []
for thr in np.linspace(0.3, 0.95, 14):
    cov, med, mn, fa, caught = evaluate(thr)
    rows.append({"threshold": round(thr, 3), "events_caught": caught,
                 "coverage": round(cov, 3), "median_lead_h": round(med, 2),
                 "min_lead_h": round(mn, 2),
                 "false_alarms_per_well_day": round(fa, 2)})

table = pd.DataFrame(rows)
print("\n--- Deployable operating curve ---")
print(table.to_string(index=False))
table.to_csv(Path("../reports/hydrate_combined_spec.csv"), index=False)
print("\nSaved to reports/hydrate_combined_spec.csv")