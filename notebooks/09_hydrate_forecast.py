# Build the hydrate forecasting dataset: developing-hydrate windows from
# fault-8 wells, PLUS abundant normal-operation windows from folder 0 as
# negatives. The normal data teaches the model what calm looks like so it
# stays quiet during normal operation instead of alarming constantly.

import sys
from pathlib import Path

sys.path.insert(0, str(Path("../src").resolve()))

import pandas as pd
from forewell.hydrate import (
    build_hydrate_dataset, build_normal_negatives,
    HORIZON_HOURS, GAP_SECONDS,
)

DATA_ROOT = Path("../../3W/dataset")

print(f"Building hydrate windows (horizon {HORIZON_HOURS}h)...")
hyd = build_hydrate_dataset(DATA_ROOT)
print(f"  hydrate-well windows: {len(hyd)} "
      f"(positives {int(hyd['label'].sum())})")

print("Building normal-operation negatives from folder 0...")
norm = build_normal_negatives(DATA_ROOT, max_files=60)
print(f"  normal-well windows: {len(norm)} (all negative)")

df = pd.concat([hyd, norm], ignore_index=True)

print(f"\nCombined: {len(df)} windows from {df['well'].nunique()} wells.")
print("\nLabel balance (1 = hydrate within horizon, 0 = not):")
print(df["label"].value_counts())
print(f"Positive rate: {df['label'].mean():.3f}")

print("\nPositives per well (only hydrate wells have any):")
print(df.groupby("well")["label"].agg(["count", "sum"]).to_string())

out = Path("../data/processed/hydrate_forecast.parquet")
df.to_parquet(out, index=False)
print(f"\nSaved to {out.resolve()}")