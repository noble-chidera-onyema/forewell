# Build the hydrate forecasting dataset and report its shape.
#
# Every positive here is a genuine ahead-of-time example: sensor data from
# before a 30-minute gap, labelled by whether a hydrate begins within the
# following 6 hours. This is the dataset a real early-warning system would
# train on.

import sys
from pathlib import Path

sys.path.insert(0, str(Path("../src").resolve()))

import pandas as pd
from forewell.hydrate import build_hydrate_dataset, HORIZON_HOURS, GAP_SECONDS

DATA_ROOT = Path("../../3W/dataset")

print(f"Building hydrate forecast dataset "
      f"(horizon {HORIZON_HOURS}h, gap {GAP_SECONDS//60}min)...")
df = build_hydrate_dataset(DATA_ROOT)

if df.empty:
    print("No data built. Check the path.")
    sys.exit()

print(f"\nTotal windows: {len(df)} from {df['well'].nunique()} wells.")
print("\nLabel balance (1 = hydrate within horizon, 0 = not):")
print(df["label"].value_counts())
print(f"\nPositive rate: {df['label'].mean():.3f}")

print("\nWindows and positives per well:")
print(df.groupby("well")["label"].agg(["count", "sum"]).to_string())

out = Path("../data/processed/hydrate_forecast.parquet")
df.to_parquet(out, index=False)
print(f"\nSaved to {out.resolve()}")