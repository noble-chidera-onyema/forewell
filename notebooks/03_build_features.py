# Run the feature builder over the real fault instances, balance the
# fault types so none dominates, and save the table to data/processed.

import sys
from pathlib import Path

sys.path.insert(0, str(Path("../src").resolve()))

import pandas as pd
from forewell.features import build_dataset

DATA_ROOT = Path("../../3W/dataset")
fault_labels = [1, 2, 5, 6, 7, 8, 9]

print("Building features. This processes many recordings, please wait...")
dataset = build_dataset(DATA_ROOT, fault_labels)

if dataset.empty:
    print("No features were built. Check the data path.")
    sys.exit()

print(f"Raw build: {len(dataset)} windows from {dataset['well'].nunique()} wells.")
print("\nRaw windows per fault type (count = total, sum = warning windows):")
print(dataset.groupby("fault_type")["label"].agg(["count", "sum"]))

# Balance: cap the number of POSITIVE (warning) windows contributed by any
# single fault type, so slow faults like 7 and 8 cannot drown out the rest.
CAP_PER_FAULT = 2000

balanced_frames = []
for fault, group in dataset.groupby("fault_type"):
    positives = group[group["label"] == 1]
    negatives = group[group["label"] == 0]
    if len(positives) > CAP_PER_FAULT:
        positives = positives.sample(CAP_PER_FAULT, random_state=42)
    # Keep negatives at roughly the same scale as positives per fault.
    neg_cap = min(len(negatives), max(CAP_PER_FAULT, len(positives)))
    if len(negatives) > neg_cap:
        negatives = negatives.sample(neg_cap, random_state=42)
    balanced_frames.append(pd.concat([positives, negatives]))

balanced = pd.concat(balanced_frames, ignore_index=True)

print(f"\nBalanced dataset: {len(balanced)} windows.")
print("\nLabel balance (0 = normal, 1 = developing fault):")
print(balanced["label"].value_counts())
print("\nBalanced windows per fault type:")
print(balanced.groupby("fault_type")["label"].agg(["count", "sum"]))

out_path = Path("../data/processed/features.parquet")
balanced.to_parquet(out_path, index=False)
print(f"\nSaved balanced feature table to {out_path.resolve()}")