"""
First look at the Petrobras 3W dataset.
Loads one normal instance and one faulty instance, prints their
structure, and saves a plot of the sensor traces.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# The 3W dataset sits next to the forewell project folder.
DATA_ROOT = Path("../../3W/dataset")

# Pick one real normal well and one real faulty well (label 5).
normal_file = DATA_ROOT / "0" / "WELL-00001_20170201010207.parquet"
fault_file = DATA_ROOT / "5" / "WELL-00015_20170620122925.parquet"

print("Loading normal instance...")
df_normal = pd.read_parquet(normal_file)
print("Loading fault instance (rapid productivity loss)...")
df_fault = pd.read_parquet(fault_file)

# Look at the structure of the normal instance.
print("\n--- NORMAL INSTANCE ---")
print("Shape (rows, columns):", df_normal.shape)
print("\nColumn names:")
for col in df_normal.columns:
    print("  ", col)
print("\nFirst 3 rows:")
print(df_normal.head(3))
print("\nThe 'class' column tells us the label at each timestamp.")
print("Normal instance class values seen:", df_normal["class"].dropna().unique())
print("Fault instance class values seen:", df_fault["class"].dropna().unique())

# Save a quick plot of a few key sensors from the fault instance.
sensors_to_plot = ["P-PDG", "P-TPT", "T-TPT", "P-MON-CKP"]
available = [s for s in sensors_to_plot if s in df_fault.columns]

fig, axes = plt.subplots(len(available) + 1, 1, figsize=(12, 9), sharex=True)
for ax, sensor in zip(axes, available):
    ax.plot(df_fault.index, df_fault[sensor], linewidth=0.8)
    ax.set_ylabel(sensor)
    ax.grid(True, alpha=0.3)

# Bottom panel: the class label over time, so we see when the fault starts.
axes[-1].plot(df_fault.index, df_fault["class"], color="red", linewidth=1.2)
axes[-1].set_ylabel("class")
axes[-1].set_xlabel("time")
axes[-1].grid(True, alpha=0.3)

fig.suptitle("Fault instance: sensors and label over time (label 5)")
fig.tight_layout()

out_path = Path("../reports/figures/01_first_look.jpg")
fig.savefig(out_path, dpi=120, bbox_inches="tight")
print(f"\nSaved figure to {out_path.resolve()}")
print("Done.")