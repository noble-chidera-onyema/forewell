# Measure the early-warning lead time in real fault instances.
# For each real well recording that contains a fault, find when the
# transient warning label (100 + fault number) first appears and when
# the confirmed fault label first appears, then record the gap.

from pathlib import Path
import pandas as pd

DATA_ROOT = Path("../../3W/dataset")

# Fault folders are 1 through 9. Folder 0 is normal.
fault_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9]

records = []

for label in fault_labels:
    folder = DATA_ROOT / str(label)
    if not folder.exists():
        continue

    # Only real wells, not simulated or hand-drawn instances.
    real_files = sorted(folder.glob("WELL-*.parquet"))

    transient_label = 100 + label

    for f in real_files:
        # Read only the class column to keep this fast.
        df = pd.read_parquet(f, columns=["class"])
        classes = df["class"]

        # Position (row number) where the warning first appears.
        warning_rows = classes.index[classes == transient_label]
        fault_rows = classes.index[classes == label]

        if len(warning_rows) == 0 or len(fault_rows) == 0:
            # No transient window or no confirmed fault in this file.
            continue

        warning_start = warning_rows[0]
        fault_start = fault_rows[0]

        # Each row is one second. Gap in seconds, then minutes.
        gap_seconds = (fault_start - warning_start).total_seconds()
        gap_minutes = gap_seconds / 60.0

        records.append({
            "fault_label": label,
            "file": f.name,
            "warning_start": warning_start,
            "fault_start": fault_start,
            "lead_time_minutes": round(gap_minutes, 1),
        })

results = pd.DataFrame(records)

if results.empty:
    print("No instances with both a transient window and a confirmed fault were found.")
else:
    print(f"Found {len(results)} real fault instances with a warning window.\n")
    print("Lead time in minutes, summary across all faults:")
    print(results["lead_time_minutes"].describe().round(1))
    print("\nMedian lead time by fault type:")
    print(results.groupby("fault_label")["lead_time_minutes"].median().round(1))

    out_csv = Path("../reports/lead_time_by_instance.csv")
    results.to_csv(out_csv, index=False)
    print(f"\nSaved per-instance results to {out_csv.resolve()}")

    # Save a figure showing the spread of lead times.
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(
        [results.loc[results["fault_label"] == lbl, "lead_time_minutes"]
         for lbl in sorted(results["fault_label"].unique())],
        tick_labels=[str(lbl) for lbl in sorted(results["fault_label"].unique())],
        orientation="vertical",
        showfliers=True,
    )
    ax.set_yscale("log")
    ax.set_xlabel("fault type")
    ax.set_ylabel("lead time (minutes, log scale)")
    ax.set_title("Early-warning lead time before confirmed fault, by fault type")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    out_fig = Path("../reports/figures/02_lead_time.jpg")
    fig.savefig(out_fig, dpi=120, bbox_inches="tight")
    print(f"Saved figure to {out_fig.resolve()}")