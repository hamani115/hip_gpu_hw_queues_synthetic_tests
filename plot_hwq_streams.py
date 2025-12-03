#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(
        description="Plot time vs num_streams for different GPU_MAX_HW_QUEUES values."
    )
    p.add_argument(
        "--csv",
        default="results_hwq_streams.csv",
        help="Path to input CSV (default: results_hwq_streams.csv).",
    )
    p.add_argument(
        "--out",
        default="streams_vs_hwq.png",
        help="Output image filename (PNG, default: streams_vs_hwq.png).",
    )
    p.add_argument(
        "--legend",
        default="N/A",
        help="Name of the GPU used to run this test (e.g. AMD W7900)",
    )
    args = p.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV file not found: {csv_path}")

    # Read CSV
    df = pd.read_csv(csv_path)

    # Ensure correct dtypes
    df["gpu_max_hw_queues"] = df["gpu_max_hw_queues"].astype(int)
    df["num_streams"] = df["num_streams"].astype(int)
    df["time_ms"] = df["time_ms"].astype(float)

    # Group by (hwq, streams) and compute mean/std over run_id
    grouped = (
        df.groupby(["gpu_max_hw_queues", "num_streams"])["time_ms"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_time_ms", "std": "std_time_ms"})
    )

    print("Aggregated data:")
    print(grouped)

    # Plot
    plt.figure(figsize=(8, 5))

    for hwq, sub in grouped.groupby("gpu_max_hw_queues"):
        sub = sub.sort_values("num_streams")
        x = sub["num_streams"]
        y = sub["mean_time_ms"]
        yerr = sub["std_time_ms"]

        # Errorbar line per GPU_MAX_HW_QUEUES
        plt.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            linestyle="-",
            capsize=3,
            label=f"GPU_MAX_HW_QUEUES={hwq}",
        )

    plt.xlabel("number of streams")
    plt.ylabel("execution time (ms)")
    plt.title("time vs streams for different GPU_MAX_HW_QUEUES (1000 kernel launches)")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(title=f"{args.legend}")
    plt.tight_layout()

    plt.savefig(args.out, dpi=200)
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()

