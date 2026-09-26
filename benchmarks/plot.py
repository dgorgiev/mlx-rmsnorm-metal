import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def read_rows(path):
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def save_line_plot(rows, group_field, title, output, y_field="median_us"):
    groups = defaultdict(list)
    for row in rows:
        groups[row[group_field]].append(
            (int(row["hidden_dimension"]), float(row[y_field]))
        )

    if not groups:
        return

    fig, axis = plt.subplots(figsize=(8, 5))
    for label, values in sorted(groups.items()):
        values.sort()
        axis.plot(
            [value[0] for value in values],
            [value[1] for value in values],
            marker="o",
            label=label,
        )
    axis.set_title(title)
    axis.set_xlabel("Hidden dimension")
    axis.set_ylabel(
        "Latency (microseconds)"
        if y_field == "median_us"
        else "Mean absolute error"
    )
    axis.set_xscale("log", base=2)
    if y_field != "median_us":
        axis.set_yscale("log")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_implementations(rows, output_dir):
    for dtype in ("float16", "float32"):
        selected = [
            row
            for row in rows
            if row["suite"] == "implementations"
            and row["dtype"] == dtype
            and row["batch"] == "1024"
        ]
        save_line_plot(
            selected,
            "implementation",
            f"RMSNorm implementations: {dtype}, batch 1024",
            output_dir / f"implementations_{dtype}.png",
        )


def plot_threadgroups(rows, output_dir):
    for implementation in ("parallel", "optimized"):
        selected = [
            row
            for row in rows
            if row["suite"] == "threadgroups"
            and row["implementation"] == implementation
        ]
        save_line_plot(
            selected,
            "threadgroup_size",
            f"{implementation.title()} kernel by threadgroup size",
            output_dir / f"threadgroups_{implementation}.png",
        )


def plot_experiments(rows, output_dir):
    vectorization = [row for row in rows if row["suite"] == "vectorization"]
    save_line_plot(
        vectorization,
        "vector_width",
        "Vectorized memory access",
        output_dir / "vectorization.png",
    )

    precision = [row for row in rows if row["suite"] == "precision"]
    save_line_plot(
        precision,
        "accumulation",
        "Accumulation precision: latency",
        output_dir / "precision_latency.png",
    )
    save_line_plot(
        precision,
        "accumulation",
        "Accumulation precision: mean error",
        output_dir / "precision_error.png",
        y_field="mean_abs_error",
    )


def main():
    parser = argparse.ArgumentParser(description="Plot RMSNorm benchmark CSV")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/plots"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.csv_path)
    plot_implementations(rows, args.output_dir)
    plot_threadgroups(rows, args.output_dir)
    plot_experiments(rows, args.output_dir)


if __name__ == "__main__":
    main()
