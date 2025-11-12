#!/usr/bin/env python3
"""
High-sample benchmark analysis and visualization.

This script targets the refreshed parameter sweep captured in
`highsample_benchmark2.csv`. The sweep densifies the sample-count axis (N) for a
fixed resolution, so we focus on:
    1. quantifying current runtime distributions and GPU speedups, and
    2. highlighting which (resolution, sample) cells are still incomplete.

It emits a short textual report plus whichever plots make sense for the loaded
dataset (runtime scaling, GPU speedups, parameter sensitivity, coverage).

Usage:
    python3 tools/highsample_analysis.py \
        --csv highsample_benchmark2.csv \
        --outdir benchmark_outputs/highsample_analysis
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class BenchmarkRow:
    algorithm: str
    device: str
    thread_mode: str
    m: int
    n: int
    N: int
    mu_r: float
    sigma_r_ratio: float
    s: int
    intensity_pattern: str
    runtime_seconds: float


@dataclass(frozen=True)
class SpeedupRecord:
    algorithm: str
    m: int
    N: int
    mu_r: float
    intensity_pattern: str
    cpu_median: float
    gpu_median: float
    speedup: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate statistical analysis from high-sample benchmarks."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("highsample_benchmark2.csv"),
        help="Benchmark CSV to load (default: highsample_benchmark2.csv)",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("benchmark_outputs/highsample_analysis"),
        help="Directory where tables/plots will be written",
    )
    return parser.parse_args()


def load_rows(csv_path: Path) -> List[BenchmarkRow]:
    rows: List[BenchmarkRow] = []
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for entry in reader:
            rows.append(
                BenchmarkRow(
                    algorithm=entry["algorithm"],
                    device=entry["device"],
                    thread_mode=entry["thread_mode"],
                    m=int(entry["m"]),
                    n=int(entry["n"]),
                    N=int(entry["N"]),
                    mu_r=float(entry["mu_r"]),
                    sigma_r_ratio=float(entry["sigma_r_ratio"]),
                    s=int(entry["s"]),
                    intensity_pattern=entry["intensity_pattern"],
                    runtime_seconds=float(entry["runtime_seconds"]),
                )
            )
    return rows


def describe(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("describe() received an empty sequence")
    arr.sort()
    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    return {
        "count": float(arr.size),
        "median": float(np.median(arr)),
        "mean": float(arr.mean()),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "min": float(arr[0]),
        "max": float(arr[-1]),
    }


def runtime_summary(
    rows: Sequence[BenchmarkRow],
) -> Dict[Tuple[str, str], Dict[str, float]]:
    buckets: Dict[Tuple[str, str], List[float]] = {}
    for row in rows:
        key = (row.algorithm, row.device)
        buckets.setdefault(key, []).append(row.runtime_seconds)
    return {key: describe(values) for key, values in buckets.items()}


def compute_speedup_records(rows: Sequence[BenchmarkRow]) -> List[SpeedupRecord]:
    grouped: Dict[Tuple[str, int, int, float, str], Dict[str, List[float]]] = {}
    for row in rows:
        key = (row.algorithm, row.m, row.N, row.mu_r, row.intensity_pattern)
        grouped.setdefault(key, {}).setdefault(row.device, []).append(
            row.runtime_seconds
        )

    records: List[SpeedupRecord] = []
    for key, device_map in grouped.items():
        cpu_samples = device_map.get("cpu")
        gpu_samples = device_map.get("gpu")
        if not cpu_samples or not gpu_samples:
            continue
        cpu_median = float(np.median(cpu_samples))
        gpu_median = float(np.median(gpu_samples))
        if gpu_median <= 0:
            continue
        records.append(
            SpeedupRecord(
                algorithm=key[0],
                m=key[1],
                N=key[2],
                mu_r=key[3],
                intensity_pattern=key[4],
                cpu_median=cpu_median,
                gpu_median=gpu_median,
                speedup=cpu_median / gpu_median,
            )
        )
    return records


def speedup_summary(records: Sequence[SpeedupRecord]) -> Dict[str, Dict[str, float]]:
    buckets: Dict[str, List[float]] = {}
    for record in records:
        buckets.setdefault(record.algorithm, []).append(record.speedup)
    return {algo: describe(values) for algo, values in buckets.items()}


def sorted_unique(rows: Sequence[BenchmarkRow], attr: str) -> List[float | int | str]:
    return sorted({getattr(row, attr) for row in rows})


def plot_runtime_vs_field(
    rows: Sequence[BenchmarkRow],
    outdir: Path,
    field: str,
    filename: str,
    xlabel: str,
) -> Optional[Path]:
    algorithms = sorted_unique(rows, "algorithm")
    devices = sorted_unique(rows, "device")
    field_values = sorted_unique(rows, field)
    if len(field_values) < 2:
        return None
    aggregated: Dict[Tuple[str, str, float | int], List[float]] = {}

    for row in rows:
        key = (row.algorithm, row.device, getattr(row, field))
        aggregated.setdefault(key, []).append(row.runtime_seconds)

    fig, axes = plt.subplots(
        1, len(algorithms), figsize=(5 * len(algorithms), 4), sharey=True
    )
    axes = np.atleast_1d(axes)

    for idx, algorithm in enumerate(algorithms):
        ax = axes[idx]
        for device in devices:
            xs: List[float | int] = []
            medians: List[float] = []
            lower_err: List[float] = []
            upper_err: List[float] = []
            for value in field_values:
                stats = aggregated.get((algorithm, device, value))
                if not stats:
                    continue
                desc = describe(stats) if isinstance(stats, list) else stats
                median = desc["median"]
                xs.append(value)  # type: ignore[arg-type]
                medians.append(median)
                lower_err.append(median - desc["q1"])
                upper_err.append(desc["q3"] - median)
            if xs:
                ax.errorbar(
                    xs,
                    medians,
                    yerr=[lower_err, upper_err],
                    marker="o",
                    capsize=4,
                    label=device.upper(),
                )
        ax.set_title(f"{algorithm.capitalize()} runtime vs {xlabel.lower()}")
        ax.set_xlabel(xlabel)
        ax.set_yscale("log")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.4)
        if idx == 0:
            ax.set_ylabel("Median runtime (s, log scale)")

    axes[-1].legend(title="Device", loc="upper left")
    fig.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / filename
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_speedup_boxplots(records: Sequence[SpeedupRecord], outdir: Path) -> Path:
    algorithms = sorted({record.algorithm for record in records})
    fig, axes = plt.subplots(
        1, len(algorithms), figsize=(5 * len(algorithms), 4), sharey=True
    )
    axes = np.atleast_1d(axes)

    for idx, algorithm in enumerate(algorithms):
        ax = axes[idx]
        subset = [record for record in records if record.algorithm == algorithm]
        by_resolution: Dict[int, List[float]] = {}
        for record in subset:
            by_resolution.setdefault(record.m, []).append(record.speedup)
        ordered = sorted(by_resolution.items())
        data = [values for _, values in ordered]
        labels = [f"{resolution}px" for resolution, _ in ordered]
        if data:
            ax.boxplot(
                data,
                tick_labels=labels,
                showmeans=True,
                medianprops={"color": "black"},
                meanprops={
                    "marker": "o",
                    "markerfacecolor": "white",
                    "markeredgecolor": "black",
                },
            )
        ax.axhline(1.0, color="red", linestyle="--", linewidth=0.9, alpha=0.6)
        ax.set_title(f"{algorithm.capitalize()} GPU speedup vs CPU")
        ax.set_ylabel("CPU runtime / GPU runtime" if idx == 0 else "")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)

    fig.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "gpu_speedup_boxplots.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_parameter_heatmap(rows: Sequence[BenchmarkRow], outdir: Path) -> Path:
    algorithms = sorted_unique(rows, "algorithm")
    devices = sorted_unique(rows, "device")
    mu_values = sorted_unique(rows, "mu_r")
    patterns = sorted_unique(rows, "intensity_pattern")

    fig, axes = plt.subplots(
        len(devices),
        len(algorithms),
        figsize=(4 * len(algorithms), 3.5 * len(devices)),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_2d(axes)

    vmax = 0.0
    matrices: Dict[Tuple[int, int], np.ndarray] = {}
    for row_idx, device in enumerate(devices):
        for col_idx, algorithm in enumerate(algorithms):
            matrix = np.full((len(mu_values), len(patterns)), np.nan, dtype=float)
            for i, mu in enumerate(mu_values):
                for j, pattern in enumerate(patterns):
                    samples = [
                        r.runtime_seconds
                        for r in rows
                        if r.device == device
                        and r.algorithm == algorithm
                        and r.mu_r == mu
                        and r.intensity_pattern == pattern
                    ]
                    if samples:
                        matrix[i, j] = float(np.median(samples))
                        vmax = max(vmax, matrix[i, j])
            matrices[(row_idx, col_idx)] = matrix

    fig.subplots_adjust(wspace=0.15, hspace=0.25)
    for (row_idx, col_idx), matrix in matrices.items():
        ax = axes[row_idx, col_idx]
        im = ax.imshow(
            matrix,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            vmin=0,
            vmax=vmax or None,
        )
        ax.set_xticks(
            range(len(patterns)),
            labels=[p.capitalize() for p in patterns],
            rotation=30,
            ha="right",
        )
        ax.set_yticks(range(len(mu_values)), labels=[f"{mu:.2f}" for mu in mu_values])
        if col_idx == 0:
            ax.set_ylabel(f"{devices[row_idx].upper()} · μ_r")
        if row_idx == 0:
            ax.set_title(f"{algorithms[col_idx].capitalize()} runtime (s)")

        for i in range(len(mu_values)):
            for j in range(len(patterns)):
                value = matrix[i, j]
                if np.isnan(value):
                    continue
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    va="center",
                    ha="center",
                    color="white"
                    if value > (vmax * 0.55 if vmax else value)
                    else "black",
                    fontsize=8,
                )

    cbar = fig.colorbar(im, ax=axes, shrink=0.75, location="right", pad=0.02)
    cbar.set_label("Median runtime (s)")

    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "parameter_sensitivity_heatmap.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_coverage_heatmap(rows: Sequence[BenchmarkRow], outdir: Path) -> Path:
    ms = sorted_unique(rows, "m")
    sample_counts = sorted_unique(rows, "N")
    algorithms = sorted_unique(rows, "algorithm")
    devices = sorted_unique(rows, "device")
    mu_values = sorted_unique(rows, "mu_r")
    patterns = sorted_unique(rows, "intensity_pattern")

    expected_per_cell = len(algorithms) * len(devices) * len(mu_values) * len(patterns)
    if expected_per_cell == 0:
        raise ValueError("Cannot compute coverage with zero expected combinations")

    coverage: Dict[Tuple[int, int], set[Tuple[str, str, float, str]]] = {}
    for row in rows:
        key = (row.m, row.N)
        combo = (row.algorithm, row.device, row.mu_r, row.intensity_pattern)
        coverage.setdefault(key, set()).add(combo)

    matrix = np.zeros((len(sample_counts), len(ms)), dtype=float)
    for i, N in enumerate(sample_counts):
        for j, m in enumerate(ms):
            combos = coverage.get((m, N))
            ratio = len(combos) / expected_per_cell if combos else 0.0
            matrix[i, j] = ratio

    fig, ax = plt.subplots(figsize=(4 + len(ms), 3 + len(sample_counts)))
    im = ax.imshow(matrix, origin="lower", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(ms)), labels=[f"{m}px" for m in ms])
    ax.set_yticks(range(len(sample_counts)), labels=[f"N={N}" for N in sample_counts])
    ax.set_xlabel("Resolution (m)")
    ax.set_ylabel("Sample count")
    ax.set_title("Benchmark coverage (fraction of parameter grid filled)")

    for i in range(len(sample_counts)):
        for j in range(len(ms)):
            value = matrix[i, j]
            ax.text(
                j,
                i,
                f"{value:.0%}",
                ha="center",
                va="center",
                color="black" if value < 0.5 else "white",
            )

    fig.colorbar(im, ax=ax, shrink=0.7, label="Coverage")
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "coverage_heatmap.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def print_runtime_report(summary: Dict[Tuple[str, str], Dict[str, float]]) -> None:
    header = f"{'Algorithm':<10}{'Device':<8}{'Count':>8}{'Median (s)':>12}{'IQR (s)':>10}{'Mean (s)':>10}"
    print("\nRuntime distribution by algorithm and device")
    print(header)
    print("-" * len(header))
    for (algorithm, device), stats in sorted(summary.items()):
        print(
            f"{algorithm:<10}{device:<8}"
            f"{int(stats['count']):>8d}"
            f"{stats['median']:>12.4f}"
            f"{stats['iqr']:>10.4f}"
            f"{stats['mean']:>10.4f}"
        )


def print_speedup_report(
    summary: Dict[str, Dict[str, float]], records: Sequence[SpeedupRecord]
) -> None:
    header = f"{'Algorithm':<10}{'Pairs':>8}{'Median×':>10}{'p25×':>9}{'p75×':>9}{'Min×':>9}{'Max×':>9}"
    print("\nGPU speedup vs CPU (paired where both devices were run)")
    print(header)
    print("-" * len(header))
    counts: Dict[str, int] = {}
    for record in records:
        counts[record.algorithm] = counts.get(record.algorithm, 0) + 1
    for algorithm, stats in sorted(summary.items()):
        print(
            f"{algorithm:<10}"
            f"{counts.get(algorithm, 0):>8d}"
            f"{stats['median']:>10.2f}"
            f"{stats['q1']:>9.2f}"
            f"{stats['q3']:>9.2f}"
            f"{stats['min']:>9.2f}"
            f"{stats['max']:>9.2f}"
        )


def main() -> None:
    args = parse_args()
    if not args.csv.exists():
        raise SystemExit(f"CSV file not found: {args.csv}")

    rows = load_rows(args.csv)
    print(f"Loaded {len(rows)} benchmark rows from {args.csv}")

    runtime_stats = runtime_summary(rows)
    print_runtime_report(runtime_stats)

    speedup_records = compute_speedup_records(rows)
    if speedup_records:
        speedup_stats = speedup_summary(speedup_records)
        print_speedup_report(speedup_stats, speedup_records)
    else:
        print("\nNo CPU/GPU pairs found to compute speedups.")

    args.outdir.mkdir(parents=True, exist_ok=True)
    plot_paths = [
        plot_runtime_vs_field(
            rows, args.outdir, "m", "runtime_vs_resolution.png", "Resolution (px)"
        ),
        plot_runtime_vs_field(
            rows, args.outdir, "N", "runtime_vs_samples.png", "Sample count (N)"
        ),
        plot_runtime_vs_field(rows, args.outdir, "mu_r", "runtime_vs_mu.png", "μ_r"),
        plot_speedup_boxplots(speedup_records, args.outdir)
        if speedup_records
        else None,
        plot_parameter_heatmap(rows, args.outdir),
        plot_coverage_heatmap(rows, args.outdir),
    ]
    saved = [str(path) for path in plot_paths if path is not None]
    print("\nGenerated plots:")
    for path in saved:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
