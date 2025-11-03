#!/usr/bin/env python3
"""
Benchmark harness for the Rust `filmgrain` renderer.

The script sweeps a grid of model parameters, renders synthetic intensity fields
with the filmgrain CLI, and records timing statistics. Each configuration is
run twice: once with Rayon pinned to a single thread and once with the default
multithreaded scheduler. Runs execute sequentially so the renderer can occupy
the whole machine when multithreaded.

Usage example:
    python3 tools/film_grain_bench.py --csv results.csv --repeats 3 \
        --keep-outputs --resume
"""

import argparse
import csv
import itertools
import os
import statistics
import struct
import subprocess
import sys
import time
import uuid
import zlib
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

RESOLUTIONS: Sequence[int] = [128, 256, 512, 1024]
N_VALUES: Sequence[int] = [1]
MU_VALUES: Sequence[float] = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
SIGMA_RATIO_VALUES: Sequence[float] = [0.0, 0.25, 0.5, 0.75, 1.0]
ZOOM_VALUES: Sequence[int] = [1, 2, 4, 8]
INTENSITY_PATTERNS: Sequence[str] = ["constant", "step", "ramp", "natural"]
DEFAULT_SIGMA_FILTER = 0.8
RUN_VARIANTS: Sequence[Tuple[str, str, Optional[int]]] = (
    ("cpu", "single", 1),
    ("cpu", "multi", None),
    ("gpu", "gpu", None),
)
TOTAL_BASE_CONFIGS = (
    len(RESOLUTIONS)
    * len(N_VALUES)
    * len(MU_VALUES)
    * len(SIGMA_RATIO_VALUES)
    * len(ZOOM_VALUES)
    * len(INTENSITY_PATTERNS)
)

CSV_HEADER: Sequence[str] = (
    "algorithm",
    "device",
    "thread_mode",
    "m",
    "n",
    "N",
    "mu_r",
    "sigma_r_ratio",
    "s",
    "intensity_pattern",
    "alpha",
    "delta",
    "runtime_seconds",
)
KEY_COLUMNS: Sequence[str] = CSV_HEADER[:-1]


def fmt_float(value: Optional[float]) -> str:
    if value is None:
        return ""
    if abs(value) >= 1e6 or (0 < abs(value) < 1e-3):
        return f"{value:.6e}"
    return f"{value:.10g}"


def parse_cargo_args(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    return [token for token in raw.strip().split(" ") if token]


def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    length = struct.pack(">I", len(data))
    crc = struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    return length + chunk_type + data + crc


def write_grayscale_png(path: Path, image: np.ndarray) -> None:
    if image.dtype != np.uint8:
        raise ValueError("PNG writer expects uint8 data in range [0, 255]")
    height, width = image.shape
    header = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)  # 8-bit grayscale
    raw_rows = b"".join(b"\x00" + row.tobytes() for row in image)
    compressed = zlib.compress(raw_rows, level=3)
    payload = (
        header
        + png_chunk(b"IHDR", ihdr)
        + png_chunk(b"IDAT", compressed)
        + png_chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def bilinear_resize(field: np.ndarray, size: int) -> np.ndarray:
    src_h, src_w = field.shape
    xs = np.linspace(0, src_w - 1, num=size)
    intermediate = np.empty((src_h, size), dtype=np.float64)
    x_coords = np.arange(src_w, dtype=np.float64)
    for row_idx in range(src_h):
        intermediate[row_idx] = np.interp(xs, x_coords, field[row_idx])
    ys = np.linspace(0, src_h - 1, num=size)
    y_coords = np.arange(src_h, dtype=np.float64)
    resized = np.empty((size, size), dtype=np.float64)
    for col_idx in range(size):
        resized[:, col_idx] = np.interp(ys, y_coords, intermediate[:, col_idx])
    return resized


class IntensityFieldCache:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[Tuple[str, int], Path] = {}
        self._natural_base: Optional[np.ndarray] = None

    def ensure(self, pattern: str, size: int) -> Path:
        key = (pattern, size)
        cached = self._cache.get(key)
        if cached is not None and cached.exists():
            return cached
        path = self.root / f"{pattern}_{size}.png"
        if not path.exists():
            print(
                f"[input] generating intensity field pattern={pattern} size={size}",
                flush=True,
            )
            image = self._generate_field(pattern, size)
            write_grayscale_png(path, image)
        self._cache[key] = path
        return path

    def _generate_field(self, pattern: str, size: int) -> np.ndarray:
        if pattern == "constant":
            field = np.full((size, size), 0.5, dtype=np.float64)
        elif pattern == "step":
            field = np.zeros((size, size), dtype=np.float64)
            field[size // 2 :, :] = 1.0
        elif pattern == "ramp":
            field = np.tile(
                np.linspace(0.0, 1.0, num=size, dtype=np.float64), (size, 1)
            )
        elif pattern == "natural":
            field = self._natural_field(size)
        else:
            raise ValueError(f"Unknown intensity pattern '{pattern}'")
        field = np.clip(field, 0.0, 1.0)
        return np.rint(field * 255.0).astype(np.uint8)

    def _natural_field(self, size: int) -> np.ndarray:
        base = self._natural_base_field()
        if size == base.shape[0]:
            return base
        return bilinear_resize(base, size)

    def _natural_base_field(self) -> np.ndarray:
        if self._natural_base is None:
            rng = np.random.default_rng(20240611)
            base = rng.normal(
                loc=0.5, scale=0.2, size=(min(RESOLUTIONS), min(RESOLUTIONS))
            )
            self._natural_base = np.clip(base, 0.0, 1.0)
        return self._natural_base


def build_binary(repo_root: Path, cargo_args: List[str]) -> None:
    cmd = ["cargo", "build", "--release"] + cargo_args
    subprocess.run(cmd, cwd=str(repo_root), check=True)


def parse_runtime(stdout: str, stderr: str) -> Optional[float]:
    for line in f"{stdout}\n{stderr}".splitlines():
        if "time elapsed" in line:
            try:
                return float(line.split(":")[-1].strip())
            except ValueError:
                continue
    return None


def run_algorithm(
    repo_root: Path,
    binary: Path,
    image_path: Path,
    output_dir: Path,
    config: Dict[str, float],
    algorithm_name: str,
    device_mode: str,
    thread_mode: str,
    repeats: int,
    sigma_filter: float,
    env: Dict[str, str],
    keep_outputs: bool,
) -> Tuple[float, str, str]:
    runtimes: List[float] = []
    captured_stdout, captured_stderr = "", ""
    for run_idx in range(repeats):
        token = (
            f"{algorithm_name}"
            f"_device-{device_mode}"
            f"_threads-{thread_mode}"
            f"_pattern-{config['pattern']}"
            f"_m-{int(config['m'])}"
            f"_zoom-{fmt_float(config['s'])}"
            f"_mu-{fmt_float(config['mu_r'])}"
            f"_sigmaRatio-{fmt_float(config['sigma_r_ratio'])}"
            f"_N-{int(config['N'])}"
            f"_seed-{5489 + run_idx}"
        )
        safe_token = token.replace("/", "_")
        output_file = output_dir / f"{safe_token}.png"
        cmd = [
            str(binary),
            str(image_path),
            "--output",
            str(output_file),
            "--radius",
            fmt_float(config["mu_r"]),
            "--iters",
            str(int(config["N"])),
            "--zoom",
            fmt_float(config["s"]),
            "--sigma",
            fmt_float(sigma_filter),
            "--algo",
            algorithm_name,
            "--color-mode",
            "luma",
            "--device",
            device_mode,
        ]
        sigma_ratio = config["sigma_r_ratio"]
        if sigma_ratio > 0.0:
            sigma_linear = config["mu_r"] * sigma_ratio
            cmd.extend(
                [
                    "--radius-dist",
                    "lognorm",
                    "--radius-stddev",
                    fmt_float(sigma_linear),
                ]
            )
        else:
            cmd.extend(["--radius-dist", "const"])
        cmd.extend(["--seed", str(5489 + run_idx)])
        start = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                check=True,
                env=env,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Command failed: {' '.join(cmd)}\nstdout:\n{exc.stdout}\nstderr:\n{exc.stderr}"
            ) from exc
        elapsed = time.perf_counter() - start
        captured_stdout, captured_stderr = proc.stdout, proc.stderr
        measured = parse_runtime(proc.stdout, proc.stderr)
        runtimes.append(measured if measured is not None else elapsed)
        if not keep_outputs and output_file.exists():
            output_file.unlink()
    median_runtime = statistics.median(runtimes)
    return median_runtime, captured_stdout, captured_stderr


def load_existing(csv_path: Path) -> Dict[Tuple[str, ...], Dict[str, str]]:
    if not csv_path.exists():
        return {}
    existing: Dict[Tuple[str, ...], Dict[str, str]] = {}
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = tuple(row.get(col, "") for col in KEY_COLUMNS)
            existing[key] = row
    return existing


def open_csv(csv_path: Path, resume: bool) -> Tuple[csv.DictWriter, object]:
    file_exists = csv_path.exists()
    mode = "a" if resume and file_exists else "w"
    handle = csv_path.open(mode, newline="")
    writer = csv.DictWriter(handle, fieldnames=CSV_HEADER)
    if not (resume and file_exists):
        writer.writeheader()
    return writer, handle


def make_key(row: Dict[str, str]) -> Tuple[str, ...]:
    return tuple(row[col] for col in KEY_COLUMNS)


def base_config_iterator() -> Iterator[Dict[str, float]]:
    for size, N, mu_r, sigma_ratio, zoom, pattern in itertools.product(
        RESOLUTIONS,
        N_VALUES,
        MU_VALUES,
        SIGMA_RATIO_VALUES,
        ZOOM_VALUES,
        INTENSITY_PATTERNS,
    ):
        config = {
            "m": size,
            "n": size,
            "m_out": int(size * zoom),
            "n_out": int(size * zoom),
            "N": N,
            "mu_r": mu_r,
            "sigma_r_ratio": sigma_ratio,
            "s": zoom,
            "pattern": pattern,
        }
        yield config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark grain-wise vs pixel-wise algorithms."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Project root (defaults to the directory containing the Makefile).",
    )
    parser.add_argument(
        "--binary",
        type=Path,
        default=None,
        help="Path to the filmgrain binary (defaults to Cargo release build).",
    )
    parser.add_argument(
        "--skip-build", action="store_true", help="Assume the binary already exists."
    )
    parser.add_argument(
        "--cargo-args",
        type=str,
        default="",
        help="Extra args forwarded to `cargo build --release`.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("benchmark_results.csv"),
        help="CSV output path.",
    )
    parser.add_argument(
        "--input-cache-dir",
        type=Path,
        default=Path("benchmark_inputs"),
        help="Cache for generated PNGs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_outputs"),
        help="Where to place temporary outputs.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Timed runs per configuration (median reported).",
    )
    parser.add_argument(
        "--sigma-filter",
        type=float,
        default=DEFAULT_SIGMA_FILTER,
        help="sigmaFilter value passed to binary.",
    )
    parser.add_argument(
        "--max-configs",
        type=int,
        default=None,
        help="Cap total new rows (useful for smoke tests).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing CSV instead of starting fresh.",
    )
    parser.add_argument(
        "--keep-outputs",
        action="store_true",
        help="Keep generated grain images instead of deleting them.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    binary = (
        args.binary.resolve()
        if args.binary
        else (repo_root / "target" / "release" / "film_grain")
    )
    print(f"[setup] repo root: {repo_root}", flush=True)
    print(f"[setup] target binary: {binary}", flush=True)
    if not binary.exists():
        if args.skip_build:
            raise FileNotFoundError(
                f"Binary not found at {binary}; rerun without --skip-build."
            )
        print("[build] running `cargo build --release` for filmgrain...", flush=True)
        build_binary(repo_root, parse_cargo_args(args.cargo_args))
        if not binary.exists():
            raise FileNotFoundError(
                f"Compilation finished but binary still missing at {binary}"
            )
        print("[build] build completed.", flush=True)
    else:
        print("[setup] existing binary found; skipping build.", flush=True)

    input_cache = IntensityFieldCache(repo_root / args.input_cache_dir)
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_rows = load_existing(args.csv.resolve()) if args.resume else {}
    if existing_rows:
        print(
            f"[resume] loaded {len(existing_rows)} existing rows from {args.csv}.",
            flush=True,
        )
    else:
        print("[resume] starting fresh; CSV will be created.", flush=True)
    writer, handle = open_csv(args.csv.resolve(), resume=args.resume)
    env = os.environ.copy()

    total_algorithms = 2
    total_variants = len(RUN_VARIANTS)
    total_runs = TOTAL_BASE_CONFIGS * total_algorithms * total_variants
    print(
        f"[bench] sweeping {TOTAL_BASE_CONFIGS} base configurations "
        f"× {total_algorithms} algos × {total_variants} device/thread variants "
        f"= {total_runs} runs.",
        flush=True,
    )
    if args.max_configs is not None:
        print(
            f"[bench] will stop after {args.max_configs} new results.",
            flush=True,
        )

    # Build list of work items (do not execute yet). Each item corresponds to one CSV row to produce.
    tasks: List[Tuple[Dict[str, float], str, str, str, Optional[int], Dict[str, str]]] = []
    try:
        for base_config in base_config_iterator():
            for algorithm_name in ("grain", "pixel"):
                for device_mode, mode_name, rayon_threads in RUN_VARIANTS:
                    result_row = {
                        "algorithm": algorithm_name,
                        "device": device_mode,
                        "thread_mode": mode_name,
                        "m": str(int(base_config["m"])),
                        "n": str(int(base_config["n"])),
                        "N": str(int(base_config["N"])),
                        "mu_r": fmt_float(base_config["mu_r"]),
                        "sigma_r_ratio": fmt_float(base_config["sigma_r_ratio"]),
                        "s": fmt_float(base_config["s"]),
                        "intensity_pattern": base_config["pattern"],
                        "alpha": "",
                        "delta": "",
                        "runtime_seconds": "",
                    }
                    key = make_key(result_row)
                    if key in existing_rows:
                        continue
                    tasks.append(
                        (
                            base_config.copy(),
                            algorithm_name,
                            device_mode,
                            mode_name,
                            rayon_threads,
                            result_row,
                        )
                    )
                    if args.max_configs is not None and len(tasks) >= args.max_configs:
                        break
                if args.max_configs is not None and len(tasks) >= args.max_configs:
                    break
            if args.max_configs is not None and len(tasks) >= args.max_configs:
                break
    except Exception:
        # In case of unexpected generator issues, keep behavior conservative.
        raise

    if not tasks:
        print(
            "[bench] no new tasks to run (all rows already present or no tasks generated).",
            flush=True,
        )
        handle.close()
        return

    # Pre-generate (ensure) input PNGs to avoid races inside IntensityFieldCache.ensure
    unique_inputs = {(t[0]["pattern"], int(t[0]["m"])) for t in tasks}
    input_paths: Dict[Tuple[str, int], Path] = {}
    for pattern, size in unique_inputs:
        input_paths[(pattern, size)] = input_cache.ensure(pattern, size)

    total_written = 0
    try:
        for (
            base_config,
            algorithm_name,
            device_mode,
            mode_name,
            rayon_threads,
            result_row,
        ) in tasks:
            pattern = base_config["pattern"]
            size = int(base_config["m"])
            image_path = input_paths[(pattern, size)]
            config = {
                **base_config,
                "n_out": base_config["n_out"],
                "m_out": base_config["m_out"],
            }
            run_env = env.copy()
            if device_mode == "cpu" and rayon_threads is not None:
                run_env["RAYON_NUM_THREADS"] = str(max(1, rayon_threads))
            else:
                run_env.pop("RAYON_NUM_THREADS", None)

            key = make_key(result_row)
            try:
                runtime, stdout, stderr = run_algorithm(
                    repo_root,
                    binary,
                    image_path,
                    output_dir,
                    config,
                    algorithm_name,
                    device_mode,
                    mode_name,
                    max(1, args.repeats),
                    args.sigma_filter,
                    run_env,
                    args.keep_outputs,
                )
            except Exception as exc:
                print(f"[error] run failed for key={key}: {exc}", flush=True)
                continue

            result_row["runtime_seconds"] = fmt_float(runtime)
            writer.writerow(result_row)
            handle.flush()
            existing_rows[key] = result_row
            total_written += 1
            print(
                f"[{total_written}] device={device_mode} mode={mode_name} algo={result_row['algorithm']} "
                f"pattern={result_row['intensity_pattern']} size={result_row['m']} "
                f"N={result_row['N']} mu_r={result_row['mu_r']} "
                f"sigma_ratio={result_row['sigma_r_ratio']} zoom={result_row['s']} "
                f"runtime={runtime:.4f}s",
                flush=True,
            )
    finally:
        handle.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(130)
