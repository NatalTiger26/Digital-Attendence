"""Command-Line Interface (CLI) for Digital Attendance.

Provides commands to run benchmarks, process group images to mark attendance,
generate synthetic collages, and inspect dataset status.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from digital_attendance.config import (
    COLLAGES_DIR,
    DATASET_DIR,
    DEFAULT_OUTPUTS_DIR,
    DEFAULT_PCA_COMPONENTS,
    DEFAULT_TRAIN_SAMPLES_PER_CLASS,
)
from digital_attendance.data.collage import compose_grid_collage, save_collage
from digital_attendance.data.loader import load_dataset, split_dataset
from digital_attendance.evaluation.metrics import format_markdown_table, run_benchmark_suite
from digital_attendance.models.eigenfaces import EigenfaceClassifier
from digital_attendance.models.euclidean import MeanFaceClassifier
from digital_attendance.pipeline.attendance import DigitalAttendancePipeline


def build_parser() -> argparse.ArgumentParser:
    """Construct command-line argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="digital-attendance",
        description="Digital Attendance: Automated face detection & recognition pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # 1. Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmark across models on test split.")
    bench_parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_DIR,
        help="Path to Dataset root directory (default: %(default)s).",
    )
    bench_parser.add_argument(
        "--train-per-class",
        type=int,
        default=DEFAULT_TRAIN_SAMPLES_PER_CLASS,
        help="Number of training samples per class (default: %(default)s).",
    )
    bench_parser.add_argument(
        "--pca-components",
        type=int,
        default=DEFAULT_PCA_COMPONENTS,
        help="Number of PCA components K (default: %(default)s).",
    )

    # 2. Attendance command
    att_parser = subparsers.add_parser("attendance", help="Process group image / collage to mark attendance.")
    att_parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to target group photo or collage image.",
    )
    att_parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_DIR,
        help="Path to Dataset for training enrolled faces (default: %(default)s).",
    )
    att_parser.add_argument(
        "--model",
        choices=["pca", "euclidean"],
        default="pca",
        help="Classifier architecture to use (default: %(default)s).",
    )
    att_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUTS_DIR,
        help="Directory to save JSON, CSV, and annotated image (default: %(default)s).",
    )

    # 3. Make-collages command
    coll_parser = subparsers.add_parser("make-collages", help="Synthesize test collages from Dataset.")
    coll_parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_DIR,
        help="Path to Dataset root directory (default: %(default)s).",
    )
    coll_parser.add_argument(
        "--output-dir",
        type=Path,
        default=COLLAGES_DIR,
        help="Directory to write generated collages (default: %(default)s).",
    )
    coll_parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Number of collages to generate (default: %(default)s).",
    )

    return parser


def handle_benchmark(args: argparse.Namespace) -> int:
    """Execute the benchmark subcommand."""
    print(f"Loading dataset from: {args.dataset}")
    try:
        X, y, classes = load_dataset(args.dataset)
    except Exception as e:
        print(f"Error loading dataset: {e}", file=sys.stderr)
        return 1

    print(f"Loaded {X.shape[0]} total images across {len(classes)} classes.")
    X_train, y_train, X_test, y_test = split_dataset(X, y, train_per_class=args.train_per_class)
    print(f"Split: {X_train.shape[0]} train samples, {X_test.shape[0]} test samples.")

    print("\nRunning benchmark suite...")
    results = run_benchmark_suite(X_train, y_train, X_test, y_test)

    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    print(format_markdown_table(results))
    print("=" * 70 + "\n")
    return 0


def handle_attendance(args: argparse.Namespace) -> int:
    """Execute the attendance subcommand."""
    image_path = args.image.expanduser().resolve()
    if not image_path.is_file():
        print(f"Error: Target image file not found: {image_path}", file=sys.stderr)
        return 1

    print(f"Loading training data from: {args.dataset}...")
    try:
        X, y, classes = load_dataset(args.dataset)
    except Exception as e:
        print(f"Error loading dataset: {e}", file=sys.stderr)
        return 1

    classifier = (
        EigenfaceClassifier(n_components=DEFAULT_PCA_COMPONENTS)
        if args.model == "pca"
        else MeanFaceClassifier()
    )

    pipeline = DigitalAttendancePipeline(classifier=classifier)
    print(f"Fitting {args.model.upper()} classifier on {X.shape[0]} enrolled faces...")
    pipeline.fit(X, y)

    print(f"Processing group image: {image_path}...")
    report, annotated = pipeline.process_image(image_path, annotate=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    json_path = args.output_dir / f"{stem}_attendance.json"
    csv_path = args.output_dir / f"{stem}_attendance.csv"
    img_path = args.output_dir / f"{stem}_annotated.jpg"

    report.to_json(json_path)
    report.to_csv(csv_path)
    cv2.imwrite(str(img_path), annotated)

    print("\n" + "=" * 50)
    print(f"DIGITAL ATTENDANCE SUMMARY: {image_path.name}")
    print("=" * 50)
    print(f"Total Faces Detected: {report.total_detected}")
    print(f"Identified Present ({len(report.present_members)}): {', '.join(report.present_members)}")
    print(f"\nArtifacts saved to: {args.output_dir}")
    print(f"  - JSON Report: {json_path.name}")
    print(f"  - CSV Roster:  {csv_path.name}")
    print(f"  - Annotated:   {img_path.name}")
    print("=" * 50 + "\n")
    return 0


def handle_make_collages(args: argparse.Namespace) -> int:
    """Execute the make-collages subcommand."""
    print(f"Scanning dataset from: {args.dataset}...")
    class_dirs = sorted([d for d in args.dataset.iterdir() if d.is_dir() and not d.name.startswith(".")])
    if len(class_dirs) < 16:
        print(f"Error: Need at least 16 classes to form a 4x4 collage, found {len(class_dirs)}.", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating {args.count} collages in: {args.output_dir}...")

    for i in range(args.count):
        tiles: list = []
        labels: list[str] = []
        for c_dir in class_dirs[:16]:
            img_file = c_dir / f"{c_dir.name}_{i}.jpg"
            if not img_file.exists():
                images = list(c_dir.glob("*.jpg"))
                img_file = images[0] if images else None
            if img_file is not None:
                im = cv2.imread(str(img_file))
                if im is not None:
                    tiles.append(im)
                    labels.append(c_dir.name)

        if len(tiles) >= 16:
            res = compose_grid_collage(tiles[:16], labels[:16], grid_rows=4, grid_cols=4)
            out_file = args.output_dir / f"Collage_{i}.jpg"
            save_collage(res.image, out_file)
            print(f"  Wrote: {out_file.name}")

    print("Collage generation complete.")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for CLI execution."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "benchmark":
        return handle_benchmark(args)
    elif args.command == "attendance":
        return handle_attendance(args)
    elif args.command == "make-collages":
        return handle_make_collages(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
