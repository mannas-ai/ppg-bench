"""ppg-bench init | list | download | preprocess | run."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Download, prepare and benchmark PPG datasets."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List available datasets and tasks")
    init = commands.add_parser("init", help="Copy model configs and an adapter example")
    init.add_argument("--output", default=".", help="Destination project directory")
    download = commands.add_parser(
        "download", help="Download raw datasets from their publishers"
    )
    download.add_argument("datasets", nargs="+", help="Dataset names, or all")
    download.add_argument("--output", default="data/raw")
    download.add_argument("--physionet-user", help="Approved account for DREAMT")
    preproc = commands.add_parser(
        "preprocess", help="Prepare the benchmark windows and labels"
    )
    preproc.add_argument("datasets", nargs="+", help="Dataset names, or all")
    preproc.add_argument(
        "--raw", default="data/raw", help="Directory containing dataset subdirectories"
    )
    preproc.add_argument(
        "--raw-root", help="Use an existing extracted dataset (one dataset only)"
    )
    preproc.add_argument("--output", default="data/processed")
    preproc.add_argument(
        "--fs", nargs="+", type=int, required=True, help="Your models' sampling rates"
    )
    run = commands.add_parser(
        "run", help="Run one dataset or the full published benchmark"
    )
    run.add_argument(
        "--model", required=True, help="Model YAML: a mapping or a list of models"
    )
    run.add_argument(
        "--datasets", nargs="+", default=["all"], help="Dataset/task names, or all"
    )
    run.add_argument("--processed", default="data/processed")
    run.add_argument(
        "--methods",
        nargs="+",
        choices=["linear", "rf", "mlp", "finetune"],
        default=["linear", "rf", "mlp", "finetune"],
    )
    run.add_argument("--output", default="results/benchmark")
    run.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:N")
    run.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Embedding extraction batch size; head batch sizes stay fixed by the published grid",
    )
    run.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        choices=[42, 43, 44, 45, 46],
        default=[42, 43, 44, 45, 46],
        help="Select a subset for a quick check; the full benchmark uses all five",
    )
    run.add_argument(
        "--folds", nargs="+", type=int, choices=range(5), default=list(range(5))
    )
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and print the evaluation count",
    )
    args = parser.parse_args()
    try:
        if args.command == "init":
            from importlib.resources import files
            from pathlib import Path
            import shutil

            templates = files("ppg_bench") / "templates"
            output = Path(args.output)
            for name in ("configs", "examples"):
                if (output / name).exists():
                    raise ValueError(f"{output / name} already exists; choose a new --output directory")
            for name in ("configs", "examples"):
                shutil.copytree(templates / name, output / name)
            print(f"Created model configs and examples in {output.resolve()}")
        elif args.command == "download":
            from .download import SOURCES, download

            datasets = (
                list(SOURCES)
                if "all" in args.datasets
                else list(dict.fromkeys(args.datasets))
            )
            if any(name not in SOURCES for name in datasets):
                raise ValueError(f"Use dataset names from: {', '.join(SOURCES)}")
            download(datasets, args.output, args.physionet_user)
        elif args.command == "preprocess":
            from .datasets import DATASETS
            from .preprocess import preprocess

            datasets = (
                list(DATASETS)
                if "all" in args.datasets
                else list(dict.fromkeys(args.datasets))
            )
            if any(name not in DATASETS for name in datasets) or any(
                rate <= 0 for rate in args.fs
            ):
                raise ValueError("Use valid dataset names and positive sampling rates")
            preprocess(datasets, args.raw, args.output, args.fs, args.raw_root)
        elif args.command == "list":
            from .datasets import DATASETS, TASKS

            for dataset in DATASETS:
                print(dataset)
                for task in TASKS.values():
                    if task.dataset == dataset:
                        print(
                            f"  {task.name:32} {task.task_type:20} {task.eval_level:7} {'primary' if task.primary else 'secondary'}"
                        )
        else:
            from .datasets import select_tasks
            from .models import load_models
            from .run import run

            if args.batch_size <= 0:
                raise ValueError("--batch-size must be positive")
            run(
                load_models(args.model),
                select_tasks(args.datasets),
                args.processed,
                list(dict.fromkeys(args.methods)),
                args.output,
                args.device,
                args.batch_size,
                list(dict.fromkeys(args.seeds)),
                list(dict.fromkeys(args.folds)),
                args.dry_run,
            )
    except (
        ValueError,
        FileNotFoundError,
        FileExistsError,
        ImportError,
        RuntimeError,
    ) as error:
        print(f"ppg-bench: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
