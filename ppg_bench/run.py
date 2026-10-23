"""One evaluation path for a dataset, a model, or the whole benchmark."""

import csv
import hashlib
import json
import os
import random
import shutil
from importlib.metadata import PackageNotFoundError, version
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import t
from sklearn.dummy import DummyClassifier, DummyRegressor
from torch.utils.data import DataLoader

from .classical import (
    fit_linear_classical_probe,
    fit_nonlinear_classical_probe,
    score_linear_classical_probe,
    score_linear_classical_subject_probe,
)
from .datasets import BenchmarkDataset, DATASETS
from .models import get_port
from .neural import fit_mlp_probe_optuna, run_finetune_probe
from .protocol import SEEDS, outer_splits, probe_config, subject_aggregation_grid


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            default=lambda x: x.item() if isinstance(x, np.generic) else str(x),
            allow_nan=False,
        )
    )
    temporary.replace(path)


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def extract(port, encoder, dataset, model, device, batch_size):
    encoder = encoder.to(device).eval()
    features = []
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            x = port.prepare_batch(batch, model, device)
            embedding = encoder(x)
            if (
                not isinstance(embedding, torch.Tensor)
                or embedding.ndim != 2
                or tuple(embedding.shape)
                != (len(batch["x"]), int(model["embedding_dim"]))
            ):
                raise ValueError(
                    f"{model['name']} must return (batch, {model['embedding_dim']}) embeddings"
                )
            features.append(embedding.detach().cpu().numpy())
    return np.concatenate(features)


def score(probe, features, dataset, task):
    if task.eval_level == "subject":
        aggregation = probe.best_params_.get("aggregation", {"name": "mean"})
        return score_linear_classical_subject_probe(
            probe,
            features,
            dataset.y,
            np.asarray(dataset.subjects),
            task.task_type,
            aggregation,
        )
    return score_linear_classical_probe(probe, features, dataset.y, task.task_type)


def predictions(probe, features, dataset, task):
    record = {
        "y_true": dataset.y,
        "y_pred": probe.predict(features),
        "subjects": np.asarray(dataset.subjects),
    }
    if task.task_type == "classification":
        record.update(y_prob=probe.predict_proba(features), classes=probe.classes_)
    return record


def baseline(train, test, task):
    y, subjects = train.y, np.asarray(train.subjects)
    if task.eval_level == "subject":
        indices = [
            np.flatnonzero(subjects == name)[0] for name in dict.fromkeys(subjects)
        ]
        y = y[indices]
    estimator = (
        DummyClassifier(strategy="most_frequent")
        if task.task_type == "classification"
        else DummyRegressor(strategy="mean")
    )
    estimator.fit(np.zeros((len(y), 1)), y)
    estimator.best_params_ = {"aggregation": {"name": "mean"}}
    return score(estimator, np.zeros((len(test), 1)), test, task)


def finite_json(value):
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return finite_json(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def evaluate(job, directory, cache_dir, device, batch_size):
    task, model = job["task"], job["model"]
    seed_everything(job["seed"])
    port = get_port(model)
    train = BenchmarkDataset(job["root"], job["train"], task)
    test = BenchmarkDataset(job["root"], job["test"], task)
    repeats = (
        range(3)
        if task.eval_level == "subject" and job["method"] in ("mlp", "finetune")
        else range(1)
    )
    x_train = x_test = None
    if job["method"] != "finetune":
        cache = (
            cache_dir
            / f"{job['model_id']}__{task.name}__seed{job['seed']}__fold{job['fold']}.npz"
        )
        if cache.exists():
            with np.load(cache) as saved:
                x_train, x_test = saved["train"], saved["test"]
        else:
            encoder = port.load_encoder()
            x_train = extract(port, encoder, train, model, device, batch_size)
            x_test = extract(port, encoder, test, model, device, batch_size)
            del encoder
            cache_dir.mkdir(parents=True, exist_ok=True)
            partial = cache.with_suffix(".tmp.npz")
            np.savez(partial, train=x_train, test=x_test)
            partial.replace(cache)
    results = []
    for repeat in repeats:
        config = probe_config(job["method"], job["seed"] + 1009 * repeat)
        config["study_name"] = f"{job['id']}__repeat{repeat}"
        if "search" in config:
            config["search"]["storage_root"] = str(directory / "search")
        aggregations = subject_aggregation_grid(task.task_type, task.eval_level)
        if job["method"] == "finetune":
            result = run_finetune_probe(
                port,
                train,
                test,
                model,
                config,
                task.task_type,
                task.eval_level,
                device,
            )
            prediction_record = result.pop("test_predictions")
        else:
            fit = {
                "linear": fit_linear_classical_probe,
                "rf": fit_nonlinear_classical_probe,
                "mlp": fit_mlp_probe_optuna,
            }[job["method"]]
            args = (
                x_train,
                train.y,
                task.task_type,
                config,
                np.asarray(train.subjects),
                task.eval_level,
                aggregations,
            )
            probe = fit(*args, device) if job["method"] == "mlp" else fit(*args)
            result = {"test": score(probe, x_test, test, task)}
            if hasattr(probe, "optuna_results_"):
                result["optuna_results"] = probe.optuna_results_
            prediction_record = predictions(probe, x_test, test, task)
        result.pop("inner_cv_results", None)
        if "optuna_results" in result:
            search = result["optuna_results"]
            search["storage_path"] = str(
                Path("search") / config["study_name"] / "optuna.db"
            )
            search["search"]["storage_root"] = "search"
        result["repeat"] = repeat
        result["predictions"] = f"predictions_repeat{repeat}.npz"
        np.savez_compressed(directory / result["predictions"], **prediction_record)
        results.append(result)
    return finite_json(
        {
            "model": model["name"],
            "task": task.name,
            "method": job["method"],
            "seed": job["seed"],
            "fold": job["fold"],
            "train_groups": job["train"],
            "test_groups": job["test"],
            "results": results,
            "baseline": baseline(train, test, task),
        }
    )


def make_jobs(models, tasks, processed, methods, seeds=SEEDS, folds=range(5)):
    jobs = []
    for model in models:
        model_id = hashlib.sha256(model["name"].encode()).hexdigest()[:12]
        for task in tasks:
            root = (
                Path(processed).resolve() / f"fs_{model['fs']}" / DATASETS[task.dataset]
            )
            if not root.is_dir():
                raise FileNotFoundError(
                    f"Prepare {task.dataset} at {model['fs']} Hz first: {root}"
                )
            for seed in seeds:
                for fold, (train, test) in enumerate(
                    outer_splits(root, task.dataset, seed)
                ):
                    if fold not in folds:
                        continue
                    for method in methods:
                        if not task.primary and method in ("mlp", "finetune"):
                            continue
                        identity = (
                            f"{model_id}__{task.name}__{method}__seed{seed}__fold{fold}"
                        )
                        jobs.append(
                            {
                                "id": identity,
                                "model_id": model_id,
                                "model": model,
                                "task": task,
                                "root": str(root),
                                "method": method,
                                "seed": seed,
                                "fold": fold,
                                "train": train,
                                "test": test,
                            }
                        )
    if not jobs:
        raise ValueError(
            "No evaluations selected; neural methods evaluate primary tasks only"
        )
    return jobs


def run_identity(models, jobs, device, batch_size):
    signatures = {}
    for job in jobs:
        root = Path(job["root"])
        if str(root) in signatures:
            continue
        signatures[str(root)] = [
            (str(p.relative_to(root)), p.stat().st_size, p.stat().st_mtime_ns)
            for p in sorted(root.glob("*/*"))
            if p.suffix in (".npy", ".json")
        ]
    checkpoint_signatures = {}
    for model in models:
        checkpoint = model["init"].get("ckpt_path")
        if checkpoint:
            path = Path(checkpoint)
            files = (
                [path]
                if path.is_file()
                else sorted(p for p in path.rglob("*") if p.is_file())
            )
            checkpoint_signatures[checkpoint] = [
                (str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in files
            ]
    code_hash = hashlib.sha256(
        b"".join(p.read_bytes() for p in sorted(Path(__file__).parent.rglob("*.py")))
    ).hexdigest()
    versions = {}
    for package in ("numpy", "scipy", "scikit-learn", "torch", "optuna", "cuml-cu12"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            pass
    return {
        "models": models,
        "datasets": signatures,
        "checkpoints": checkpoint_signatures,
        "code_sha256": code_hash,
        "device": device,
        "feature_batch_size": batch_size,
        "versions": versions,
    }


def summarize(output, jobs, cache=None):
    cache = {} if cache is None else cache
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    completed = 0
    multiple_models = len({job["model"]["name"] for job in jobs}) > 1
    for job in jobs:
        path = output / "jobs" / job["id"] / "results.json"
        if job["id"] in cache and cache[job["id"]] is None:
            continue
        metric = (
            "mae"
            if job["task"].task_type == "regression"
            else "argmax_balanced_accuracy"
            if job["task"].task_type == "regression-based-cls"
            else "balanced_accuracy"
        )
        if job["id"] not in cache:
            if not path.exists():
                cache[job["id"]] = None
                continue
            record = json.loads(path.read_text())
            cache[job["id"]] = {
                **{
                    key: record[key]
                    for key in ("model", "task", "method", "seed", "fold")
                },
                "results": [
                    {"test": {metric: item["test"].get(metric)}}
                    for item in record["results"]
                ],
                "baseline": {metric: record["baseline"].get(metric)},
            }
        record = cache[job["id"]]
        completed += 1
        for name, evaluations in [
            (record["model"], [r["test"] for r in record["results"]]),
            (
                f"TrainMean/{record['model']}" if multiple_models else "TrainMean",
                [record["baseline"]],
            ),
        ]:
            values = [r.get(metric) for r in evaluations if r.get(metric) is not None]
            if values:
                grouped[(name, record["task"], record["method"], metric)][
                    record["seed"]
                ][record["fold"]].extend(values)
    rows = []
    for (model, task, method, metric), seeds in sorted(grouped.items()):
        seed_means = [
            np.mean([np.mean(v) for v in folds.values()]) for folds in seeds.values()
        ]
        count = len(seed_means)
        rows.append(
            {
                "model": model,
                "task": task,
                "method": method,
                "metric": metric,
                "mean": float(np.mean(seed_means)),
                "ci95": float(
                    t.ppf(0.975, count - 1)
                    * np.std(seed_means, ddof=1)
                    / np.sqrt(count)
                )
                if count > 1
                else None,
                "n_seeds": count,
                "n_folds": sum(len(folds) for folds in seeds.values()),
            }
        )
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "model",
                "task",
                "method",
                "metric",
                "mean",
                "ci95",
                "n_seeds",
                "n_folds",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        output / "status.json",
        {
            "completed": completed,
            "planned": len(jobs),
            "complete": completed == len(jobs),
        },
    )


def run(
    models,
    tasks,
    processed,
    methods,
    output,
    device="auto",
    batch_size=256,
    seeds=SEEDS,
    folds=range(5),
    dry_run=False,
):
    device = (
        "cuda"
        if device == "auto" and torch.cuda.is_available()
        else "cpu"
        if device == "auto"
        else device
    )
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(
            "CUDA is unavailable; use --device cpu for linear/MLP/finetuning"
        )
    if device.startswith("cuda"):
        torch.cuda.set_device(device if ":" in device else "cuda:0")
    jobs = make_jobs(models, tasks, processed, methods, seeds, folds)
    print(f"Selected {len(jobs)} model/task/method/seed/fold evaluations", flush=True)
    if dry_run:
        return
    if "rf" in methods and any(
        j["task"].task_type != "regression-based-cls" and j["method"] == "rf"
        for j in jobs
    ):
        try:
            import cuml  # noqa: F401
        except ImportError as error:
            raise ImportError(
                "The published RF backend needs RAPIDS cuML. Install the rf extra, or select --methods linear mlp finetune."
            ) from error
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    identity = finite_json(run_identity(models, jobs, device, batch_size))
    manifest = output / "run.json"
    if manifest.exists() and json.loads(manifest.read_text())["identity"] != identity:
        raise ValueError(
            "Output belongs to a different model/data/protocol run. Choose a new --output directory."
        )
    write_json(
        manifest,
        {
            "identity": identity,
            "seeds": list(seeds),
            "folds": list(folds),
            "protocol": {method: probe_config(method, 42) for method in methods},
            "jobs": [j["id"] for j in jobs],
        },
    )
    (output / "jobs").mkdir(exist_ok=True)
    summary_cache = {}
    for index, job in enumerate(jobs, 1):
        directory = output / "jobs" / job["id"]
        if (directory / "results.json").exists():
            continue
        print(
            f"[{index}/{len(jobs)}] {job['model']['name']} {job['task'].name} {job['method']} seed={job['seed']} fold={job['fold']}",
            flush=True,
        )
        temporary = output / "jobs" / (job["id"] + ".working")
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir()
        try:
            result = evaluate(job, temporary, output / "features", device, batch_size)
            write_json(temporary / "results.json", result)
            temporary.rename(directory)
            summary_cache.pop(job["id"], None)
        finally:
            summarize(output, jobs, summary_cache)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    summarize(output, jobs, summary_cache)
    print(f"Results: {output / 'summary.csv'}", flush=True)
