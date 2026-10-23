"""PPG-Bench published evaluation implementation."""

from pathlib import Path
from typing import Any, Mapping

import numpy as np


def safe_study_name(text: str) -> str:
    return "".join(
        (ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(text))
    )


def optuna_storage_path(search_cfg: Mapping[str, Any], study_name: str) -> Path | None:
    storage_root = search_cfg.get("storage_root")
    if storage_root is None:
        return None
    path = Path(storage_root) / safe_study_name(study_name) / "optuna.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def make_optuna_storage(storage_path: Path):
    import optuna

    return optuna.storages.RDBStorage(
        f"sqlite:///{storage_path.resolve()}",
        engine_kwargs={"connect_args": {"timeout": 120}},
    )


def make_optuna_study(probe_cfg: Mapping[str, Any], study_name: str, seed: int):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    search_cfg = dict(probe_cfg.get("search", {}))
    sampler = optuna.samplers.TPESampler(
        seed=seed,
        n_startup_trials=int(search_cfg.get("n_startup_trials", 8)),
        multivariate=True,
        group=True,
        constant_liar=True,
    )
    max_resource = int(search_cfg.get("max_resource", probe_cfg["max_epochs"]))
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=int(search_cfg.get("min_resource", 1)),
        max_resource=max_resource,
        reduction_factor=int(search_cfg.get("reduction_factor", 3)),
    )
    storage_path = optuna_storage_path(search_cfg, study_name)
    storage = make_optuna_storage(storage_path) if storage_path is not None else None
    return optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=bool(search_cfg.get("load_if_exists", False)),
    )


def sample_list_space(
    trial, probe_cfg: Mapping[str, Any], default_space: Mapping[str, Any]
) -> dict[str, Any]:
    space = probe_cfg.get("space", default_space)
    if not isinstance(space, Mapping):
        raise ValueError("Optuna search space must be a mapping")
    params = {}
    for name, values in space.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"Optuna search space for {name} must be a non-empty list")
        params[name] = trial.suggest_categorical(name, values)
    return params


def report_value(current_score: float, best_score: float, mode: str) -> float:
    if mode == "current":
        value = current_score
    elif mode == "best_so_far":
        value = best_score
    else:
        raise ValueError(f"Unsupported Optuna report_score mode: {mode}")
    if not np.isfinite(value):
        raise ValueError("Optuna cannot report a non-finite validation score")
    return float(value)


def trial_records(study) -> list[dict[str, Any]]:
    records = []
    for trial in study.trials:
        records.append(
            {
                "number": int(trial.number),
                "state": trial.state.name,
                "value": None if trial.value is None else float(trial.value),
                "params": dict(trial.params),
                "intermediate_values": {
                    str(step): float(value)
                    for step, value in trial.intermediate_values.items()
                },
                "user_attrs": dict(trial.user_attrs),
            }
        )
    return records


def optuna_results(
    study,
    probe_cfg: Mapping[str, Any],
    best_candidate: Mapping[str, Any],
    effective_space: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    best_params = dict(best_candidate.get("params", {}))
    if best_candidate.get("aggregation") is not None:
        best_params["aggregation"] = dict(best_candidate["aggregation"])
    return {
        "search": dict(probe_cfg.get("search", {})),
        "space": dict(probe_cfg.get("space", effective_space or {})),
        "study_name": probe_cfg.get("study_name"),
        "storage_path": probe_cfg.get("storage_path"),
        "best_trial_number": best_candidate.get("trial_number"),
        "best_score": best_candidate.get("mean_score"),
        "best_epoch": best_candidate.get("selected_epoch"),
        "best_params": best_params,
        "trials": trial_records(study),
    }
