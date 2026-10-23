STANDARD_LINEAR_REG_PARAM_GRID = {
    "reg__alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0],
    "reg__solver": ["auto"],
}
STANDARD_LINEAR_CLS_PARAM_GRID = [
    {
        "clf__penalty": ["l2"],
        "clf__C": [0.001, 0.01, 0.1, 1.0, 10.0],
        "clf__solver": ["lbfgs"],
    }
]
STANDARD_RF_REG_PARAM_GRID = {
    "n_estimators": [100],
    "max_depth": [5, 10, 20],
    "min_samples_split": [2],
    "min_samples_leaf": [1],
    "max_features": ["sqrt"],
}
STANDARD_RF_CLS_PARAM_GRID = {
    "n_estimators": [100],
    "max_depth": [5, 10, 20],
    "min_samples_split": [2],
    "min_samples_leaf": [1],
    "max_features": ["sqrt"],
}
STANDARD_MLP_PARAM_GRID = {
    "hidden_dim": [256, 512],
    "dropout": [0.0, 0.3],
    "lr": [1e-05, 0.0001, 0.001, 0.005, 0.01],
    "weight_decay": [0.0],
    "batch_size": [16, 32, 64, 128],
    "activation": ["gelu"],
}
STANDARD_FINETUNE_PARAM_GRID = {
    "hidden_dim": [256],
    "head_dropout": [0.0, 0.3],
    "head_lr": [0.001, 0.0001, 1e-05],
    "encoder_lr_ratio": [0.1, 0.01],
    "head_weight_decay": [0.0, 0.0001],
    "encoder_weight_decay_ratio": [1.0],
    "batch_size": [128, 512, 1024, 2048],
    "activation": ["gelu"],
    "batchnorm_mode": ["train", "freeze_stats"],
}

import copy
import hashlib
import random

SEEDS = (42, 43, 44, 45, 46)
METHODS = ("linear", "rf", "mlp", "finetune")


def outer_splits(root, dataset, seed):
    from pathlib import Path

    subjects = sorted(p.name for p in Path(root).iterdir() if (p / "ppg.npy").is_file())
    if len(subjects) < 5:
        raise ValueError(f"{dataset} needs at least five processed split groups")
    digest = hashlib.sha256(f"{seed}:{dataset}".encode()).hexdigest()
    random.Random(int(digest[:16], 16)).shuffle(subjects)
    offset = 0
    for index in range(5):
        size = len(subjects) // 5 + (index < len(subjects) % 5)
        test = subjects[offset : offset + size]
        yield [s for s in subjects if s not in set(test)], test
        offset += size


def subject_aggregation_grid(task_type, eval_level, probe_cfg=None):
    if eval_level != "subject":
        return None
    if task_type == "regression":
        return [{"name": "mean"}]
    if task_type == "regression-based-cls":
        return [{"name": "mean"}, {"name": "median"}]
    return [
        {"name": "mean"},
        {"name": "median"},
        {"name": "topk_mean", "topk_frac": 0.1},
    ]


def probe_config(method, seed):
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    config = {
        "inner_folds": 4,
        "random_state": seed,
        "class_weight": None,
        "max_iter": 1000,
        "param_grid": "standard",
        "backend": "cuml",
        "n_bins": 128,
        "regression_scoring": "neg_mean_absolute_error",
        "regression_based_cls_scoring": "neg_mean_absolute_error",
        "binary_scoring": "balanced_accuracy" if method == "finetune" else "roc_auc",
        "multiclass_scoring": "balanced_accuracy",
    }
    if method in ("mlp", "finetune"):
        epochs = 25 if method == "mlp" else 15
        config.update(
            {
                "selection": "holdout",
                "val_frac": 0.2,
                "max_epochs": epochs,
                "amp": True,
                "eval_batch_size": 4096,
                "num_workers": 0,
                "space": copy.deepcopy(
                    STANDARD_MLP_PARAM_GRID
                    if method == "mlp"
                    else STANDARD_FINETUNE_PARAM_GRID
                ),
                "search": {
                    "method": "optuna_hyperband",
                    "sampler": "tpe",
                    "n_trials": 48,
                    "n_startup_trials": 8,
                    "min_resource": 1,
                    "max_resource": epochs,
                    "reduction_factor": 3,
                    "report_score": "best_so_far",
                    "multivariate": True,
                    "group": True,
                    "constant_liar": True,
                    "load_if_exists": False,
                    "trial_workers_per_gpu": 8 if method == "finetune" else 1,
                },
            }
        )
    return config
