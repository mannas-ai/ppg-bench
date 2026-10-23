"""PPG-Bench published evaluation implementation."""

import random
import threading
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from .classical import (
    score_linear_classical_probe,
    score_linear_classical_subject_probe,
    selection_score,
)
from .datasets import BenchmarkDataset
from .models import ModelPort
from .protocol import (
    STANDARD_FINETUNE_PARAM_GRID,
    STANDARD_MLP_PARAM_GRID,
    subject_aggregation_grid,
)
from .search import (
    make_optuna_study,
    optuna_results,
    optuna_storage_path,
    report_value,
    sample_list_space,
)


class FeatureStandardizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x: np.ndarray):
        self.mean = x.mean(axis=0, keepdims=True)
        self.std = x.std(axis=0, keepdims=True)
        self.std[self.std == 0.0] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise ValueError("FeatureStandardizer must be fit before transform")
        return ((x - self.mean) / self.std).astype(np.float32)


class MLPHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        dropout: float,
        activation: str,
    ):
        super().__init__()
        act = nn.GELU()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            act,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class MLPProbe:
    def __init__(
        self,
        model: MLPHead,
        standardizer: FeatureStandardizer,
        task_kind: str,
        best_params: Mapping[str, Any],
        cv_results: list[dict[str, Any]],
        classes: np.ndarray | None,
        device: str,
    ):
        self.model = model
        self.standardizer = standardizer
        self.task_kind = task_kind
        self.best_params_ = dict(best_params)
        self.cv_results_ = cv_results
        self.classes_ = None if classes is None else np.asarray(classes)
        self.device = device

    def _raw_output(self, x: np.ndarray) -> np.ndarray:
        x_std = self.standardizer.transform(x)
        self.model.eval()
        outs = []
        with torch.inference_mode():
            for start in range(0, len(x_std), 4096):
                batch = torch.from_numpy(x_std[start : start + 4096]).to(self.device)
                outs.append(self.model(batch).detach().cpu().numpy())
        return np.concatenate(outs, axis=0)

    def predict(self, x: np.ndarray):
        out = self._raw_output(x)
        if self.task_kind == "regression":
            return out.reshape(-1) if out.shape[1] == 1 else out
        if self.task_kind == "regression_based_cls":
            return out
        if self.task_kind == "binary_classification":
            prob_pos = 1.0 / (1.0 + np.exp(-out.reshape(-1)))
            return self.classes_[(prob_pos >= 0.5).astype(int)]
        probs = self.predict_proba(x)
        return self.classes_[np.argmax(probs, axis=1)]

    def predict_proba(self, x: np.ndarray):
        out = self._raw_output(x)
        if self.task_kind == "binary_classification":
            prob_pos = 1.0 / (1.0 + np.exp(-out.reshape(-1)))
            return np.stack([1.0 - prob_pos, prob_pos], axis=1)
        if self.task_kind == "multiclass_classification":
            out = out - np.max(out, axis=1, keepdims=True)
            exp = np.exp(out)
            return exp / np.sum(exp, axis=1, keepdims=True)
        raise AttributeError(
            "predict_proba is only available for classification MLP probes"
        )


def task_kind_from_targets(
    y: np.ndarray, task_type: str
) -> tuple[str, np.ndarray | None, int]:
    task_type = task_type.replace("-", "_")
    if task_type == "regression":
        y = np.asarray(y)
        output_dim = 1 if y.ndim == 1 else y.shape[1]
        return ("regression", None, output_dim)
    if task_type == "regression_based_cls":
        if y.ndim != 2:
            raise ValueError("regression-based-cls targets must be 2D")
        return ("regression_based_cls", None, y.shape[1])
    classes = np.unique(y.reshape(-1))
    if len(classes) == 2:
        return ("binary_classification", classes, 1)
    return ("multiclass_classification", classes, len(classes))


def subject_holdout_split(
    subjects: np.ndarray, val_frac: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    unique_subjects = list(dict.fromkeys(np.asarray(subjects).tolist()))
    if not 0.0 < val_frac < 1.0:
        raise ValueError("holdout selection requires 0 < val_frac < 1")
    random.Random(seed).shuffle(unique_subjects)
    n_val = max(1, int(round(len(unique_subjects) * val_frac)))
    if n_val >= len(unique_subjects):
        raise ValueError("holdout val split leaves no train subjects")
    val_subjects = set(unique_subjects[:n_val])
    val_idx = np.asarray([subject in val_subjects for subject in subjects])
    train_idx = ~val_idx
    return (np.where(train_idx)[0], np.where(val_idx)[0])


def make_targets(y: np.ndarray, task_kind: str, classes: np.ndarray | None):
    y = np.asarray(y)
    if task_kind == "regression":
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        return torch.from_numpy(y.astype(np.float32))
    if task_kind == "regression_based_cls":
        return torch.from_numpy(y.astype(np.float32))
    if task_kind == "binary_classification":
        mapped = (y.reshape(-1) == classes[1]).astype(np.float32).reshape(-1, 1)
        return torch.from_numpy(mapped)
    class_to_idx = {label: idx for idx, label in enumerate(classes.tolist())}
    mapped = np.asarray([class_to_idx[label] for label in y.reshape(-1)])
    return torch.from_numpy(mapped.astype(np.int64))


def labels_for_class_weights(
    y: np.ndarray,
    task_kind: str,
    classes: np.ndarray | None,
    subjects: np.ndarray | None = None,
) -> np.ndarray | None:
    if task_kind not in ("binary_classification", "multiclass_classification"):
        return None
    y = np.asarray(y).reshape(-1)
    if subjects is not None:
        subjects = np.asarray(subjects)
        labels = []
        for subject in dict.fromkeys(subjects.tolist()):
            subject_y = y[subjects == subject]
            unique_y = np.unique(subject_y)
            if len(unique_y) != 1:
                raise ValueError(
                    f"Subject {subject} has mixed class labels: {unique_y.tolist()}"
                )
            labels.append(unique_y[0])
        y = np.asarray(labels)
    if task_kind == "binary_classification":
        return (y == classes[1]).astype(np.int64)
    class_to_idx = {label: idx for idx, label in enumerate(classes.tolist())}
    return np.asarray([class_to_idx[label] for label in y], dtype=np.int64)


def balanced_class_weight(
    y: np.ndarray,
    task_kind: str,
    classes: np.ndarray | None,
    device: str,
    subjects: np.ndarray | None = None,
) -> torch.Tensor | None:
    labels = labels_for_class_weights(y, task_kind, classes, subjects)
    if labels is None:
        return None
    n_classes = 2 if task_kind == "binary_classification" else len(classes)
    counts = np.bincount(labels, minlength=n_classes).astype(np.float32)
    if np.any(counts == 0):
        raise ValueError(
            f"Cannot compute balanced class weights with missing classes: {counts.tolist()}"
        )
    if task_kind == "binary_classification":
        return torch.tensor([counts[0] / counts[1]], dtype=torch.float32, device=device)
    weights = counts.sum() / (n_classes * counts)
    return torch.from_numpy(weights.astype(np.float32)).to(device)


def loss_for_task(task_kind: str, class_weight: torch.Tensor | None = None):
    if task_kind in ("regression", "regression_based_cls"):
        return nn.MSELoss()
    if task_kind == "binary_classification":
        return nn.BCEWithLogitsLoss(pos_weight=class_weight)
    return nn.CrossEntropyLoss(weight=class_weight)


def init_mlp_model(
    input_dim: int,
    output_dim: int,
    params: Mapping[str, Any],
    random_state: int,
    device: str,
) -> MLPHead:
    torch.manual_seed(random_state)
    return MLPHead(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=int(params["hidden_dim"]),
        dropout=float(params["dropout"]),
        activation=str(params["activation"]),
    ).to(device)


def scoring_name(task_type: str, task_kind: str, probe_cfg: Mapping[str, Any]) -> str:
    task_type = task_type.replace("-", "_")
    if task_type == "regression":
        return probe_cfg.get("regression_scoring", "neg_mean_absolute_error")
    if task_type == "regression_based_cls":
        return probe_cfg.get("regression_based_cls_scoring", "neg_mean_absolute_error")
    if task_kind == "binary_classification":
        return probe_cfg.get("binary_scoring", "roc_auc")
    return probe_cfg.get("multiclass_scoring", "balanced_accuracy")


def score_mlp_candidate(
    model: MLPHead,
    standardizer: FeatureStandardizer,
    x_heldout: np.ndarray,
    y_heldout: np.ndarray,
    subjects_heldout: np.ndarray,
    task_type: str,
    task_kind: str,
    classes: np.ndarray | None,
    device: str,
    eval_level: str,
    aggregation: Mapping[str, Any] | None,
):
    probe = MLPProbe(model, standardizer, task_kind, {}, [], classes, device)
    if eval_level == "subject":
        return score_linear_classical_subject_probe(
            probe, x_heldout, y_heldout, subjects_heldout, task_type, aggregation
        )
    return score_linear_classical_probe(probe, x_heldout, y_heldout, task_type)


def fit_mlp_probe_optuna(
    x_train: np.ndarray,
    y_train: np.ndarray,
    task_type: str,
    probe_cfg: Mapping[str, Any],
    subjects_train: np.ndarray,
    eval_level: str,
    aggregation_grid: list[dict[str, Any]] | None,
    device: str,
):
    import optuna

    y_train = np.asarray(y_train)
    subjects_train = np.asarray(subjects_train)
    task_kind, classes, output_dim = task_kind_from_targets(y_train, task_type)
    random_state = int(probe_cfg.get("random_state", 0))
    scoring = scoring_name(task_type, task_kind, probe_cfg)
    max_epochs = int(probe_cfg["max_epochs"])
    report_mode = str(probe_cfg.get("search", {}).get("report_score", "best_so_far"))
    train_idx, val_idx = subject_holdout_split(
        subjects_train, float(probe_cfg.get("val_frac", 0.2)), random_state
    )
    aggregations = aggregation_grid if eval_level == "subject" else [None]
    if eval_level == "subject" and (not aggregations):
        raise ValueError("Subject-level MLP probe requires aggregation candidates")
    study = make_optuna_study(
        probe_cfg,
        study_name=str(
            probe_cfg.get("study_name", f"mlp__{task_type}__seed{random_state}")
        ),
        seed=random_state,
    )
    best_candidate = None
    best_state = None
    best_standardizer = None
    best_score = -np.inf

    def objective(trial):
        nonlocal best_candidate, best_state, best_standardizer, best_score
        params = sample_list_space(trial, probe_cfg, STANDARD_MLP_PARAM_GRID)
        aggregation = None
        aggregation_idx = None
        if eval_level == "subject":
            aggregation_idx = trial.suggest_categorical(
                "aggregation_idx", list(range(len(aggregations)))
            )
            aggregation = aggregations[int(aggregation_idx)]
        standardizer = FeatureStandardizer().fit(x_train[train_idx])
        x_fold_train = standardizer.transform(x_train[train_idx])
        weight_subjects = subjects_train[train_idx] if eval_level == "subject" else None
        class_weight = balanced_class_weight(
            y_train[train_idx], task_kind, classes, device, weight_subjects
        )
        model = init_mlp_model(
            x_fold_train.shape[1],
            output_dim,
            params,
            random_state=random_state + trial.number,
            device=device,
        )
        criterion = loss_for_task(task_kind, class_weight)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(params["lr"]),
            weight_decay=float(params["weight_decay"]),
        )
        x_tensor = torch.from_numpy(x_fold_train.astype(np.float32))
        y_tensor = make_targets(y_train[train_idx], task_kind, classes)
        generator = torch.Generator()
        generator.manual_seed(random_state + trial.number)
        loader = DataLoader(
            TensorDataset(x_tensor, y_tensor),
            batch_size=int(params["batch_size"]),
            shuffle=True,
            generator=generator,
        )
        trial_best_score = -np.inf
        trial_best_epoch = 0
        trial_best_state = None
        epoch_scores = []
        train_losses = []
        for epoch_idx in range(max_epochs):
            model.train()
            losses = []
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            result = score_mlp_candidate(
                model,
                standardizer,
                x_train[val_idx],
                y_train[val_idx],
                subjects_train[val_idx],
                task_type,
                task_kind,
                classes,
                device,
                eval_level,
                aggregation,
            )
            score = selection_score(result, scoring)
            if score is None or not np.isfinite(score):
                raise ValueError("Optuna MLP validation score was not finite")
            score = float(score)
            epoch_scores.append(score)
            train_losses.append(float(np.mean(losses)) if losses else float("nan"))
            if score > trial_best_score:
                trial_best_score = score
                trial_best_epoch = epoch_idx + 1
                trial_best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            trial.set_user_attr("best_score", float(trial_best_score))
            trial.set_user_attr("best_epoch", int(trial_best_epoch))
            trial.set_user_attr("score_by_epoch", list(epoch_scores))
            trial.set_user_attr("train_losses", list(train_losses))
            trial.report(
                report_value(score, trial_best_score, report_mode), step=epoch_idx + 1
            )
            if trial.should_prune():
                raise optuna.TrialPruned()
        candidate = {
            "params": dict(params),
            "aggregation": None if aggregation is None else dict(aggregation),
            "aggregation_idx": aggregation_idx,
            "selected_epoch": int(trial_best_epoch),
            "mean_score": float(trial_best_score),
            "score_by_epoch": epoch_scores,
            "train_losses": train_losses,
            "val_subjects": sorted(set(subjects_train[val_idx].tolist())),
            "trial_number": int(trial.number),
        }
        if trial_best_score > best_score:
            best_score = trial_best_score
            best_candidate = candidate
            best_state = trial_best_state
            best_standardizer = standardizer
        return float(trial_best_score)

    study.optimize(
        objective,
        n_trials=int(probe_cfg.get("search", {}).get("n_trials", 48)),
        catch=(),
    )
    if best_candidate is None or best_state is None or best_standardizer is None:
        raise ValueError("No completed Optuna MLP trial was found")
    final_model = init_mlp_model(
        x_train.shape[1],
        output_dim,
        best_candidate["params"],
        random_state=random_state,
        device=device,
    )
    final_model.load_state_dict(best_state)
    best_params = dict(best_candidate["params"])
    best_params["selected_epoch"] = int(best_candidate["selected_epoch"])
    best_params["selection_score"] = float(best_candidate["mean_score"])
    best_params["selection"] = "holdout"
    best_params["val_frac"] = float(probe_cfg.get("val_frac", 0.2))
    best_params["search_method"] = "optuna_hyperband"
    if best_candidate["aggregation"] is not None:
        best_params["aggregation"] = best_candidate["aggregation"]
    probe = MLPProbe(
        final_model, best_standardizer, task_kind, best_params, [], classes, device
    )
    probe.optuna_results_ = optuna_results(
        study, probe_cfg, best_candidate, STANDARD_MLP_PARAM_GRID
    )
    return probe


class FinetuneModel(nn.Module):
    def __init__(
        self, encoder: nn.Module, head: nn.Module, model_cfg: Mapping[str, Any]
    ):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.pooling = model_cfg.get("pooling")
        self.embedding_dim = int(model_cfg["embedding_dim"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self.encoder(x)
        if not isinstance(embeddings, torch.Tensor):
            raise TypeError(
                f"Encoder must return a tensor, got {type(embeddings).__name__}"
            )
        pooled = pool_embeddings(embeddings, self.pooling, self.embedding_dim)
        return self.head(pooled)


class PredictionProbe:
    def __init__(
        self,
        y_pred: np.ndarray,
        y_prob: np.ndarray | None,
        best_params: Mapping[str, Any],
        classes: np.ndarray | None,
    ):
        self.y_pred = y_pred
        self.y_prob = y_prob
        self.best_params_ = dict(best_params)
        self.classes_ = None if classes is None else np.asarray(classes)

    def predict(self, x):
        return self.y_pred

    def predict_proba(self, x):
        if self.y_prob is None:
            raise AttributeError("This prediction probe does not have probabilities")
        return self.y_prob


def dataset_targets_subjects(
    dataset: BenchmarkDataset,
) -> tuple[np.ndarray, np.ndarray]:
    if not hasattr(dataset, "y") or not hasattr(dataset, "subjects"):
        raise ValueError(
            "Finetuning requires datasets to expose y and subjects attributes"
        )
    y = np.asarray(dataset.y)
    subjects = np.asarray(dataset.subjects)
    if len(y) != len(dataset) or len(subjects) != len(dataset):
        raise ValueError("Dataset y and subjects must match dataset length")
    return (y, subjects)


def make_loader(
    dataset: BenchmarkDataset,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    subset = Subset(dataset, np.asarray(indices).astype(int).tolist())
    return DataLoader(
        subset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=int(num_workers),
        generator=generator if shuffle else None,
    )


def subjects_from_batch(batch: Mapping[str, Any]) -> list[Any]:
    subjects = batch.get("subject")
    if subjects is None:
        return []
    if isinstance(subjects, torch.Tensor):
        return subjects.detach().cpu().numpy().tolist()
    if isinstance(subjects, np.ndarray):
        return subjects.tolist()
    if isinstance(subjects, str):
        return [subjects]
    return list(subjects)


def batch_targets(
    y, task_kind: str, classes: np.ndarray | None, device: str
) -> torch.Tensor:
    if isinstance(y, torch.Tensor):
        y_np = y.detach().cpu().numpy()
    else:
        y_np = np.asarray(y)
    return make_targets(y_np, task_kind, classes).to(device)


def build_finetune_model(
    model_port: ModelPort,
    model_cfg: Mapping[str, Any],
    head_input_dim: int,
    output_dim: int,
    params: Mapping[str, Any],
    device: str,
) -> FinetuneModel:
    if "embedding_dim" not in model_cfg:
        raise ValueError("model.embedding_dim must be specified")
    encoder = model_port.load_encoder()
    parameters = list(encoder.parameters())
    if not parameters:
        raise ValueError("Finetuning requires an encoder with trainable parameters")
    for parameter in parameters:
        parameter.requires_grad_(True)
    head = MLPHead(
        input_dim=int(head_input_dim),
        output_dim=output_dim,
        hidden_dim=int(params["hidden_dim"]),
        dropout=float(params["head_dropout"]),
        activation=str(params["activation"]),
    )
    return FinetuneModel(encoder, head, model_cfg).to(device)


def infer_head_input_dim(
    model_port: ModelPort,
    dataset: BenchmarkDataset,
    model_cfg: Mapping[str, Any],
    device: str,
) -> int:
    encoder = model_port.load_encoder().to(device)
    encoder.eval()
    loader = make_loader(
        dataset, np.asarray([0]), batch_size=1, shuffle=False, num_workers=0, seed=0
    )
    with torch.inference_mode():
        batch = next(iter(loader))
        batch = dataset.prepare_batch(batch)
        x = model_port.prepare_batch(batch, model_cfg, device)
        embeddings = encoder(x)
        if not isinstance(embeddings, torch.Tensor):
            raise TypeError(
                f"Encoder must return a tensor, got {type(embeddings).__name__}"
            )
        pooled = pool_embeddings(
            embeddings, model_cfg.get("pooling"), int(model_cfg["embedding_dim"])
        )
        if pooled.ndim != 2:
            raise ValueError(f"Pooled embeddings must be 2D, got {tuple(pooled.shape)}")
    return int(pooled.shape[1])


def apply_train_mode(model: FinetuneModel, batchnorm_mode: str):
    model.train()
    if batchnorm_mode == "train":
        return
    if batchnorm_mode != "freeze_stats":
        raise ValueError(f"Unsupported batchnorm_mode: {batchnorm_mode}")
    for module in model.encoder.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.eval()


def make_optimizer(
    model: FinetuneModel, params: Mapping[str, Any]
) -> torch.optim.Optimizer:
    head_lr = float(params["head_lr"])
    encoder_lr = head_lr * float(params["encoder_lr_ratio"])
    head_wd = float(params["head_weight_decay"])
    encoder_wd = head_wd * float(params["encoder_weight_decay_ratio"])
    return torch.optim.AdamW(
        [
            {
                "params": model.encoder.parameters(),
                "lr": encoder_lr,
                "weight_decay": encoder_wd,
            },
            {"params": model.head.parameters(), "lr": head_lr, "weight_decay": head_wd},
        ]
    )


def train_one_epoch(
    model: FinetuneModel,
    model_port: ModelPort,
    dataset: BenchmarkDataset,
    loader: DataLoader,
    model_cfg: Mapping[str, Any],
    task_kind: str,
    classes: np.ndarray | None,
    criterion,
    optimizer,
    device: str,
    amp: bool,
    scaler,
    batchnorm_mode: str,
) -> float:
    apply_train_mode(model, batchnorm_mode)
    losses = []
    use_amp = amp and str(device).startswith("cuda")
    for batch in loader:
        batch = dataset.prepare_batch(batch)
        x = model_port.prepare_batch(batch, model_cfg, device)
        target = batch_targets(batch["y"], task_kind, classes, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            pred = model(x)
            loss = criterion(pred, target)
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if not any(
            parameter.grad is not None for parameter in model.encoder.parameters()
        ):
            raise ValueError(
                "Finetuning requires embeddings with gradients to the encoder"
            )
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def evaluate_finetune_model(
    model: FinetuneModel,
    model_port: ModelPort,
    dataset: BenchmarkDataset,
    loader: DataLoader,
    model_cfg: Mapping[str, Any],
    task_kind: str,
    classes: np.ndarray | None,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    model.eval()
    y_all = []
    subjects_all = []
    pred_all = []
    prob_all = []
    with torch.inference_mode():
        for batch in loader:
            batch = dataset.prepare_batch(batch)
            x = model_port.prepare_batch(batch, model_cfg, device)
            out = model(x).detach().cpu().numpy()
            y = batch["y"]
            if isinstance(y, torch.Tensor):
                y_all.append(y.detach().cpu().numpy())
            else:
                y_all.append(np.asarray(y))
            subjects_all.extend(subjects_from_batch(batch))
            if task_kind in ("regression", "regression_based_cls"):
                pred_all.append(
                    out.reshape(-1)
                    if task_kind == "regression" and out.shape[1] == 1
                    else out
                )
            elif task_kind == "binary_classification":
                prob_pos = 1.0 / (1.0 + np.exp(-out.reshape(-1)))
                prob = np.stack([1.0 - prob_pos, prob_pos], axis=1)
                prob_all.append(prob)
                pred_all.append(classes[(prob_pos >= 0.5).astype(int)])
            else:
                out = out - np.max(out, axis=1, keepdims=True)
                exp = np.exp(out)
                prob = exp / np.sum(exp, axis=1, keepdims=True)
                prob_all.append(prob)
                pred_all.append(classes[np.argmax(prob, axis=1)])
    y_true = np.concatenate(y_all, axis=0)
    subjects = np.asarray(subjects_all)
    y_pred = np.concatenate(pred_all, axis=0)
    y_prob = np.concatenate(prob_all, axis=0) if prob_all else None
    return (y_true, subjects, y_pred, y_prob)


def score_finetune_outputs(
    y_true: np.ndarray,
    subjects: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    classes: np.ndarray | None,
    task_type: str,
    eval_level: str,
    aggregation: Mapping[str, Any] | None,
    best_params: Mapping[str, Any],
):
    probe = PredictionProbe(y_pred, y_prob, best_params, classes)
    x_dummy = np.zeros((len(y_true), 1), dtype=np.float32)
    if eval_level == "subject":
        return score_linear_classical_subject_probe(
            probe, x_dummy, y_true, subjects, task_type, aggregation
        )
    return score_linear_classical_probe(probe, x_dummy, y_true, task_type)


def evaluate_and_score(
    model: FinetuneModel,
    model_port: ModelPort,
    dataset: BenchmarkDataset,
    indices: np.ndarray,
    model_cfg: Mapping[str, Any],
    probe_cfg: Mapping[str, Any],
    task_type: str,
    eval_level: str,
    task_kind: str,
    classes: np.ndarray | None,
    aggregation: Mapping[str, Any] | None,
    best_params: Mapping[str, Any],
    device: str,
) -> dict[str, Any]:
    loader = make_loader(
        dataset,
        indices,
        batch_size=int(
            probe_cfg.get("eval_batch_size", probe_cfg.get("batch_size", 256))
        ),
        shuffle=False,
        num_workers=int(probe_cfg.get("num_workers", 0)),
        seed=int(probe_cfg.get("random_state", 0)),
    )
    y_true, subjects, y_pred, y_prob = evaluate_finetune_model(
        model, model_port, dataset, loader, model_cfg, task_kind, classes, device
    )
    return score_finetune_outputs(
        y_true,
        subjects,
        y_pred,
        y_prob,
        classes,
        task_type,
        eval_level,
        aggregation,
        best_params,
    )


def fit_finetune_holdout_optuna(
    model_port: ModelPort,
    train_dataset: BenchmarkDataset,
    y_train: np.ndarray,
    subjects_train: np.ndarray,
    model_cfg: Mapping[str, Any],
    probe_cfg: Mapping[str, Any],
    task_type: str,
    eval_level: str,
    task_kind: str,
    classes: np.ndarray | None,
    output_dim: int,
    head_input_dim: int,
    aggregations: list[dict[str, Any] | None],
    scoring: str,
    device: str,
):
    import optuna

    random_state = int(probe_cfg.get("random_state", 0))
    max_epochs = int(probe_cfg["max_epochs"])
    report_mode = str(probe_cfg.get("search", {}).get("report_score", "best_so_far"))
    train_idx, val_idx = subject_holdout_split(
        subjects_train, float(probe_cfg.get("val_frac", 0.2)), random_state
    )
    study_name = str(
        probe_cfg.get("study_name", f"finetune__{task_type}__seed{random_state}")
    )
    storage_path = optuna_storage_path(dict(probe_cfg.get("search", {})), study_name)
    if isinstance(probe_cfg, dict) and storage_path is not None:
        probe_cfg["storage_path"] = str(storage_path)
    study = make_optuna_study(probe_cfg, study_name=study_name, seed=random_state)
    best_candidate = None
    best_state = None
    best_score = -np.inf
    best_lock = threading.Lock()

    def objective(trial):
        nonlocal best_candidate, best_state, best_score
        params = sample_list_space(trial, probe_cfg, STANDARD_FINETUNE_PARAM_GRID)
        aggregation = None
        aggregation_idx = None
        if eval_level == "subject":
            aggregation_idx = trial.suggest_categorical(
                "aggregation_idx", list(range(len(aggregations)))
            )
            aggregation = aggregations[int(aggregation_idx)]
        weight_subjects = subjects_train[train_idx] if eval_level == "subject" else None
        class_weight = balanced_class_weight(
            y_train[train_idx], task_kind, classes, device, weight_subjects
        )
        model = build_finetune_model(
            model_port, model_cfg, head_input_dim, output_dim, params, device
        )
        loader = make_loader(
            train_dataset,
            train_idx,
            batch_size=int(params["batch_size"]),
            shuffle=True,
            num_workers=int(probe_cfg.get("num_workers", 0)),
            seed=random_state + trial.number,
        )
        criterion = loss_for_task(task_kind, class_weight)
        optimizer = make_optimizer(model, params)
        use_amp = bool(probe_cfg.get("amp", False)) and str(device).startswith("cuda")
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        trial_best_score = -np.inf
        trial_best_epoch = 0
        trial_best_state = None
        epoch_scores = []
        epoch_results = []
        train_losses = []
        for epoch_idx in range(max_epochs):
            loss = train_one_epoch(
                model,
                model_port,
                train_dataset,
                loader,
                model_cfg,
                task_kind,
                classes,
                criterion,
                optimizer,
                device,
                bool(probe_cfg.get("amp", False)),
                scaler,
                str(params["batchnorm_mode"]),
            )
            result = evaluate_and_score(
                model,
                model_port,
                train_dataset,
                val_idx,
                model_cfg,
                probe_cfg,
                task_type,
                eval_level,
                task_kind,
                classes,
                aggregation,
                {},
                device,
            )
            score = selection_score(result, scoring)
            if score is None or not np.isfinite(score):
                raise ValueError("Optuna finetune validation score was not finite")
            score = float(score)
            epoch_scores.append(score)
            epoch_results.append(result)
            train_losses.append(loss)
            if score > trial_best_score:
                trial_best_score = score
                trial_best_epoch = epoch_idx + 1
                trial_best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            trial.set_user_attr("best_score", float(trial_best_score))
            trial.set_user_attr("best_epoch", int(trial_best_epoch))
            trial.set_user_attr("score_by_epoch", list(epoch_scores))
            trial.set_user_attr("train_losses", list(train_losses))
            trial.report(
                report_value(score, trial_best_score, report_mode), step=epoch_idx + 1
            )
            if trial.should_prune():
                raise optuna.TrialPruned()
        candidate = {
            "params": dict(params),
            "aggregation": None if aggregation is None else dict(aggregation),
            "aggregation_idx": aggregation_idx,
            "selected_epoch": int(trial_best_epoch),
            "mean_score": float(trial_best_score),
            "score_by_epoch": epoch_scores,
            "epoch_results": epoch_results,
            "train_losses": train_losses,
            "val_subjects": sorted(set(subjects_train[val_idx].tolist())),
            "trial_number": int(trial.number),
        }
        with best_lock:
            if trial_best_score > best_score:
                best_score = trial_best_score
                best_candidate = candidate
                best_state = trial_best_state
        return float(trial_best_score)

    search_cfg = probe_cfg.get("search", {})
    trial_workers = int(
        search_cfg.get("trial_workers_per_gpu", search_cfg.get("n_jobs", 1))
    )
    if trial_workers < 1:
        raise ValueError("search.trial_workers_per_gpu must be at least 1")
    study.optimize(
        objective,
        n_trials=int(probe_cfg.get("search", {}).get("n_trials", 48)),
        n_jobs=trial_workers,
        gc_after_trial=True,
        catch=(),
    )
    if best_candidate is None or best_state is None:
        raise ValueError("No completed Optuna finetune trial was found")
    final_model = build_finetune_model(
        model_port,
        model_cfg,
        head_input_dim,
        output_dim,
        best_candidate["params"],
        device,
    )
    final_model.load_state_dict(best_state)
    return (
        final_model,
        best_candidate,
        [],
        optuna_results(study, probe_cfg, best_candidate, STANDARD_FINETUNE_PARAM_GRID),
    )


def run_finetune_probe(
    model_port: ModelPort,
    train_dataset: BenchmarkDataset,
    test_dataset: BenchmarkDataset,
    model_cfg: Mapping[str, Any],
    probe_cfg: Mapping[str, Any],
    task_type: str,
    eval_level: str,
    device: str,
) -> dict[str, Any]:
    y_train, subjects_train = dataset_targets_subjects(train_dataset)
    task_kind, classes, output_dim = task_kind_from_targets(y_train, task_type)
    head_input_dim = infer_head_input_dim(model_port, train_dataset, model_cfg, device)
    scoring = scoring_name(task_type, task_kind, probe_cfg)
    aggregations = subject_aggregation_grid(task_type, eval_level, probe_cfg)
    aggregation_candidates = aggregations if eval_level == "subject" else [None]
    if eval_level == "subject" and (not aggregation_candidates):
        raise ValueError("Subject-level finetune probe requires aggregation candidates")
    model, best_candidate, cv_results, optuna_payload = fit_finetune_holdout_optuna(
        model_port,
        train_dataset,
        y_train,
        subjects_train,
        model_cfg,
        probe_cfg,
        task_type,
        eval_level,
        task_kind,
        classes,
        output_dim,
        head_input_dim,
        aggregation_candidates,
        scoring,
        device,
    )
    best_params = dict(best_candidate["params"])
    best_params["selected_epoch"] = int(best_candidate["selected_epoch"])
    best_params["selection_score"] = float(best_candidate["mean_score"])
    best_params["selection"] = "holdout"
    best_params["search_method"] = "optuna_hyperband"
    best_params["holdout_score"] = float(best_candidate["mean_score"])
    best_params["val_frac"] = float(probe_cfg.get("val_frac", 0.2))
    if best_candidate["aggregation"] is not None:
        best_params["aggregation"] = best_candidate["aggregation"]
    test_indices = np.arange(len(test_dataset))
    selected_aggregation = (
        best_candidate["aggregation"] if eval_level == "subject" else None
    )
    test_result = evaluate_and_score(
        model,
        model_port,
        test_dataset,
        test_indices,
        model_cfg,
        probe_cfg,
        task_type,
        eval_level,
        task_kind,
        classes,
        selected_aggregation,
        best_params,
        device,
    )
    loader = make_loader(
        test_dataset,
        test_indices,
        batch_size=int(probe_cfg.get("eval_batch_size", 4096)),
        shuffle=False,
        num_workers=0,
        seed=int(probe_cfg["random_state"]),
    )
    y_true, subjects, y_pred, y_prob = evaluate_finetune_model(
        model, model_port, test_dataset, loader, model_cfg, task_kind, classes, device
    )
    prediction_record = {"y_true": y_true, "subjects": subjects, "y_pred": y_pred}
    if y_prob is not None:
        prediction_record.update(y_prob=y_prob, classes=classes)
    result = {
        "test": test_result,
        "inner_cv_results": cv_results,
        "test_predictions": prediction_record,
        "optuna_results": optuna_payload,
    }
    if eval_level == "subject":
        result["selected_subject_aggregation"] = selected_aggregation
    return result


def pool_embeddings(x, pooling, embedding_dim):
    if x.ndim != 2 or x.shape[1] != embedding_dim:
        raise ValueError(
            f"Encoder must return (batch, {embedding_dim}); got {tuple(x.shape)}"
        )
    return x
