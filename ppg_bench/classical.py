"""PPG-Bench published evaluation implementation."""

import random
from itertools import product
from math import ceil
from typing import Any, Mapping

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .protocol import (
    STANDARD_LINEAR_CLS_PARAM_GRID,
    STANDARD_LINEAR_REG_PARAM_GRID,
    STANDARD_RF_CLS_PARAM_GRID,
    STANDARD_RF_REG_PARAM_GRID,
)


class FittedProbe:
    def __init__(
        self,
        estimator,
        best_params: Mapping[str, Any],
        cv_results: list[dict[str, Any]],
    ):
        self.best_estimator_ = estimator
        self.best_params_ = dict(best_params)
        self.cv_results_ = cv_results

    def predict(self, x):
        return self.best_estimator_.predict(x)

    def predict_proba(self, x):
        return self.best_estimator_.predict_proba(x)

    @property
    def classes_(self):
        if hasattr(self.best_estimator_, "classes_"):
            return self.best_estimator_.classes_
        if (
            hasattr(self.best_estimator_, "named_steps")
            and "clf" in self.best_estimator_.named_steps
        ):
            return self.best_estimator_.named_steps["clf"].classes_
        raise AttributeError("Wrapped estimator does not expose classes_")


def expand_param_grid(param_grid) -> list[dict[str, Any]]:
    if param_grid is None:
        return [{}]
    if isinstance(param_grid, list):
        expanded = []
        for grid in param_grid:
            expanded.extend(expand_param_grid(grid))
        return expanded
    keys = list(param_grid)
    values = [
        value if isinstance(value, list) else [value] for value in param_grid.values()
    ]
    return [dict(zip(keys, combo)) for combo in product(*values)]


def make_subject_folds(subjects: np.ndarray, n_folds: int, seed: int):
    subjects = np.asarray(subjects)
    unique_subjects = list(dict.fromkeys(subjects.tolist()))
    if n_folds < 2:
        raise ValueError("inner_folds must be at least 2")
    if n_folds > len(unique_subjects):
        raise ValueError(
            f"inner_folds={n_folds} exceeds n_train_subjects={len(unique_subjects)}"
        )
    random.Random(seed).shuffle(unique_subjects)
    base_size = len(unique_subjects) // n_folds
    extra = len(unique_subjects) % n_folds
    fold_sizes = [base_size + int(i < extra) for i in range(n_folds)]
    folds = []
    cursor = 0
    for fold_size in fold_sizes:
        heldout_subjects = set(unique_subjects[cursor : cursor + fold_size])
        cursor += fold_size
        heldout_idx = np.asarray([subject in heldout_subjects for subject in subjects])
        train_idx = ~heldout_idx
        folds.append((train_idx, heldout_idx, sorted(heldout_subjects)))
    return folds


def selection_score(result: Mapping[str, Any], scoring: str) -> float | None:
    if scoring == "neg_mean_absolute_error":
        value = result.get("mae")
        value = None if value is None else -float(value)
    elif scoring == "neg_mean_squared_error":
        value = result.get("mse")
        value = None if value is None else -float(value)
    elif scoring == "roc_auc":
        value = result.get("auroc")
    elif scoring == "roc_auc_ovr":
        value = result.get("auroc_ovr")
    elif scoring == "balanced_accuracy":
        value = result.get("balanced_accuracy")
    elif scoring == "accuracy":
        value = result.get("accuracy")
    elif scoring == "f1_macro":
        value = result.get("f1_macro")
    elif scoring == "argmax_balanced_accuracy":
        value = result.get("argmax_balanced_accuracy")
    else:
        raise ValueError(f"Unsupported selection scoring: {scoring}")
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value):
        return None
    return value


def fit_grouped_grid_search(
    estimator,
    param_grid,
    x_train: np.ndarray,
    y_train: np.ndarray,
    subjects_train: np.ndarray,
    task_type: str,
    eval_level: str,
    scoring: str,
    inner_folds: int,
    random_state: int,
    score_fn,
    subject_score_fn,
    aggregation_grid: list[dict[str, Any]] | None = None,
) -> FittedProbe:
    subjects_train = np.asarray(subjects_train)
    if len(subjects_train) != len(x_train):
        raise ValueError(
            f"subjects_train length must match x_train: {len(subjects_train)} vs {len(x_train)}"
        )
    param_candidates = expand_param_grid(param_grid)
    if eval_level == "subject" and not aggregation_grid:
        raise ValueError("Subject-level grid search requires aggregation candidates")
    aggregations = aggregation_grid if eval_level == "subject" else [None]
    folds = make_subject_folds(subjects_train, inner_folds, random_state)
    cv_results = []
    best_result = None
    best_score = -np.inf
    for params in param_candidates:
        scores_by_aggregation = [[] for _ in aggregations]
        results_by_aggregation = [[] for _ in aggregations]
        for fold_idx, (train_idx, heldout_idx, heldout_subjects) in enumerate(folds):
            fit_error = None
            try:
                fitted = clone(estimator)
                fitted.set_params(**params)
                fitted.fit(x_train[train_idx], y_train[train_idx])
            except Exception as exc:
                fit_error = str(exc)
            for agg_idx, aggregation in enumerate(aggregations):
                result = {"error": fit_error}
                score = None
                if fit_error is None:
                    try:
                        probe = FittedProbe(fitted, params, [])
                        if eval_level == "subject":
                            result = subject_score_fn(
                                probe,
                                x_train[heldout_idx],
                                y_train[heldout_idx],
                                subjects_train[heldout_idx],
                                task_type,
                                aggregation,
                            )
                        else:
                            result = score_fn(
                                probe,
                                x_train[heldout_idx],
                                y_train[heldout_idx],
                                task_type,
                            )
                        score = selection_score(result, scoring)
                    except Exception as exc:
                        result = {"error": str(exc)}
                if score is not None:
                    scores_by_aggregation[agg_idx].append(score)
                results_by_aggregation[agg_idx].append(
                    {
                        "fold_idx": fold_idx,
                        "heldout_subjects": heldout_subjects,
                        "score": score,
                        "result": result,
                    }
                )
        for agg_idx, aggregation in enumerate(aggregations):
            fold_scores = scores_by_aggregation[agg_idx]
            mean_score = float(np.mean(fold_scores)) if fold_scores else -np.inf
            candidate = {
                "params": dict(params),
                "aggregation": None if aggregation is None else dict(aggregation),
                "mean_score": mean_score,
                "n_valid_inner_folds": len(fold_scores),
                "n_total_inner_folds": len(folds),
                "folds": results_by_aggregation[agg_idx],
            }
            cv_results.append(candidate)
            if mean_score > best_score:
                best_score = mean_score
                best_result = candidate
    if best_result is None or not np.isfinite(best_score):
        raise ValueError("No valid hyperparameter candidate was found")
    final_estimator = clone(estimator)
    final_estimator.set_params(**best_result["params"])
    final_estimator.fit(x_train, y_train)
    best_params = dict(best_result["params"])
    if best_result["aggregation"] is not None:
        best_params["aggregation"] = best_result["aggregation"]
    best_params["inner_cv_score"] = best_result["mean_score"]
    best_params["inner_folds"] = int(inner_folds)
    return FittedProbe(final_estimator, best_params, cv_results)


def store_metric(results: dict[str, Any], name: str, compute):
    try:
        value = compute()
    except Exception as exc:
        results[name] = None
        results[f"{name}_error"] = str(exc)
        return
    if isinstance(value, np.generic):
        value = value.item()
    results[name] = value


def prepare_targets(y, task_type, phase="fitting"):
    y = np.asarray(y)
    task_type = task_type.replace("-", "_")
    if task_type == "regression_based_cls":
        if y.ndim != 2 or y.shape[1] < 2:
            raise ValueError(
                "regression-based-cls requires targets shaped (n_samples, n_targets) with n_targets > 1"
            )
        if not np.isfinite(y).all():
            raise ValueError(
                f"regression-based-cls targets must be finite before {phase}"
            )
    else:
        y = y.reshape(-1)
    if task_type == "classification":
        task_type = (
            "binary_classification"
            if len(np.unique(y)) == 2
            else "multiclass_classification"
        )
    return y, task_type


def classical_scoring(task_type, probe_cfg):
    if task_type == "regression_based_cls":
        return probe_cfg.get(
            "regression_based_cls_scoring",
            probe_cfg.get("regression_scoring", "neg_mean_absolute_error"),
        )
    key, default = {
        "regression": ("regression_scoring", "neg_mean_absolute_error"),
        "binary_classification": ("binary_scoring", "roc_auc"),
        "multiclass_classification": ("multiclass_scoring", "balanced_accuracy"),
    }[task_type]
    return probe_cfg.get(key, default)


def fit_linear_classical_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    task_type: str,
    probe_cfg: Mapping[str, Any],
    subjects_train: np.ndarray,
    eval_level: str,
    aggregation_grid: list[dict[str, Any]] | None = None,
):
    y_train, task_type = prepare_targets(y_train, task_type)
    random_state = int(probe_cfg.get("random_state", 0))
    regression = task_type in ("regression", "regression_based_cls")
    expected = {
        "regression": "ridge",
        "regression_based_cls": "ridge",
        "binary_classification": "logistic_regression",
        "multiclass_classification": "multinomial_logistic_regression",
    }
    if task_type not in expected:
        raise ValueError(f"Unsupported task_type for linear_classical: {task_type}")
    if probe_cfg.get(task_type, expected[task_type]) != expected[task_type]:
        label = {
            "regression_based_cls": "regression-based-cls",
            "binary_classification": "binary classification",
            "multiclass_classification": "multiclass",
        }.get(task_type, task_type)
        raise ValueError(
            f"linear_classical {label} currently supports only {expected[task_type]}"
        )
    if regression:
        step = ("reg", Ridge(solver=probe_cfg.get("ridge_solver", "auto")))
        param_grid = STANDARD_LINEAR_REG_PARAM_GRID
    else:
        step = (
            "clf",
            LogisticRegression(
                solver="lbfgs",
                max_iter=int(probe_cfg.get("max_iter", 1000)),
                class_weight=probe_cfg.get("class_weight", "balanced"),
                random_state=random_state,
            ),
        )
        param_grid = STANDARD_LINEAR_CLS_PARAM_GRID
    grid_name = probe_cfg.get("param_grid", "standard")
    if grid_name not in (None, "standard"):
        raise ValueError(f"Unsupported linear_classical param_grid: {grid_name}")
    return fit_grouped_grid_search(
        estimator=Pipeline([("scaler", StandardScaler()), step]),
        scoring=classical_scoring(task_type, probe_cfg),
        param_grid=param_grid,
        x_train=x_train,
        y_train=y_train,
        subjects_train=subjects_train,
        task_type=task_type,
        eval_level=eval_level,
        inner_folds=int(probe_cfg["inner_folds"]),
        random_state=random_state,
        score_fn=score_linear_classical_probe,
        subject_score_fn=score_linear_classical_subject_probe,
        aggregation_grid=aggregation_grid,
    )


def score_linear_classical_subject_probe(
    probe,
    x: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    task_type: str,
    aggregation: Mapping[str, Any],
) -> dict[str, Any]:
    y = np.asarray(y)
    subjects = np.asarray(subjects)
    task_type = task_type.replace("-", "_")
    if len(subjects) != len(x):
        raise ValueError(
            f"Subject count must match embeddings: {len(subjects)} vs {len(x)}"
        )
    is_classification = task_type in (
        "classification",
        "binary_classification",
        "multiclass_classification",
    )
    if task_type == "regression":
        if aggregation.get("name") != "mean":
            raise ValueError(
                "Subject-level regression currently supports only mean aggregation"
            )
        y = y.reshape(-1)
        window_pred = np.asarray(probe.predict(x)).reshape(-1)
        subject_labels = []
        subject_pred = []
        for subject in dict.fromkeys(subjects.tolist()):
            idx = subjects == subject
            subject_y = y[idx]
            first_y = subject_y[0]
            if not np.allclose(subject_y, first_y):
                raise ValueError(f"Subject {subject} has mixed regression labels")
            subject_labels.append(float(first_y))
            subject_pred.append(float(np.mean(window_pred[idx])))
        subject_labels = np.asarray(subject_labels)
        subject_pred = np.asarray(subject_pred)
        mse = mean_squared_error(subject_labels, subject_pred)
        return {
            "best_params": probe.best_params_,
            "aggregation": dict(aggregation),
            "n_subjects": int(len(subject_labels)),
            "n_windows": int(len(y)),
            "mae": mean_absolute_error(subject_labels, subject_pred),
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "r2": r2_score(subject_labels, subject_pred),
        }
    if is_classification:
        y = y.reshape(-1)
        window_scores = probe.predict_proba(x)
        classes = getattr(probe, "classes_", None)
        if classes is None:
            classes = probe.best_estimator_.named_steps["clf"].classes_
        classes = np.asarray(classes)
        subject_labels, subject_scores = aggregate_subject_scores(
            y, window_scores, subjects, aggregation, classification=True
        )
        y_pred = classes[np.argmax(subject_scores, axis=1)]
    elif task_type == "regression_based_cls":
        if y.ndim != 2 or y.shape[1] < 2:
            raise ValueError("regression-based-cls subject scoring requires 2D targets")
        window_scores = np.asarray(probe.predict(x))
        if window_scores.shape != y.shape:
            raise ValueError("regression-based-cls predictions must match target shape")
        subject_labels, subject_scores = aggregate_subject_scores(
            y, window_scores, subjects, aggregation, classification=False
        )
        y_pred = np.argmax(subject_scores, axis=1)
    else:
        raise ValueError(
            f"Subject aggregation is not implemented for task_type={task_type}"
        )
    results: dict[str, Any] = {
        "best_params": probe.best_params_,
        "aggregation": dict(aggregation),
        "n_subjects": int(len(subject_labels)),
        "n_windows": int(len(y)),
        "accuracy": accuracy_score(subject_labels, y_pred),
        "balanced_accuracy": balanced_accuracy_score(subject_labels, y_pred),
        "f1_macro": f1_score(subject_labels, y_pred, average="macro"),
    }
    if is_classification:
        unique_subject_labels = np.unique(subject_labels)
        if len(classes) == 2 and len(unique_subject_labels) == 2:
            positive_col = int(np.where(classes == classes[1])[0][0])
            store_metric(
                results,
                "auroc",
                lambda: roc_auc_score(subject_labels, subject_scores[:, positive_col]),
            )
            store_metric(
                results,
                "auprc",
                lambda: average_precision_score(
                    subject_labels, subject_scores[:, positive_col]
                ),
            )
        elif len(classes) == 2:
            results["auroc"] = None
            results["auroc_error"] = "Only one class present in subject labels"
            results["auprc"] = None
            results["auprc_error"] = "Only one class present in subject labels"
        elif len(classes) > 2 and len(unique_subject_labels) == len(classes):
            store_metric(
                results,
                "auroc_ovr",
                lambda: roc_auc_score(
                    subject_labels, subject_scores, multi_class="ovr"
                ),
            )
        elif len(classes) > 2:
            results["auroc_ovr"] = None
            results["auroc_ovr_error"] = "Not all classes are present in subject labels"
    return results


def aggregate_subject_scores(
    y: np.ndarray,
    window_scores: np.ndarray,
    subjects: np.ndarray,
    aggregation: Mapping[str, Any],
    classification: bool,
) -> tuple[np.ndarray, np.ndarray]:
    agg_name = aggregation["name"]
    subject_labels = []
    subject_scores = []
    for subject in dict.fromkeys(subjects.tolist()):
        idx = subjects == subject
        if classification:
            subject_y = y[idx]
            unique_y = np.unique(subject_y)
            if len(unique_y) != 1:
                raise ValueError(
                    f"Subject {subject} has mixed class labels: {unique_y.tolist()}"
                )
            subject_labels.append(unique_y[0])
        else:
            subject_y = y[idx]
            first_y = subject_y[0]
            if not np.allclose(subject_y, first_y):
                raise ValueError(
                    f"Subject {subject} has mixed regression-based-cls labels"
                )
            subject_labels.append(int(np.argmax(first_y)))
        scores = window_scores[idx]
        if agg_name == "mean":
            subject_scores.append(np.mean(scores, axis=0))
        elif agg_name == "median":
            subject_scores.append(np.median(scores, axis=0))
        elif agg_name == "topk_mean":
            topk_frac = float(aggregation["topk_frac"])
            k = max(1, int(ceil(topk_frac * len(scores))))
            sorted_scores = np.sort(scores, axis=0)
            if np.any(sorted_scores < 0.0) or np.any(sorted_scores > 1.0):
                raise ValueError(
                    "topk_mean aggregation expects confidence scores in [0, 1]"
                )
            low_mean = np.mean(sorted_scores[:k], axis=0)
            high_mean = np.mean(sorted_scores[-k:], axis=0)
            use_high = 1.0 - high_mean <= low_mean
            subject_scores.append(np.where(use_high, high_mean, low_mean))
        else:
            raise ValueError(f"Unknown subject aggregation: {agg_name}")
    return (np.asarray(subject_labels), np.asarray(subject_scores))


def score_linear_classical_probe(
    probe, x: np.ndarray, y: np.ndarray, task_type: str
) -> dict[str, Any]:
    y, task_type = prepare_targets(y, task_type, "scoring")
    y_pred = probe.predict(x)
    results: dict[str, Any] = {"best_params": probe.best_params_}
    if task_type == "regression":
        mse = mean_squared_error(y, y_pred)
        results.update(
            {
                "mae": mean_absolute_error(y, y_pred),
                "mse": mse,
                "rmse": float(np.sqrt(mse)),
                "r2": r2_score(y, y_pred),
            }
        )
        return results
    if task_type == "regression_based_cls":
        y_pred = np.asarray(y_pred)
        if y_pred.ndim != 2 or y_pred.shape[1] != y.shape[1]:
            raise ValueError("regression-based-cls predictions must match target shape")
        mse = mean_squared_error(y, y_pred)
        true_ties = np.sum(y == np.max(y, axis=1, keepdims=True), axis=1) > 1
        pred_ties = np.sum(y_pred == np.max(y_pred, axis=1, keepdims=True), axis=1) > 1
        y_argmax = np.argmax(y, axis=1)
        y_pred_argmax = np.argmax(y_pred, axis=1)
        results.update(
            {
                "mae": mean_absolute_error(y, y_pred),
                "mae_per_target": mean_absolute_error(
                    y, y_pred, multioutput="raw_values"
                ).tolist(),
                "mse": mse,
                "mse_per_target": mean_squared_error(
                    y, y_pred, multioutput="raw_values"
                ).tolist(),
                "rmse": float(np.sqrt(mse)),
                "r2": r2_score(y, y_pred),
                "r2_per_target": r2_score(y, y_pred, multioutput="raw_values").tolist(),
                "argmax_accuracy": accuracy_score(y_argmax, y_pred_argmax),
                "argmax_f1_macro": f1_score(y_argmax, y_pred_argmax, average="macro"),
                "argmax_balanced_accuracy": balanced_accuracy_score(
                    y_argmax, y_pred_argmax
                ),
                "argmax_true_tie_rate": float(np.mean(true_ties)),
                "argmax_true_tie_count": int(np.sum(true_ties)),
                "argmax_pred_tie_rate": float(np.mean(pred_ties)),
                "argmax_pred_tie_count": int(np.sum(pred_ties)),
            }
        )
        return results
    results.update(
        {
            "accuracy": accuracy_score(y, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y, y_pred),
            "f1_macro": f1_score(y, y_pred, average="macro"),
        }
    )
    if hasattr(probe, "predict_proba"):
        y_prob = probe.predict_proba(x)
        if task_type == "binary_classification" and y_prob.shape[1] == 2:
            store_metric(results, "auroc", lambda: roc_auc_score(y, y_prob[:, 1]))
            store_metric(
                results, "auprc", lambda: average_precision_score(y, y_prob[:, 1])
            )
        elif task_type == "multiclass_classification" and y_prob.shape[1] > 2:
            store_metric(
                results,
                "auroc_ovr",
                lambda: roc_auc_score(y, y_prob, multi_class="ovr"),
            )
    return results


def make_sklearn_rf_regressor(random_state: int):
    return RandomForestRegressor(random_state=random_state, n_jobs=1)


def make_sklearn_rf_classifier(random_state: int):
    return RandomForestClassifier(
        random_state=random_state, class_weight=None, n_jobs=1
    )


def make_cuml_rf_regressor(random_state: int, probe_cfg: Mapping[str, Any]):
    try:
        from cuml.ensemble import RandomForestRegressor as CuMLRandomForestRegressor
    except ImportError as exc:
        raise ImportError(
            "probe.backend='cuml' requires RAPIDS cuML in the active Python environment"
        ) from exc
    return CuMLRandomForestRegressor(
        random_state=random_state,
        output_type="numpy",
        n_bins=int(probe_cfg.get("n_bins", 128)),
    )


def make_cuml_rf_classifier(random_state: int, probe_cfg: Mapping[str, Any]):
    try:
        from cuml.ensemble import RandomForestClassifier as CuMLRandomForestClassifier
    except ImportError as exc:
        raise ImportError(
            "probe.backend='cuml' requires RAPIDS cuML in the active Python environment"
        ) from exc
    return CuMLRandomForestClassifier(
        random_state=random_state,
        output_type="numpy",
        n_bins=int(probe_cfg.get("n_bins", 128)),
    )


def rf_backend(probe_cfg: Mapping[str, Any]) -> str:
    backend = str(probe_cfg.get("backend", "cuml")).lower()
    if backend not in ("sklearn", "cuml"):
        raise ValueError(f"Unsupported nonlinear_classical backend: {backend}")
    return backend


def fit_nonlinear_classical_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    task_type: str,
    probe_cfg: Mapping[str, Any],
    subjects_train: np.ndarray,
    eval_level: str,
    aggregation_grid: list[dict[str, Any]] | None = None,
):
    requested_backend = rf_backend(probe_cfg)
    y_train, task_type = prepare_targets(y_train, task_type)
    random_state = int(probe_cfg.get("random_state", 0))
    effective_backend = (
        "sklearn" if task_type == "regression_based_cls" else requested_backend
    )
    regression = task_type in ("regression", "regression_based_cls")
    if task_type not in (
        "regression",
        "regression_based_cls",
        "binary_classification",
        "multiclass_classification",
    ):
        raise ValueError(f"Unsupported task_type for nonlinear_classical: {task_type}")
    if probe_cfg.get(task_type, "random_forest") != "random_forest":
        label = {
            "regression_based_cls": "regression-based-cls",
            "binary_classification": "binary classification",
            "multiclass_classification": "multiclass",
        }.get(task_type, task_type)
        raise ValueError(
            f"nonlinear_classical {label} currently supports only random_forest"
        )
    if effective_backend == "cuml":
        make_estimator = (
            make_cuml_rf_regressor if regression else make_cuml_rf_classifier
        )
        estimator = make_estimator(random_state, probe_cfg)
    else:
        make_estimator = (
            make_sklearn_rf_regressor if regression else make_sklearn_rf_classifier
        )
        estimator = make_estimator(random_state)
    grid_name = probe_cfg.get("param_grid")
    if grid_name not in (None, "standard"):
        raise ValueError(f"Unsupported nonlinear_classical param_grid: {grid_name}")
    probe = fit_grouped_grid_search(
        estimator=estimator,
        scoring=classical_scoring(task_type, probe_cfg),
        param_grid=STANDARD_RF_REG_PARAM_GRID
        if regression
        else STANDARD_RF_CLS_PARAM_GRID,
        x_train=x_train,
        y_train=y_train,
        subjects_train=subjects_train,
        task_type=task_type,
        eval_level=eval_level,
        inner_folds=int(probe_cfg["inner_folds"]),
        random_state=random_state,
        score_fn=score_linear_classical_probe,
        subject_score_fn=score_linear_classical_subject_probe,
        aggregation_grid=aggregation_grid,
    )
    probe.best_params_.update(
        backend=effective_backend,
        requested_backend=requested_backend,
        class_weight=None,
    )
    if effective_backend == "cuml":
        probe.best_params_["n_bins"] = int(probe_cfg.get("n_bins", 128))
    if requested_backend == "cuml" and effective_backend != "cuml":
        probe.best_params_["backend_note"] = (
            "cuML is not used for regression-based-cls/multioutput RF."
        )
    return probe
