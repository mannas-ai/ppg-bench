"""Task definitions and one loader for the processed benchmark format."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


DATASETS = {
    "dalia": "Dalia",
    "dreamt": "DREAMT",
    "wesad": "WESAD",
    "ecsmp": "ECSMP",
    "ppgbp": "PPG-BP",
    "bidmc": "BIDMC",
    "butppg": "BUT-PPG",
    "realworldppg": "RealWorldPPG",
    "wildppg": "WildPPG",
}


@dataclass(frozen=True)
class Task:
    name: str
    dataset: str
    target: str
    task_type: str
    eval_level: str = "window"
    primary: bool = True


TASKS = {}


def _tasks(dataset, rows):
    for suffix, target, kind, level, primary in rows:
        name = f"{dataset}-{suffix}"
        TASKS[name] = Task(name, dataset, target, kind, level, primary)


_tasks(
    "dalia",
    [
        ("hr", "hr", "regression", "window", True),
        ("activity", "activity", "classification", "window", True),
    ],
)
_tasks("dreamt", [("sleep-5class", "sleep_5class", "classification", "window", True)])
_tasks(
    "wesad",
    [
        ("4class", "condition_4class", "classification", "window", True),
        ("valence-reg", "valence_reg", "regression", "window", False),
        ("arousal-reg", "arousal_reg", "regression", "window", False),
    ],
)
_tasks(
    "ecsmp",
    [
        ("sds-subject", "sds_class", "classification", "subject", True),
        ("emotion-reg-cls", "emotion_reg_cls", "regression-based-cls", "window", False),
    ],
)
_tasks(
    "ppgbp",
    [
        (target.replace("_", "-"), target, kind, "subject", primary)
        for target, kind, primary in [
            ("age", "regression", True),
            ("bmi", "regression", True),
            ("sysbp", "regression", True),
            ("diasbp", "regression", True),
            ("hr", "regression", True),
            ("hypertension", "classification", True),
            ("diabetes", "classification", True),
            ("hypertension_binary", "classification", False),
            ("cerebral_infarction", "classification", False),
            ("cerebrovascular_disease", "classification", False),
        ]
    ],
)
_tasks(
    "bidmc",
    [(target, target, "regression", "window", True) for target in ("resp", "spo2")],
)
_tasks(
    "butppg",
    [
        (target, target, "regression", "window", target == "glyc")
        for target in ("glyc", "sbp", "dbp", "spo2")
    ],
)
_tasks("realworldppg", [("identification", "id", "classification", "window", True)])
_tasks("wildppg", [("hr", "hr", "regression", "window", True)])


def select_tasks(selection=None):
    """Accept dataset names, task names, or all (the default)."""
    selected = []
    for name in selection or ["all"]:
        matches = (
            list(TASKS)
            if name == "all"
            else [name]
            if name in TASKS
            else [key for key, task in TASKS.items() if task.dataset == name]
        )
        if not matches:
            raise ValueError(f"Unknown dataset/task: {name}. Run ppg-bench list.")
        selected.extend(matches)
    return [TASKS[name] for name in dict.fromkeys(selected)]


class BenchmarkDataset(Dataset):
    def __init__(self, root, subjects, task):
        self.subjects = []
        signals, targets, means, stds = [], [], [], []
        for subject in subjects:
            directory = Path(root) / subject
            x = np.load(directory / "ppg.npy").astype(np.float32)
            if x.ndim == 1:
                x = x[None, :]
            filename = (
                "cesaf_scores" if task.target == "emotion_reg_cls" else task.target
            )
            y = np.load(directory / f"{filename}.npy")
            if len(x) != len(y) or x.ndim != 2:
                raise ValueError(f"Invalid signal/target shapes in {directory}")
            if task.task_type == "regression-based-cls":
                valid = np.load(directory / "has_video_label.npy").astype(
                    bool
                ) & np.isfinite(y).all(axis=1)
            elif task.task_type == "classification":
                y = y.astype(np.int64).reshape(-1)
                valid = y != -1
            else:
                y = y.astype(np.float32).reshape(-1)
                valid = np.isfinite(y)
            y = y[valid]
            if not len(y):
                continue
            stats = json.loads((directory / "stats.json").read_text())
            signals.append(x[valid])
            targets.append(y)
            means.append(np.full(len(y), stats["ppg_mean"], dtype=np.float32))
            stds.append(np.full(len(y), stats["ppg_std"], dtype=np.float32))
            self.subjects.extend([subject] * len(y))
        if not signals:
            raise ValueError(f"No usable examples for {task.name} in {root}")
        self.x = np.concatenate(signals)
        self.y = np.concatenate(targets)
        self.ppg_mean, self.ppg_std = np.concatenate(means), np.concatenate(stds)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return {
            "x": torch.from_numpy(self.x[index]),
            "y": torch.as_tensor(self.y[index]),
            "subject": self.subjects[index],
            "ppg_mean": torch.as_tensor(self.ppg_mean[index]),
            "ppg_std": torch.as_tensor(self.ppg_std[index]),
        }

    def prepare_batch(self, batch):
        return batch
