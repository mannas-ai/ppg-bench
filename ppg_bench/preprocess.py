"""Dataset parsers retaining benchmark windows, labels and rejection rules."""

import csv
import gc
import json
import pickle
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from math import gcd
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import scipy.io as sio
from scipy.io import loadmat
from scipy.ndimage import maximum_filter1d, uniform_filter1d
from scipy.signal import butter, find_peaks, resample_poly, sosfiltfilt


def butterworth_filter(data, low_freq, high_freq, fs, order=4):
    from scipy.signal import butter, filtfilt

    nyquist = 0.5 * fs
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data)


def _resample_signal(signal, native_fs, target_fs):
    if target_fs == native_fs:
        return signal
    rate_gcd = gcd(target_fs, native_fs)
    return resample_poly(signal, target_fs // rate_gcd, native_fs // rate_gcd).astype(
        np.float32
    )


def save_stats(directory, stats):
    with (directory / "stats.json").open("w") as file:
        json.dump(stats, file, indent=2)


_XLSX_NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _xlsx_col_to_index(cell_ref):
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    value = 0
    for ch in letters:
        value = value * 26 + ord(ch.upper()) - 64
    return value - 1


def _xlsx_strings(archive):
    shared = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    return [
        "".join(t.text or "" for t in item.iter(f"{{{_XLSX_NS['a']}}}t"))
        for item in shared.findall("a:si", _XLSX_NS)
    ]


def _xlsx_rows(sheet, strings, numeric=False):
    rows = []
    for row in sheet.findall("a:sheetData/a:row", _XLSX_NS):
        values = {}
        for cell in row.findall("a:c", _XLSX_NS):
            idx = _xlsx_col_to_index(cell.attrib.get("r", ""))
            value_node = cell.find("a:v", _XLSX_NS)
            if value_node is None:
                continue
            value = value_node.text
            if cell.attrib.get("t") == "s":
                value = strings[int(value)]
            elif numeric:
                try:
                    value = float(value)
                except ValueError:
                    pass
            values[idx] = value
        rows.append(values)
    return rows


def _load_pickled_subject(subject_id, data_root):
    pkl_path = data_root / subject_id / f"{subject_id}.pkl"
    with pkl_path.open("rb") as f:
        return pickle.load(f, encoding="latin1")


def preprocdalia(cfg):
    WINDOW_LEN = 8
    FS_NATIVE = {"PPG": 64, "ACC": 32, "ECG": 700, "ACT": 4}
    DATA_ROOT = Path(cfg["dalia"]["root"])
    DATASET_NAME = cfg["dalia"]["name"]
    target_fs = int(cfg.get("common", {}).get("target_fs", FS_NATIVE["PPG"]))
    SAVE_ROOT = Path(cfg["dalia"]["save_root"]) / f"fs_{target_fs}"
    subjects = sorted(
        [p.name for p in DATA_ROOT.glob("S*") if (p / f"{p.name}.pkl").exists()],
        key=lambda s: int(s[1:]),
    )
    for s in subjects:
        sub_data = _load_pickled_subject(s, DATA_ROOT)
        ppg_raw = sub_data["signal"]["wrist"]["BVP"].squeeze()
        hr_raw = sub_data["label"].squeeze()
        activity_raw = sub_data["activity"].squeeze()
        low_f = cfg["dalia"].get("low_f", -1)
        high_f = cfg["dalia"].get("high_f", -1)
        if low_f > 0 and high_f > 0:
            ppg_raw = butterworth_filter(ppg_raw, low_f, high_f, FS_NATIVE["PPG"])
        ppg_raw = _resample_signal(ppg_raw, FS_NATIVE["PPG"], target_fs)
        ppg_windows = []
        ppg_len = WINDOW_LEN * target_fs
        STRIDE_S = 2
        STRIDE_SAMPLES_PPG = STRIDE_S * target_fs
        act_len = WINDOW_LEN * FS_NATIVE["ACT"]
        STRIDE_SAMPLES_ACT = STRIDE_S * FS_NATIVE["ACT"]
        ppg_windows = [
            ppg_raw[i : i + ppg_len]
            for i in range(0, len(ppg_raw) - ppg_len + 1, STRIDE_SAMPLES_PPG)
        ]
        activity_windows = [
            activity_raw[i : i + act_len]
            for i in range(0, len(activity_raw) - act_len + 1, STRIDE_SAMPLES_ACT)
        ]
        assert len(ppg_windows) == len(hr_raw), (
            f"Number of PPG windows ({len(ppg_windows)}) does not match number of HR labels ({len(hr_raw)}) for subject {s}"
        )
        ppg_for_stats = np.concatenate(
            [np.asarray(window, dtype=np.float32) for window in ppg_windows]
        )
        ppg_sub_mean = np.mean(ppg_for_stats)
        ppg_sub_std = np.std(ppg_for_stats)
        reject_idx = []
        window_stds = [np.std(window) for window in ppg_windows]
        subject_median_window_std = np.median(window_stds)
        reject_idx = [w < 0.05 * subject_median_window_std for w in window_stds]
        for i, act_window in enumerate(activity_windows):
            if len(set(act_window)) > 1:
                reject_idx[i] = True
        out_dir = SAVE_ROOT / DATASET_NAME / s
        out_dir.mkdir(parents=True, exist_ok=True)
        keep_idx = [i for i, reject in enumerate(reject_idx) if not reject]
        ppg_save = np.asarray([ppg_windows[i] for i in keep_idx], dtype=np.float32)
        hr_save = np.asarray([hr_raw[i] for i in keep_idx], dtype=np.float32)
        activity_save = np.asarray(
            [activity_windows[i][0] for i in keep_idx], dtype=np.int64
        )
        np.save(out_dir / "ppg.npy", ppg_save)
        np.save(out_dir / "hr.npy", hr_save)
        np.save(out_dir / "activity.npy", activity_save)
        save_stats(
            out_dir,
            {
                "subject": s,
                "ppg_mean": float(ppg_sub_mean),
                "ppg_std": float(ppg_sub_std),
            },
        )
        with (out_dir / "metadata.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "subject",
                    "window_idx",
                    "start_sec",
                    "end_sec",
                    "ppg_start",
                    "ppg_end",
                    "hr",
                    "activity",
                ]
            )
            for i in keep_idx:
                ppg_start = i * STRIDE_SAMPLES_PPG
                ppg_end = ppg_start + ppg_len
                start_sec = i * STRIDE_S
                end_sec = start_sec + WINDOW_LEN
                writer.writerow(
                    [
                        s,
                        i,
                        start_sec,
                        end_sec,
                        ppg_start,
                        ppg_end,
                        float(hr_raw[i]),
                        int(activity_windows[i][0]),
                    ]
                )
        print(
            f"{s}: saved {len(keep_idx)} windows out of {len(ppg_windows)} to {out_dir}. Rejected {np.sum(reject_idx)} windows."
        )
        print(
            f"Shapes: PPG {ppg_save.shape}, HR {hr_save.shape}, Activity {activity_save.shape}"
        )


def _wesad_load_dim_label_maps(subject_id, root):
    quest_path = root / subject_id / f"{subject_id}_quest.csv"
    condition_to_label = {"Base": 1, "TSST": 2, "Fun": 3, "Medi 1": 4, "Medi 2": 4}
    order = []
    dim_rows = []
    with quest_path.open(newline="") as f:
        reader = csv.reader(f, delimiter=";")
        for row in reader:
            if not row:
                continue
            key = row[0].strip()
            values = [v.strip() for v in row[1:] if v.strip()]
            if key == "# ORDER":
                order = values
            elif key == "# DIM" and len(values) >= 2:
                dim_rows.append((float(values[0]), float(values[1])))
    valence_values = {}
    arousal_values = {}
    for condition, (valence, arousal) in zip(order, dim_rows):
        label = condition_to_label.get(condition)
        if label is None:
            continue
        valence_values.setdefault(label, []).append(valence)
        arousal_values.setdefault(label, []).append(arousal)
    valence_by_label = {
        label: float(np.mean(values)) for label, values in valence_values.items()
    }
    arousal_by_label = {
        label: float(np.mean(values)) for label, values in arousal_values.items()
    }
    return (valence_by_label, arousal_by_label)


def preprocwesad(cfg):
    WINDOW_LEN = cfg["wesad"]["window_len_s"]
    FS_NATIVE = {"PPG": 64, "ACC": 32, "ECG": 700, "label": 700}
    DATA_ROOT = Path(cfg["wesad"]["root"])
    DATASET_NAME = cfg["wesad"]["name"]
    target_fs = int(cfg.get("common", {}).get("target_fs", FS_NATIVE["PPG"]))
    SAVE_ROOT = Path(cfg["wesad"]["save_root"]) / f"fs_{target_fs}"
    subjects = sorted(
        [p.name for p in DATA_ROOT.glob("S*") if (p / f"{p.name}.pkl").exists()],
        key=lambda s: int(s[1:]),
    )
    for s in subjects:
        sub_data = _load_pickled_subject(s, DATA_ROOT)
        ppg_raw = sub_data["signal"]["wrist"]["BVP"].squeeze()
        label_raw = sub_data["label"].squeeze()
        low_f = cfg["wesad"].get("low_f", -1)
        high_f = cfg["wesad"].get("high_f", -1)
        if low_f > 0 and high_f > 0:
            ppg_raw = butterworth_filter(ppg_raw, low_f, high_f, FS_NATIVE["PPG"])
        ppg_raw = _resample_signal(ppg_raw, FS_NATIVE["PPG"], target_fs)
        label_for_windows = label_raw
        ppg_windows = []
        ppg_len = int(round(WINDOW_LEN * target_fs))
        STRIDE_S = cfg["wesad"].get("stride_s", WINDOW_LEN)
        STRIDE_SAMPLES_PPG = int(round(STRIDE_S * target_fs))
        label_windows = []
        label_len = int(round(WINDOW_LEN * FS_NATIVE["label"]))
        STRIDE_SAMPLES_LABEL = int(round(STRIDE_S * FS_NATIVE["label"]))
        ppg_windows = [
            ppg_raw[i : i + ppg_len]
            for i in range(0, len(ppg_raw) - ppg_len + 1, STRIDE_SAMPLES_PPG)
        ]
        label_windows = [
            label_for_windows[i : i + label_len]
            for i in range(
                0, len(label_for_windows) - label_len + 1, STRIDE_SAMPLES_LABEL
            )
        ]
        assert len(ppg_windows) == len(label_windows), (
            f"Number of PPG windows ({len(ppg_windows)}) does not match number of label windows ({len(label_windows)}) for subject {s}"
        )
        condition_3class = []
        condition_4class = []
        valence_reg = []
        arousal_reg = []
        valence_class = []
        arousal_class = []
        valence_by_label, arousal_by_label = _wesad_load_dim_label_maps(s, DATA_ROOT)
        for window in label_windows:
            unique, counts = np.unique(window, return_counts=True)
            majority_label = unique[np.argmax(counts)]
            discard = np.max(counts) / len(window) < 0.9
            if discard:
                condition_3class.append(-1)
                condition_4class.append(-1)
                valence_reg.append(np.nan)
                arousal_reg.append(np.nan)
                valence_class.append(-1)
                arousal_class.append(-1)
                continue
            if majority_label == 1:
                condition_3class.append(0)
                condition_4class.append(0)
            elif majority_label == 2:
                condition_3class.append(1)
                condition_4class.append(1)
            elif majority_label == 3:
                condition_3class.append(2)
                condition_4class.append(2)
            elif majority_label == 4:
                condition_3class.append(-1)
                condition_4class.append(3)
            else:
                condition_3class.append(-1)
                condition_4class.append(-1)
            if majority_label in valence_by_label.keys():
                valence_reg.append(valence_by_label[majority_label])
                valence_class.append(1 if valence_by_label[majority_label] > 5 else 0)
            else:
                valence_reg.append(np.nan)
                valence_class.append(-1)
            if majority_label in arousal_by_label.keys():
                arousal_reg.append(arousal_by_label[majority_label])
                arousal_class.append(1 if arousal_by_label[majority_label] > 5 else 0)
            else:
                arousal_reg.append(np.nan)
                arousal_class.append(-1)
        assert (
            len(ppg_windows)
            == len(condition_3class)
            == len(condition_4class)
            == len(valence_reg)
            == len(arousal_reg)
            == len(valence_class)
            == len(arousal_class)
        ), f"Length mismatch for subject {s}"
        ppg_sub_mean = np.mean(ppg_raw)
        ppg_sub_std = np.std(ppg_raw)
        reject_idx = []
        window_stds = [np.std(window) for window in ppg_windows]
        subject_median_window_std = np.median(window_stds)
        reject_idx = [w < 0.05 * subject_median_window_std for w in window_stds]
        out_dir = SAVE_ROOT / DATASET_NAME / s
        out_dir.mkdir(parents=True, exist_ok=True)
        keep_idx = [i for i, reject in enumerate(reject_idx) if not reject]
        ppg_save = np.asarray([ppg_windows[i] for i in keep_idx], dtype=np.float32)
        condition_3class_save = np.asarray(
            [condition_3class[i] for i in keep_idx], dtype=np.int64
        )
        condition_4class_save = np.asarray(
            [condition_4class[i] for i in keep_idx], dtype=np.int64
        )
        valence_reg_save = np.asarray(
            [valence_reg[i] for i in keep_idx], dtype=np.float32
        )
        arousal_reg_save = np.asarray(
            [arousal_reg[i] for i in keep_idx], dtype=np.float32
        )
        valence_class_save = np.asarray(
            [valence_class[i] for i in keep_idx], dtype=np.int64
        )
        arousal_class_save = np.asarray(
            [arousal_class[i] for i in keep_idx], dtype=np.int64
        )
        np.save(out_dir / "ppg.npy", ppg_save)
        np.save(out_dir / "condition_3class.npy", condition_3class_save)
        np.save(out_dir / "condition_4class.npy", condition_4class_save)
        np.save(out_dir / "valence_reg.npy", valence_reg_save)
        np.save(out_dir / "arousal_reg.npy", arousal_reg_save)
        np.save(out_dir / "valence_class.npy", valence_class_save)
        np.save(out_dir / "arousal_class.npy", arousal_class_save)
        save_stats(
            out_dir,
            {
                "subject": s,
                "ppg_mean": float(ppg_sub_mean),
                "ppg_std": float(ppg_sub_std),
            },
        )
        with (out_dir / "metadata.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "subject",
                    "window_idx",
                    "start_sec",
                    "end_sec",
                    "ppg_start",
                    "ppg_end",
                    "condition_3class",
                    "condition_4class",
                    "valence_reg",
                    "arousal_reg",
                    "valence_class",
                    "arousal_class",
                ]
            )
            for i in keep_idx:
                ppg_start = i * STRIDE_SAMPLES_PPG
                ppg_end = ppg_start + ppg_len
                start_sec = i * STRIDE_S
                end_sec = start_sec + WINDOW_LEN
                writer.writerow(
                    [
                        s,
                        i,
                        start_sec,
                        end_sec,
                        ppg_start,
                        ppg_end,
                        condition_3class[i],
                        condition_4class[i],
                        valence_reg[i],
                        arousal_reg[i],
                        valence_class[i],
                        arousal_class[i],
                    ]
                )
        print(
            f"{s}: saved {len(keep_idx)} windows out of {len(ppg_windows)} to {out_dir}. Rejected {np.sum(reject_idx)} windows."
        )
        print(
            f"Shapes: PPG {ppg_save.shape}, condition_3class {condition_3class_save.shape}, condition_4class {condition_4class_save.shape}, valence_reg {valence_reg_save.shape}, arousal_reg {arousal_reg_save.shape}, valence_class {valence_class_save.shape}, arousal_class {arousal_class_save.shape}"
        )


_dreamt_SLEEP_STAGE_TO_CLASS_4CLASS = {"W": 0, "N1": 1, "N2": 1, "N3": 2, "R": 3}
_dreamt_SLEEP_STAGE_TO_CLASS_3CLASS = {"W": 0, "N1": 1, "N2": 1, "N3": 1, "R": 2}
_dreamt_SLEEP_STAGE_TO_CLASS_5CLASS = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4}


def _dreamt_load_bvp_and_sleep(csv_path):
    bvp_values = []
    sleep_values = []
    with csv_path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        bvp_idx = header.index("BVP")
        sleep_idx = header.index("Sleep_Stage")
        for row in reader:
            try:
                bvp_values.append(float(row[bvp_idx]))
            except ValueError:
                bvp_values.append(float("nan"))
            sleep_values.append(row[sleep_idx].strip())
    ppg = np.asarray(bvp_values, dtype=np.float32)
    sleep = np.asarray(sleep_values)
    return (ppg, sleep)


def preprocdreamt(cfg):
    dreamt_cfg = cfg["dreamt"]
    WINDOW_LEN = dreamt_cfg["window_len_s"]
    STRIDE_S = dreamt_cfg.get("stride_s", WINDOW_LEN)
    FS_NATIVE = 64
    DATA_ROOT = Path(dreamt_cfg["root"])
    DATASET_NAME = dreamt_cfg["name"]
    target_fs = int(cfg.get("common", {}).get("target_fs", FS_NATIVE))
    SAVE_ROOT = Path(dreamt_cfg["save_root"]) / f"fs_{target_fs}"
    subject_paths = sorted(
        (DATA_ROOT / "data_64Hz").glob("S*_whole_df.csv"),
        key=lambda p: int(p.name.split("_")[0][1:]),
    )
    flatline_std_frac = float(dreamt_cfg.get("flatline_std_frac", 0.05))
    for subject_path in subject_paths:
        s = subject_path.name.split("_")[0]
        ppg_raw, sleep_raw = _dreamt_load_bvp_and_sleep(subject_path)
        low_f = dreamt_cfg.get("low_f", -1)
        high_f = dreamt_cfg.get("high_f", -1)
        if low_f > 0 and high_f > 0:
            ppg_raw = butterworth_filter(ppg_raw, low_f, high_f, FS_NATIVE)
        ppg_raw = _resample_signal(ppg_raw, FS_NATIVE, target_fs)
        sleep_for_windows = sleep_raw
        finite_for_windows = np.isfinite(ppg_raw)
        ppg_len = int(round(WINDOW_LEN * target_fs))
        stride_samples_ppg = int(round(STRIDE_S * target_fs))
        label_len = int(round(WINDOW_LEN * FS_NATIVE))
        stride_samples_label = int(round(STRIDE_S * FS_NATIVE))
        n_ppg_windows = (len(ppg_raw) - ppg_len) // stride_samples_ppg + 1
        n_label_windows = (
            len(sleep_for_windows) - label_len
        ) // stride_samples_label + 1
        n_windows = min(n_ppg_windows, n_label_windows)
        if n_windows <= 0:
            raise ValueError(f"{s}: no full DREAMT windows found")
        ppg_windows = []
        sleep_3class = []
        sleep_4class = []
        sleep_5class = []
        sleep_stage = []
        reject_idx = []
        for i in range(n_windows):
            ppg_start = i * stride_samples_ppg
            label_start = i * stride_samples_label
            ppg_window = ppg_raw[ppg_start : ppg_start + ppg_len]
            finite_window = finite_for_windows[ppg_start : ppg_start + ppg_len]
            ppg_windows.append(ppg_window)
            sleep_window = sleep_for_windows[label_start : label_start + label_len]
            unique = np.unique(sleep_window)
            reject = False
            stage = ""
            if not finite_window.all():
                reject = True
            elif len(unique) != 1:
                reject = True
            else:
                stage = str(unique[0])
                if stage not in _dreamt_SLEEP_STAGE_TO_CLASS_3CLASS:
                    reject = True
            if reject:
                sleep_3class.append(-1)
                sleep_4class.append(-1)
                sleep_5class.append(-1)
            else:
                sleep_3class.append(_dreamt_SLEEP_STAGE_TO_CLASS_3CLASS[stage])
                sleep_4class.append(_dreamt_SLEEP_STAGE_TO_CLASS_4CLASS[stage])
                sleep_5class.append(_dreamt_SLEEP_STAGE_TO_CLASS_5CLASS[stage])
            sleep_stage.append(stage)
            reject_idx.append(reject)
        finite_ppg = ppg_raw[np.isfinite(ppg_raw)]
        if len(finite_ppg) == 0:
            raise ValueError(f"{s}: no finite PPG samples found")
        ppg_sub_mean = np.mean(finite_ppg)
        ppg_sub_std = np.std(finite_ppg)
        window_stds = [np.std(window) for window in ppg_windows]
        valid_window_stds = [
            w for w, reject in zip(window_stds, reject_idx) if not reject
        ]
        subject_median_window_std = (
            np.median(valid_window_stds) if valid_window_stds else 0
        )
        reject_idx = [
            reject or w < flatline_std_frac * subject_median_window_std
            for w, reject in zip(window_stds, reject_idx)
        ]
        out_dir = SAVE_ROOT / DATASET_NAME / s
        out_dir.mkdir(parents=True, exist_ok=True)
        keep_idx = [i for i, reject in enumerate(reject_idx) if not reject]
        ppg_save = np.asarray([ppg_windows[i] for i in keep_idx], dtype=np.float32)
        sleep_3class_save = np.asarray(
            [sleep_3class[i] for i in keep_idx], dtype=np.int64
        )
        sleep_4class_save = np.asarray(
            [sleep_4class[i] for i in keep_idx], dtype=np.int64
        )
        sleep_5class_save = np.asarray(
            [sleep_5class[i] for i in keep_idx], dtype=np.int64
        )
        np.save(out_dir / "ppg.npy", ppg_save)
        np.save(out_dir / "sleep_3class.npy", sleep_3class_save)
        np.save(out_dir / "sleep_4class.npy", sleep_4class_save)
        np.save(out_dir / "sleep_5class.npy", sleep_5class_save)
        save_stats(
            out_dir,
            {
                "subject": s,
                "ppg_mean": float(ppg_sub_mean),
                "ppg_std": float(ppg_sub_std),
            },
        )
        with (out_dir / "metadata.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "subject",
                    "window_idx",
                    "start_sec",
                    "end_sec",
                    "ppg_start",
                    "ppg_end",
                    "sleep_stage",
                    "sleep_3class",
                    "sleep_4class",
                    "sleep_5class",
                ]
            )
            for i in keep_idx:
                ppg_start = i * stride_samples_ppg
                ppg_end = ppg_start + ppg_len
                start_sec = i * STRIDE_S
                end_sec = start_sec + WINDOW_LEN
                writer.writerow(
                    [
                        s,
                        i,
                        start_sec,
                        end_sec,
                        ppg_start,
                        ppg_end,
                        sleep_stage[i],
                        sleep_3class[i],
                        sleep_4class[i],
                        sleep_5class[i],
                    ]
                )
        print(
            f"{s}: saved {len(keep_idx)} windows out of {n_windows} to {out_dir}. Rejected {np.sum(reject_idx)} windows."
        )
        print(
            f"Shapes: PPG {ppg_save.shape}, Sleep 3-class {sleep_3class_save.shape}, Sleep 4-class {sleep_4class_save.shape}, Sleep 5-class {sleep_5class_save.shape}"
        )


_ecsmp_EMOTION_CODE_TO_NAME = {
    101: "neutral",
    102: "fear",
    103: "sad",
    104: "happy",
    105: "anger",
    106: "disgust",
}
_ecsmp_EMOTIONS = ["neutral", "fear", "sad", "happy", "anger", "disgust"]
_ecsmp_EMOTION_TO_CLASS = {emotion: idx for idx, emotion in enumerate(_ecsmp_EMOTIONS)}


def _ecsmp_format_subject_id(value):
    if isinstance(value, float) and value.is_integer():
        return str(int(value)).zfill(3)
    return str(value).strip().zfill(3)


def _ecsmp_load_bvp(subject_id, data_root):
    bvp_path = data_root / "E4" / subject_id / "BVP.csv"
    if not bvp_path.exists():
        raise FileNotFoundError(f"Missing BVP file: {bvp_path}")
    with bvp_path.open() as f:
        start_utc = float(f.readline().strip())
        fs = int(round(float(f.readline().strip())))
    signal = np.loadtxt(bvp_path, delimiter=",", skiprows=2).astype(np.float32)
    return (signal, fs, start_utc)


def _ecsmp_load_eeg_events(subject_id, data_root):
    matches = sorted((data_root / "EEG_downsample").glob(f"*_{subject_id}.mat"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one EEG .mat for {subject_id}, found {len(matches)}: {matches}"
        )
    mat = sio.loadmat(matches[0], squeeze_me=True, struct_as_record=False)
    eeg = mat["EEG"]
    srate = float(eeg.srate)
    events = []
    for event in np.ravel(eeg.urevent):
        code = int(event.type)
        eeg_sample = float(event.latency)
        events.append(
            {"code": code, "eeg_sample": eeg_sample, "eeg_time_s": eeg_sample / srate}
        )
    return (events, srate, matches[0])


def _ecsmp_make_video_intervals(events, bvp_fs, bvp_duration_s):
    starts = [event for event in events if event["code"] in _ecsmp_EMOTION_CODE_TO_NAME]
    intervals = []
    for i, start_event in enumerate(starts):
        start_s = start_event["eeg_time_s"]
        next_start_s = (
            starts[i + 1]["eeg_time_s"] if i + 1 < len(starts) else float("inf")
        )
        end_candidates_20 = [
            event
            for event in events
            if event["code"] == 20 and start_s < event["eeg_time_s"] < next_start_s
        ]
        end_candidates_12 = [
            event
            for event in events
            if event["code"] == 12 and start_s < event["eeg_time_s"] < next_start_s
        ]
        if end_candidates_20:
            end_event = end_candidates_20[0]
        elif end_candidates_12:
            end_event = end_candidates_12[0]
        else:
            end_event = None
        if end_event is None:
            end_s = np.nan
            end_marker = None
            end_idx = None
        else:
            end_s = end_event["eeg_time_s"]
            end_marker = end_event["code"]
            end_idx = int(round(end_s * bvp_fs))
        start_idx = int(round(start_s * bvp_fs))
        intervals.append(
            {
                "emotion": _ecsmp_EMOTION_CODE_TO_NAME[start_event["code"]],
                "start_marker": start_event["code"],
                "end_marker": end_marker,
                "start_eeg_time_s": start_s,
                "end_eeg_time_s": end_s,
                "start_bvp_idx": start_idx,
                "end_bvp_idx": end_idx,
                "valid_for_bvp": end_idx is not None
                and start_idx >= 0
                and (end_idx > start_idx)
                and (end_idx <= int(round(bvp_duration_s * bvp_fs))),
            }
        )
    return intervals


def _ecsmp_read_xlsx_sheet(xlsx_path, sheet_name):
    ns = {
        "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    }
    with ZipFile(xlsx_path) as zf:
        strings = _xlsx_strings(zf) if "xl/sharedStrings.xml" in zf.namelist() else []
        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_targets = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}
        target = None
        for sheet in workbook.findall("a:sheets/a:sheet", ns):
            if sheet.attrib["name"] == sheet_name:
                rid = sheet.attrib[
                    "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
                ]
                target = rel_targets[rid]
                break
        if target is None:
            raise ValueError(f"Missing sheet: {sheet_name}")
        if not target.startswith("worksheets/"):
            target = "worksheets/" + target.split("/")[-1]
        xml = ET.fromstring(zf.read("xl/" + target))
        return _xlsx_rows(xml, strings, numeric=True)


def _ecsmp_load_scale_maps(data_root):
    cesaf_rows = _ecsmp_read_xlsx_sheet(data_root / "scale.xlsx", "CESAF")
    sds_rows = _ecsmp_read_xlsx_sheet(data_root / "scale.xlsx", "SDS")
    cesaf_by_subject = {}
    for row in cesaf_rows[2:]:
        subject_id = _ecsmp_format_subject_id(row.get(0, ""))
        if subject_id == "000":
            continue
        video_order = [row.get(i) for i in range(42, 48)]
        if len(video_order) != 6 or any(
            (video not in _ecsmp_EMOTIONS for video in video_order)
        ):
            continue
        videos = []
        for video_idx in range(6):
            base_col = 5 + video_idx * 6
            scores = {
                emotion: row.get(base_col + i)
                for i, emotion in enumerate(_ecsmp_EMOTIONS)
            }
            if any((score is None for score in scores.values())):
                continue
            values = np.asarray(
                [scores[emotion] for emotion in _ecsmp_EMOTIONS], dtype=np.float32
            )
            max_score = float(np.max(values))
            winners = np.where(values == max_score)[0]
            videos.append(
                {
                    "stimulus": video_order[video_idx],
                    "scores": scores,
                    "score_vector": values,
                    "felt_argmax": int(winners[0]) if len(winners) == 1 else -1,
                    "felt_tie": len(winners) != 1,
                }
            )
        if len(videos) == 6:
            cesaf_by_subject[subject_id] = videos
    sds_by_subject = {}
    for row in sds_rows[1:]:
        subject_id = _ecsmp_format_subject_id(row.get(0, ""))
        score = row.get(25)
        if subject_id == "000" or score is None:
            continue
        sds_by_subject[subject_id] = float(score)
    return (cesaf_by_subject, sds_by_subject)


def _ecsmp_list_subjects(data_root, configured_subjects=None):
    if configured_subjects:
        return [
            _ecsmp_format_subject_id(subject_id) for subject_id in configured_subjects
        ]
    e4_subjects = {
        path.name
        for path in (data_root / "E4").iterdir()
        if path.is_dir() and (path / "BVP.csv").exists()
    }
    eeg_subjects = {
        path.stem.split("_")[-1]
        for path in (data_root / "EEG_downsample").glob("*.mat")
    }
    return sorted(e4_subjects & eeg_subjects, key=lambda value: int(value))


def _ecsmp_add_cesaf_to_intervals(intervals, cesaf):
    event_order = [row["emotion"] for row in intervals]
    cesaf_order = [row["stimulus"] for row in cesaf]
    if event_order != cesaf_order:
        raise ValueError(
            f"EEG event order does not match CESAF order: {event_order} vs {cesaf_order}"
        )
    for interval, label_row in zip(intervals, cesaf):
        interval["stimulus"] = label_row["stimulus"]
        interval["stimulus_class"] = _ecsmp_EMOTION_TO_CLASS[label_row["stimulus"]]
        interval["felt_scores"] = label_row["score_vector"]
        interval["felt_argmax"] = label_row["felt_argmax"]
        interval["felt_tie"] = label_row["felt_tie"]
    return intervals


def _ecsmp_find_containing_interval(window_start_s, window_end_s, intervals):
    matches = []
    for idx, interval in enumerate(intervals):
        if (
            window_start_s >= interval["start_eeg_time_s"]
            and window_end_s <= interval["end_eeg_time_s"]
        ):
            matches.append(idx)
    if len(matches) == 1:
        return matches[0]
    return None


def preprocecsmp(cfg):
    ecsmp_cfg = cfg["ecsmp"]
    window_len_s = ecsmp_cfg["window_len_s"]
    stride_s = ecsmp_cfg.get("stride_s", window_len_s)
    fs_native = 64
    data_root = Path(ecsmp_cfg.get("root", ecsmp_cfg.get("data_root")))
    dataset_name = ecsmp_cfg["name"]
    target_fs = int(cfg.get("common", {}).get("target_fs", fs_native))
    save_root = Path(ecsmp_cfg["save_root"]) / f"fs_{target_fs}"
    strict = bool(ecsmp_cfg.get("strict", False))
    cesaf_by_subject, sds_by_subject = _ecsmp_load_scale_maps(data_root)
    subjects = _ecsmp_list_subjects(data_root, ecsmp_cfg.get("subjects"))
    skipped = []
    for subject_id in subjects:
        try:
            ppg_raw, bvp_fs, bvp_start_utc = _ecsmp_load_bvp(subject_id, data_root)
            events, eeg_fs, eeg_path = _ecsmp_load_eeg_events(subject_id, data_root)
            bvp_duration_s = len(ppg_raw) / bvp_fs
            intervals = _ecsmp_make_video_intervals(events, bvp_fs, bvp_duration_s)
            if len(intervals) != 6:
                raise ValueError(f"Expected six video starts, found {len(intervals)}")
            if not all((row["valid_for_bvp"] for row in intervals)):
                raise ValueError(
                    "At least one video interval does not fit inside the BVP recording"
                )
            cesaf = cesaf_by_subject.get(subject_id)
            if cesaf is None:
                raise ValueError("Missing CESAF data")
            intervals = _ecsmp_add_cesaf_to_intervals(intervals, cesaf)
            sds_score = sds_by_subject.get(subject_id)
            if sds_score is None:
                raise ValueError("Missing SDS score")
            sds_class = int(sds_score >= 50)
            low_f = ecsmp_cfg.get("low_f", -1)
            high_f = ecsmp_cfg.get("high_f", -1)
            if low_f > 0 and high_f > 0:
                ppg_raw = butterworth_filter(ppg_raw, low_f, high_f, bvp_fs)
            ppg_raw = _resample_signal(ppg_raw, bvp_fs, target_fs)
            ppg_len = int(round(window_len_s * target_fs))
            stride_samples = int(round(stride_s * target_fs))
            ppg_windows = [
                ppg_raw[i : i + ppg_len]
                for i in range(0, len(ppg_raw) - ppg_len + 1, stride_samples)
            ]
            cesaf_scores = []
            stimulus = []
            felt_argmax = []
            felt_tie = []
            has_video_label = []
            video_idx = []
            interval_start_s = []
            interval_end_s = []
            for i in range(len(ppg_windows)):
                window_start_sample = i * stride_samples
                window_end_sample = window_start_sample + ppg_len
                window_start_s = window_start_sample / target_fs
                window_end_s = window_end_sample / target_fs
                interval_idx = _ecsmp_find_containing_interval(
                    window_start_s, window_end_s, intervals
                )
                if interval_idx is None:
                    cesaf_scores.append(
                        np.full(len(_ecsmp_EMOTIONS), np.nan, dtype=np.float32)
                    )
                    stimulus.append(-1)
                    felt_argmax.append(-1)
                    felt_tie.append(False)
                    has_video_label.append(False)
                    video_idx.append(-1)
                    interval_start_s.append(np.nan)
                    interval_end_s.append(np.nan)
                    continue
                interval = intervals[interval_idx]
                cesaf_scores.append(interval["felt_scores"])
                stimulus.append(interval["stimulus_class"])
                felt_argmax.append(interval["felt_argmax"])
                felt_tie.append(interval["felt_tie"])
                has_video_label.append(True)
                video_idx.append(interval_idx)
                interval_start_s.append(interval["start_eeg_time_s"])
                interval_end_s.append(interval["end_eeg_time_s"])
            ppg_sub_mean = np.mean(ppg_raw)
            ppg_sub_std = np.std(ppg_raw)
            window_stds = np.asarray([np.std(window) for window in ppg_windows])
            subject_median_window_std = np.median(window_stds)
            reject_idx = window_stds < 0.05 * subject_median_window_std
            keep_idx = [i for i, reject in enumerate(reject_idx) if not reject]
            out_dir = save_root / dataset_name / subject_id
            out_dir.mkdir(parents=True, exist_ok=True)
            ppg_save = np.asarray([ppg_windows[i] for i in keep_idx], dtype=np.float32)
            cesaf_save = np.asarray(
                [cesaf_scores[i] for i in keep_idx], dtype=np.float32
            )
            stimulus_save = np.asarray([stimulus[i] for i in keep_idx], dtype=np.int64)
            felt_argmax_save = np.asarray(
                [felt_argmax[i] for i in keep_idx], dtype=np.int64
            )
            felt_tie_save = np.asarray([felt_tie[i] for i in keep_idx], dtype=bool)
            has_video_label_save = np.asarray(
                [has_video_label[i] for i in keep_idx], dtype=bool
            )
            sds_score_save = np.full(len(keep_idx), sds_score, dtype=np.float32)
            sds_class_save = np.full(len(keep_idx), sds_class, dtype=np.int64)
            np.save(out_dir / "ppg.npy", ppg_save)
            np.save(out_dir / "cesaf_scores.npy", cesaf_save)
            np.save(out_dir / "stimulus.npy", stimulus_save)
            np.save(out_dir / "felt_argmax.npy", felt_argmax_save)
            np.save(out_dir / "felt_tie.npy", felt_tie_save)
            np.save(out_dir / "has_video_label.npy", has_video_label_save)
            np.save(out_dir / "sds_score.npy", sds_score_save)
            np.save(out_dir / "sds_class.npy", sds_class_save)
            np.save(out_dir / "reject_idx.npy", reject_idx)
            save_stats(
                out_dir,
                {
                    "subject": subject_id,
                    "ppg_mean": float(ppg_sub_mean),
                    "ppg_std": float(ppg_sub_std),
                    "sds_score": float(sds_score),
                    "sds_class": int(sds_class),
                    "bvp_start_utc": float(bvp_start_utc),
                    "bvp_fs_original": int(bvp_fs),
                    "target_fs": int(target_fs),
                    "eeg_fs": float(eeg_fs),
                    "eeg_file": str(eeg_path.name),
                    "num_windows": int(len(ppg_windows)),
                    "num_kept_windows": int(len(keep_idx)),
                    "num_video_labeled_windows": int(np.sum(has_video_label_save)),
                    "num_tied_felt_windows": int(
                        np.sum(felt_tie_save & has_video_label_save)
                    ),
                },
            )
            with (out_dir / "metadata.csv").open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "subject",
                        "window_idx",
                        "start_sec",
                        "end_sec",
                        "ppg_start",
                        "ppg_end",
                        "has_video_label",
                        "video_idx",
                        "stimulus",
                        "felt_argmax",
                        "felt_tie",
                        "sds_score",
                        "sds_class",
                        "video_start_sec",
                        "video_end_sec",
                    ]
                )
                for i in keep_idx:
                    ppg_start = i * stride_samples
                    ppg_end = ppg_start + ppg_len
                    start_sec = ppg_start / target_fs
                    end_sec = ppg_end / target_fs
                    writer.writerow(
                        [
                            subject_id,
                            i,
                            start_sec,
                            end_sec,
                            ppg_start,
                            ppg_end,
                            int(has_video_label[i]),
                            video_idx[i],
                            stimulus[i],
                            felt_argmax[i],
                            int(felt_tie[i]),
                            sds_score,
                            sds_class,
                            interval_start_s[i],
                            interval_end_s[i],
                        ]
                    )
            print(
                f"{subject_id}: saved {len(keep_idx)} windows out of {len(ppg_windows)} to {out_dir}. Video-labeled windows: {int(np.sum(has_video_label_save))}. Rejected {int(np.sum(reject_idx))} windows."
            )
            print(
                f"Shapes: PPG {ppg_save.shape}, CESAF {cesaf_save.shape}, SDS {sds_class_save.shape}"
            )
        except Exception as exc:
            if strict:
                raise
            skipped.append((subject_id, str(exc)))
            print(f"{subject_id}: skipped - {exc}")
    if skipped:
        print("Skipped ECSMP subjects:")
        for subject_id, reason in skipped:
            print(f"  {subject_id}: {reason}")


_ppgbp_HYPERTENSION_TO_CLASS = {
    "Normal": 0,
    "Prehypertension": 1,
    "Stage 1 hypertension": 2,
    "Stage 2 hypertension": 3,
}


def _ppgbp_read_workbook_rows(xlsx_path):
    with ZipFile(xlsx_path) as zf:
        strings = _xlsx_strings(zf)
        sheet = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
        return _xlsx_rows(sheet, strings)


def _ppgbp_load_labels(data_root):
    rows = _ppgbp_read_workbook_rows(data_root / "PPG-BP dataset.xlsx")
    header = [rows[1].get(i, "") for i in range(14)]
    col = {name: idx for idx, name in enumerate(header)}
    labels = {}
    for row in rows[2:]:
        subject = str(int(float(row[col["subject_ID"]]))).strip()
        hypertension = str(row.get(col["Hypertension"], "")).strip()
        diabetes = str(row.get(col["Diabetes"], "")).strip()
        cerebral_infarction = str(row.get(col["cerebral infarction"], "")).strip()
        cerebrovascular_disease = str(
            row.get(col["cerebrovascular disease"], "")
        ).strip()
        labels[subject] = {
            "age": float(row[col["Age(year)"]]),
            "bmi": float(row[col["BMI(kg/m^2)"]]),
            "sysbp": float(row[col["Systolic Blood Pressure(mmHg)"]]),
            "diasbp": float(row[col["Diastolic Blood Pressure(mmHg)"]]),
            "hr": float(row[col["Heart Rate(b/m)"]]),
            "hypertension": _ppgbp_HYPERTENSION_TO_CLASS[hypertension],
            "hypertension_binary": int(hypertension != "Normal"),
            "diabetes": int(diabetes != ""),
            "cerebral_infarction": int(cerebral_infarction != ""),
            "cerebrovascular_disease": int(cerebrovascular_disease != ""),
        }
    return labels


def _ppgbp_load_ppg_segment(segment_path):
    signal = np.fromstring(segment_path.read_text(), sep="\t").astype(np.float32)
    if len(signal) == 0:
        raise ValueError(f"Empty PPG-BP segment: {segment_path}")
    return signal


def preprocppgbp(cfg):
    ppgbp_cfg = cfg["ppgbp"]
    FS_NATIVE = 1000
    SEGMENT_LEN_NATIVE = 2100
    SEGMENT_LEN_S = SEGMENT_LEN_NATIVE / FS_NATIVE
    DATA_ROOT = Path(ppgbp_cfg["root"])
    DATASET_NAME = ppgbp_cfg["name"]
    target_fs = int(cfg.get("common", {}).get("target_fs", FS_NATIVE))
    SAVE_ROOT = Path(ppgbp_cfg["save_root"]) / f"fs_{target_fs}"
    labels_by_subject = _ppgbp_load_labels(DATA_ROOT)
    skip_subjects = {"231"}
    subject_ids = [
        subject
        for subject in sorted(labels_by_subject.keys(), key=lambda value: int(value))
        if subject not in skip_subjects
    ]
    flatline_std_frac = float(ppgbp_cfg.get("flatline_std_frac", 0.05))
    for subject in subject_ids:
        subject_labels = labels_by_subject[subject]
        ppg_segments = []
        segment_ids = []
        rejected = 0
        for segment_idx in range(1, 4):
            segment_path = DATA_ROOT / "0_subject" / f"{subject}_{segment_idx}.txt"
            if not segment_path.exists():
                raise FileNotFoundError(segment_path)
            ppg = _ppgbp_load_ppg_segment(segment_path)
            if len(ppg) != SEGMENT_LEN_NATIVE:
                rejected += 1
                continue
            low_f = ppgbp_cfg.get("low_f", -1)
            high_f = ppgbp_cfg.get("high_f", -1)
            if low_f > 0 and high_f > 0:
                ppg = butterworth_filter(ppg, low_f, high_f, FS_NATIVE)
            ppg = _resample_signal(ppg, FS_NATIVE, target_fs)
            if not np.isfinite(ppg).all():
                rejected += 1
                continue
            ppg_segments.append(ppg)
            segment_ids.append(segment_idx)
        if not ppg_segments:
            raise ValueError(f"{subject}: no valid PPG-BP segments found")
        ppg_subject = np.concatenate(ppg_segments)
        ppg_sub_mean = np.mean(ppg_subject)
        ppg_sub_std = np.std(ppg_subject)
        segment_stds = [np.std(segment) for segment in ppg_segments]
        subject_median_segment_std = np.median(segment_stds)
        reject_idx = [
            std < flatline_std_frac * subject_median_segment_std for std in segment_stds
        ]
        rejected += int(np.sum(reject_idx))
        keep_idx = [i for i, reject in enumerate(reject_idx) if not reject]
        out_dir = SAVE_ROOT / DATASET_NAME / subject
        out_dir.mkdir(parents=True, exist_ok=True)
        ppg_save = np.asarray([ppg_segments[i] for i in keep_idx], dtype=np.float32)
        np.save(out_dir / "ppg.npy", ppg_save)
        for label_name in [
            "age",
            "bmi",
            "sysbp",
            "diasbp",
            "hr",
            "hypertension",
            "hypertension_binary",
            "diabetes",
            "cerebral_infarction",
            "cerebrovascular_disease",
        ]:
            dtype = (
                np.float32
                if label_name in {"age", "bmi", "sysbp", "diasbp", "hr"}
                else np.int64
            )
            np.save(
                out_dir / f"{label_name}.npy",
                np.full(len(keep_idx), subject_labels[label_name], dtype=dtype),
            )
        save_stats(
            out_dir,
            {
                "subject": subject,
                "ppg_mean": float(ppg_sub_mean),
                "ppg_std": float(ppg_sub_std),
            },
        )
        with (out_dir / "metadata.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "subject",
                    "segment_idx",
                    "raw_file",
                    "start_sec",
                    "end_sec",
                    "age",
                    "bmi",
                    "sysbp",
                    "diasbp",
                    "hr",
                    "hypertension",
                    "hypertension_binary",
                    "diabetes",
                    "cerebral_infarction",
                    "cerebrovascular_disease",
                ]
            )
            for i in keep_idx:
                segment_idx = segment_ids[i]
                writer.writerow(
                    [
                        subject,
                        segment_idx,
                        f"{subject}_{segment_idx}.txt",
                        0.0,
                        SEGMENT_LEN_S,
                        subject_labels["age"],
                        subject_labels["bmi"],
                        subject_labels["sysbp"],
                        subject_labels["diasbp"],
                        subject_labels["hr"],
                        subject_labels["hypertension"],
                        subject_labels["hypertension_binary"],
                        subject_labels["diabetes"],
                        subject_labels["cerebral_infarction"],
                        subject_labels["cerebrovascular_disease"],
                    ]
                )
        print(
            f"{subject}: saved {len(keep_idx)} segments out of 3 to {out_dir}. Rejected {rejected} segments."
        )
        print(f"Shapes: PPG {ppg_save.shape}")


_bidmc_FS_NATIVE = 125


def _bidmc_record_ids(csv_root):
    return sorted(
        (path.name.split("_")[1] for path in csv_root.glob("bidmc_*_Signals.csv"))
    )


def _bidmc_load_csv(path):
    with Path(path).open("r") as f:
        header = [name.strip() for name in f.readline().strip().split(",")]
    data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] != len(header):
        raise ValueError(
            f"Unexpected schema in {path}: header has {len(header)} columns, data has {data.shape[1]}"
        )
    return (header, data)


def _bidmc_column(header, name):
    try:
        return header.index(name)
    except ValueError as exc:
        raise ValueError(f"Missing required column {name!r}; found {header}") from exc


def preprocbidmc(cfg):
    dataset_cfg = cfg["bidmc"]
    data_root = Path(dataset_cfg["root"])
    csv_root = data_root / "bidmc_csv"
    save_root = Path(dataset_cfg["save_root"])
    dataset_name = dataset_cfg["name"]
    target_fs = int(cfg.get("common", {}).get("target_fs", _bidmc_FS_NATIVE))
    window_len_s = int(dataset_cfg.get("window_len_s", 10))
    stride_s = int(dataset_cfg.get("stride_s", window_len_s))
    window_samples = window_len_s * target_fs
    stride_samples = stride_s * target_fs
    low_f = dataset_cfg.get("low_f", -1)
    high_f = dataset_cfg.get("high_f", -1)
    for record_id in _bidmc_record_ids(csv_root):
        signal_header, signals = _bidmc_load_csv(
            csv_root / f"bidmc_{record_id}_Signals.csv"
        )
        numeric_header, numerics = _bidmc_load_csv(
            csv_root / f"bidmc_{record_id}_Numerics.csv"
        )
        ppg = signals[:, _bidmc_column(signal_header, "PLETH")]
        numeric_time = numerics[:, _bidmc_column(numeric_header, "Time [s]")]
        resp = numerics[:, _bidmc_column(numeric_header, "RESP")]
        spo2 = numerics[:, _bidmc_column(numeric_header, "SpO2")]
        if low_f > 0 and high_f > 0:
            ppg = butterworth_filter(ppg, low_f, high_f, _bidmc_FS_NATIVE)
        ppg = _resample_signal(ppg, _bidmc_FS_NATIVE, target_fs)
        ppg_windows = []
        resp_targets = []
        spo2_targets = []
        metadata_rows = []
        for window_idx, ppg_start in enumerate(
            range(0, len(ppg) - window_samples + 1, stride_samples)
        ):
            ppg_end = ppg_start + window_samples
            start_sec = window_idx * stride_s
            end_sec = start_sec + window_len_s
            target_mask = (numeric_time >= start_sec) & (numeric_time < end_sec)
            resp_window = resp[target_mask]
            spo2_window = spo2[target_mask]
            if len(resp_window) != window_len_s or len(spo2_window) != window_len_s:
                continue
            if not np.isfinite(ppg[ppg_start:ppg_end]).all():
                continue
            if not np.isfinite(resp_window).all() or not np.isfinite(spo2_window).all():
                continue
            ppg_windows.append(ppg[ppg_start:ppg_end])
            resp_targets.append(resp_window.mean())
            spo2_targets.append(spo2_window.mean())
            metadata_rows.append(
                (
                    window_idx,
                    start_sec,
                    end_sec,
                    ppg_start,
                    ppg_end,
                    resp_targets[-1],
                    spo2_targets[-1],
                )
            )
        out_dir = save_root / f"fs_{target_fs}" / dataset_name / f"bidmc{record_id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        ppg_save = np.asarray(ppg_windows, dtype=np.float32)
        resp_save = np.asarray(resp_targets, dtype=np.float32)
        spo2_save = np.asarray(spo2_targets, dtype=np.float32)
        np.save(out_dir / "ppg.npy", ppg_save)
        np.save(out_dir / "resp.npy", resp_save)
        np.save(out_dir / "spo2.npy", spo2_save)
        save_stats(
            out_dir,
            {
                "subject": f"bidmc{record_id}",
                "ppg_mean": float(ppg.mean()),
                "ppg_std": float(ppg.std()),
                "native_fs": _bidmc_FS_NATIVE,
                "target_fs": target_fs,
                "window_len_s": window_len_s,
                "stride_s": stride_s,
            },
        )
        with (out_dir / "metadata.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "subject",
                    "window_idx",
                    "start_sec",
                    "end_sec",
                    "ppg_start",
                    "ppg_end",
                    "resp",
                    "spo2",
                ]
            )
            for (
                window_idx,
                start_sec,
                end_sec,
                ppg_start,
                ppg_end,
                resp_target,
                spo2_target,
            ) in metadata_rows:
                writer.writerow(
                    [
                        f"bidmc{record_id}",
                        window_idx,
                        start_sec,
                        end_sec,
                        ppg_start,
                        ppg_end,
                        resp_target,
                        spo2_target,
                    ]
                )
        print(f"bidmc{record_id}: saved {len(ppg_save)} windows to {out_dir}")


_butppg_FS_NATIVE = 30
_butppg_SEGMENT_SECONDS = 10
_butppg_SEGMENT_SAMPLES = _butppg_FS_NATIVE * _butppg_SEGMENT_SECONDS
_butppg_TARGETS = ("hr", "sbp", "dbp", "spo2", "glyc")


def _butppg_as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _butppg_blood_pressure(value):
    values = re.findall("[-+]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)", value or "")
    if len(values) < 2:
        return (np.nan, np.nan)
    return (float(values[0]), float(values[1]))


def _butppg_load_metadata(data_root):
    labels_by_id = {}
    with (data_root / "subject-info.csv").open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            sbp, dbp = _butppg_blood_pressure(row["Blood pressure [mmHg]"])
            labels_by_id[row["ID"]] = {
                "sbp": sbp,
                "dbp": dbp,
                "spo2": _butppg_as_float(row["SpO2 [%]"]),
                "glyc": _butppg_as_float(row["Glycaemia [mmol/l]"]),
            }
    records_by_subject = defaultdict(list)
    with (data_root / "quality-hr-ann.csv").open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if int(row["Quality"]) != 1:
                continue
            record_id = row["ID"]
            labels = labels_by_id.get(record_id, {})
            records_by_subject[record_id[:3]].append(
                {
                    "record_id": record_id,
                    "hr": _butppg_as_float(row["HR"]),
                    "sbp": labels.get("sbp", np.nan),
                    "dbp": labels.get("dbp", np.nan),
                    "spo2": labels.get("spo2", np.nan),
                    "glyc": labels.get("glyc", np.nan),
                }
            )
    return records_by_subject


def _butppg_load_ppg(data_root, record_id):
    path = data_root / record_id / f"{record_id}_PPG.dat"
    header_path = data_root / record_id / f"{record_id}_PPG.hea"
    ppg = np.fromfile(path, dtype="<i2").astype(np.float32)
    with header_path.open("r") as f:
        lines = [line.strip() for line in f if line.strip()]
    first = lines[0].split()
    n_channels = int(first[1])
    n_samples = int(first[3])
    if (
        n_channels == _butppg_SEGMENT_SAMPLES
        and n_samples == 1
        and (len(ppg) == _butppg_SEGMENT_SAMPLES)
    ):
        return ppg
    if n_samples == _butppg_SEGMENT_SAMPLES and len(ppg) == n_channels * n_samples:
        channel_names = [line.split()[-1] for line in lines[1 : 1 + n_channels]]
        channel_idx = channel_names.index("PPG_G") if "PPG_G" in channel_names else 0
        return ppg.reshape(n_samples, n_channels)[:, channel_idx]
    if len(ppg) != _butppg_SEGMENT_SAMPLES:
        raise ValueError(
            f"Expected {_butppg_SEGMENT_SAMPLES} PPG samples in {path}, got {len(ppg)}"
        )
    return ppg


def preprocbutppg(cfg):
    dataset_cfg = cfg["butppg"]
    data_root = Path(dataset_cfg["root"])
    save_root = Path(dataset_cfg["save_root"])
    dataset_name = dataset_cfg["name"]
    target_fs = int(cfg.get("common", {}).get("target_fs", _butppg_FS_NATIVE))
    low_f = dataset_cfg.get("low_f", -1)
    high_f = dataset_cfg.get("high_f", -1)
    for subject, records in sorted(_butppg_load_metadata(data_root).items()):
        ppg_segments = []
        target_values = {target: [] for target in _butppg_TARGETS}
        metadata_rows = []
        for record in sorted(records, key=lambda item: item["record_id"]):
            ppg = _butppg_load_ppg(data_root, record["record_id"])
            if low_f > 0 and high_f > 0:
                ppg = butterworth_filter(ppg, low_f, high_f, _butppg_FS_NATIVE)
            ppg = _resample_signal(ppg, _butppg_FS_NATIVE, target_fs)
            if not np.isfinite(ppg).all():
                raise ValueError(f"Non-finite PPG values in {record['record_id']}")
            ppg_segments.append(ppg)
            for target in _butppg_TARGETS:
                target_values[target].append(record[target])
            metadata_rows.append(record)
        ppg_save = np.stack(ppg_segments).astype(np.float32)
        out_dir = save_root / f"fs_{target_fs}" / dataset_name / subject
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "ppg.npy", ppg_save)
        for target in _butppg_TARGETS:
            np.save(
                out_dir / f"{target}.npy",
                np.asarray(target_values[target], dtype=np.float32),
            )
        save_stats(
            out_dir,
            {
                "subject": subject,
                "ppg_mean": float(ppg_save.mean()),
                "ppg_std": float(ppg_save.std()),
                "native_fs": _butppg_FS_NATIVE,
                "target_fs": target_fs,
                "segment_len_s": _butppg_SEGMENT_SECONDS,
            },
        )
        with (out_dir / "metadata.csv").open("w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["subject", "record_id", "quality", *_butppg_TARGETS]
            )
            writer.writeheader()
            for record in metadata_rows:
                writer.writerow({"subject": subject, "quality": 1, **record})
        print(
            f"{subject}: saved {len(ppg_save)} quality-approved segments to {out_dir}"
        )


_realworldppg_FS_NATIVE = 50
_realworldppg_N_PPG_SAMPLES = 300
_realworldppg_SERIAL_ROW = 303
_realworldppg_ID_ROW = 305
_realworldppg_MAIN_NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
}
_realworldppg_REL_NS = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
)
_realworldppg_PKG_REL_NS = (
    "http://schemas.openxmlformats.org/package/2006/relationships"
)


def _realworldppg_column_index(cell_ref):
    index = 0
    for char in "".join((char for char in cell_ref if char.isalpha())):
        index = index * 26 + ord(char) - ord("A") + 1
    return index - 1


def _realworldppg_sheet_path(archive, sheet_name):
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    rel_id = next(
        (
            sheet.attrib[f"{{{_realworldppg_REL_NS}}}id"]
            for sheet in workbook.findall(
                "main:sheets/main:sheet", _realworldppg_MAIN_NS
            )
            if sheet.attrib["name"] == sheet_name
        )
    )
    relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    target = next(
        (
            relationship.attrib["Target"]
            for relationship in relationships.findall(
                f"{{{_realworldppg_PKG_REL_NS}}}Relationship"
            )
            if relationship.attrib["Id"] == rel_id
        )
    )
    return target if target.startswith("xl/") else f"xl/{target.lstrip('/')}"


def _realworldppg_load_records(workbook_path):
    with ZipFile(workbook_path) as archive:
        root = ET.fromstring(archive.read(_realworldppg_sheet_path(archive, "Sheet1")))
    cells_by_row = {}
    for cell in root.findall("main:sheetData/main:row/main:c", _realworldppg_MAIN_NS):
        ref = cell.attrib["r"]
        row = int("".join((char for char in ref if char.isdigit())))
        value = cell.findtext("main:v", default="", namespaces=_realworldppg_MAIN_NS)
        if value:
            cells_by_row.setdefault(row, {})[_realworldppg_column_index(ref)] = value
    records = []
    for column, serial_value in cells_by_row.get(_realworldppg_SERIAL_ROW, {}).items():
        participant_value = cells_by_row.get(_realworldppg_ID_ROW, {}).get(column)
        waveform_values = [
            cells_by_row.get(row, {}).get(column)
            for row in range(1, _realworldppg_N_PPG_SAMPLES + 1)
        ]
        if participant_value is None or any(
            (value is None for value in waveform_values)
        ):
            continue
        serial = int(float(serial_value))
        participant_id = int(float(participant_value))
        ppg = np.asarray(waveform_values, dtype=np.float32)
        if not np.isfinite(ppg).all():
            continue
        records.append((serial, participant_id, ppg))
    if not records:
        raise ValueError(f"No serial/ID-aligned records found in {workbook_path}")
    return records


def preprocrealworldppg(cfg):
    dataset_cfg = cfg["realworldppg"]
    data_root = Path(dataset_cfg["root"])
    save_root = Path(dataset_cfg["save_root"])
    dataset_name = dataset_cfg["name"]
    target_fs = int(cfg.get("common", {}).get("target_fs", _realworldppg_FS_NATIVE))
    for split in ("train", "test"):
        workbook_path = data_root / f"{split}8.xlsx"
        for serial, participant_id, ppg in _realworldppg_load_records(workbook_path):
            ppg = _resample_signal(ppg, _realworldppg_FS_NATIVE, target_fs)
            record_id = f"{split}-{serial:04d}"
            out_dir = save_root / f"fs_{target_fs}" / dataset_name / record_id
            out_dir.mkdir(parents=True, exist_ok=True)
            np.save(out_dir / "ppg.npy", ppg[None, :])
            np.save(out_dir / "id.npy", np.asarray([participant_id], dtype=np.int64))
            save_stats(
                out_dir,
                {
                    "subject": record_id,
                    "split": split,
                    "serial": serial,
                    "participant_id": participant_id,
                    "native_fs": _realworldppg_FS_NATIVE,
                    "target_fs": target_fs,
                    "ppg_mean": float(ppg.mean()),
                    "ppg_std": float(ppg.std()),
                },
            )
            with (out_dir / "metadata.csv").open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "subject",
                        "split",
                        "serial",
                        "participant_id",
                        "start_sec",
                        "end_sec",
                    ]
                )
                writer.writerow(
                    [
                        record_id,
                        split,
                        serial,
                        participant_id,
                        0.0,
                        _realworldppg_N_PPG_SAMPLES / _realworldppg_FS_NATIVE,
                    ]
                )
        print(
            f"{split}: saved {len(_realworldppg_load_records(workbook_path))} RealWorldPPG records"
        )


_wildppg_DEFAULT_FS = 128


def _wildppg_bandpass(data, low_hz, high_hz, fs, order):
    return sosfiltfilt(
        butter(order, [low_hz, high_hz], btype="band", fs=fs, output="sos"), data
    )


def _wildppg_pan_peak_detect(detection, fs):
    fs = int(round(fs))
    min_distance = int(0.25 * fs)
    signal_peaks = [0]
    noise_peaks = []
    threshold_1 = 0.0
    threshold_2 = 0.0
    spki = 0.0
    npki = 0.0
    rr_missed = 0
    peak_indices = []
    peaks = find_peaks(detection, distance=min_distance)[0]
    threshold_weight = 0.125
    for peak_index, peak in enumerate(peaks):
        if peak > 4 * fs and threshold_1 > np.max(detection[peak - 4 * fs : peak]):
            new_spki = np.max(detection[peak - 4 * fs : peak])
            npki = (
                min(
                    npki * new_spki / spki,
                    np.percentile(detection[peak - 4 * fs : peak], 80),
                )
                if spki
                else 0.0
            )
            spki = new_spki
            threshold_1 = npki + 0.25 * (spki - npki)
            threshold_2 = 0.5 * threshold_1
        if detection[peak] > threshold_1 and peak - signal_peaks[-1] > 0.3 * fs:
            signal_peaks.append(peak)
            peak_indices.append(peak_index)
            spki = (
                threshold_weight * detection[signal_peaks[-1]]
                + (1 - threshold_weight) * spki
            )
            if rr_missed and signal_peaks[-1] - signal_peaks[-2] > rr_missed:
                missed_candidates = peaks[peak_indices[-2] + 1 : peak_indices[-1]]
                missed_candidates = [
                    candidate
                    for candidate in missed_candidates
                    if candidate - signal_peaks[-2] > min_distance
                    and signal_peaks[-1] - candidate > min_distance
                    and (detection[candidate] > threshold_2)
                ]
                if missed_candidates:
                    missed_peak = max(
                        missed_candidates, key=lambda candidate: detection[candidate]
                    )
                    signal_peaks.append(signal_peaks[-1])
                    signal_peaks[-2] = missed_peak
        else:
            noise_peaks.append(peak)
            npki = (
                threshold_weight * detection[noise_peaks[-1]]
                + (1 - threshold_weight) * npki
            )
        threshold_1 = npki + 0.25 * (spki - npki)
        threshold_2 = 0.5 * threshold_1
        if len(signal_peaks) > 8:
            rr_missed = int(1.66 * np.mean(np.diff(signal_peaks[-9:])))
    return np.asarray(signal_peaks[1:], dtype=np.int64)


def _wildppg_official_rr_intervals(ecg, fs):
    fs = int(round(fs))
    filtered_ecg = _wildppg_bandpass(ecg, 5, 15, fs, order=1)
    derivative_squared = np.diff(filtered_ecg) ** 2
    moving_average = uniform_filter1d(derivative_squared, size=int(0.15 * fs))
    max_values = maximum_filter1d(filtered_ecg, size=int(0.15 * fs))[:-1] / 400
    moving_average = np.minimum(moving_average, max_values)
    moving_average[: int(0.15 * fs * 2)] = 0
    search_radius = int(0.15 * fs)
    peak_signal = _wildppg_bandpass(ecg, 7.5, 20, fs, order=1)
    approximate_peaks = _wildppg_pan_peak_detect(moving_average, fs)
    refined_peaks = []
    for peak in approximate_peaks:
        start = max(0, peak - search_radius)
        end = min(len(peak_signal), peak + search_radius + 1)
        refined = start + np.argmax(peak_signal[start:end])
        local_start = max(0, refined - 2)
        local_end = min(len(ecg), refined + 3)
        refined_peaks.append(local_start + np.argmax(ecg[local_start:local_end]))
    good_peaks = []
    rr_intervals = []
    rr_midpoints = []
    for start_index in range(max(0, len(refined_peaks) - 4)):
        beat_window = refined_peaks[start_index : start_index + 5]
        if len(beat_window) < 5:
            break
        intervals = np.diff(beat_window)
        hr = 60 / (np.mean(intervals) / fs)
        if intervals.min() > intervals.max() * 0.75 and 35 < hr < 185:
            for peak in beat_window:
                if not good_peaks or peak > good_peaks[-1]:
                    good_peaks.append(peak)
                    if len(good_peaks) > 1:
                        rr = good_peaks[-1] - good_peaks[-2]
                        if rr < intervals.min() / 0.75 and rr > intervals.max() * 0.75:
                            rr_intervals.append(rr)
                            rr_midpoints.append((good_peaks[-1] + good_peaks[-2]) / 2)
    return (
        np.asarray(rr_intervals, dtype=np.float64),
        np.asarray(rr_midpoints, dtype=np.float64),
    )


def _wildppg_get_trace(participant, body_site, trace_name):
    sensor = getattr(participant[body_site], trace_name)
    return (
        np.asarray(sensor.v, dtype=np.float32).squeeze(),
        float(np.asarray(sensor.fs).squeeze()),
    )


def preprocwildppg(cfg):
    dataset_cfg = cfg["wildppg"]
    data_root = Path(dataset_cfg["root"])
    save_root = Path(dataset_cfg["save_root"])
    dataset_name = dataset_cfg["name"]
    target_fs = int(cfg.get("common", {}).get("target_fs", _wildppg_DEFAULT_FS))
    window_len_s = int(dataset_cfg.get("window_len_s", 8))
    stride_s = int(dataset_cfg.get("stride_s", window_len_s))
    low_f = dataset_cfg.get("low_f", -1)
    high_f = dataset_cfg.get("high_f", -1)
    window_samples = window_len_s * target_fs
    stride_samples = stride_s * target_fs
    for path in sorted(data_root.glob("WildPPG_Part_*.mat")):
        subject = path.stem.removeprefix("WildPPG_Part_")
        participant = loadmat(path, struct_as_record=False, squeeze_me=True)
        ppg, ppg_fs = _wildppg_get_trace(participant, "wrist", "ppg_g")
        ecg, ecg_fs = _wildppg_get_trace(participant, "sternum", "ecg")
        if low_f > 0 and high_f > 0:
            ppg = butterworth_filter(ppg, low_f, high_f, ppg_fs)
        ppg = _resample_signal(ppg, int(ppg_fs), target_fs)
        rr_intervals, rr_midpoints = _wildppg_official_rr_intervals(ecg, ecg_fs)
        ppg_windows = []
        hr_targets = []
        metadata_rows = []
        for window_index, ppg_start in enumerate(
            range(0, len(ppg) - window_samples + 1, stride_samples)
        ):
            ppg_end = ppg_start + window_samples
            window = ppg[ppg_start:ppg_end]
            if not np.isfinite(window).all():
                continue
            start_s = ppg_start / target_fs
            end_s = ppg_end / target_fs
            midpoint_mask = (rr_midpoints > start_s * ecg_fs) & (
                rr_midpoints < end_s * ecg_fs
            )
            window_rrs = rr_intervals[midpoint_mask]
            hr = np.nan
            if len(window_rrs) > 1:
                hr = float(60 * len(window_rrs) / (window_rrs.sum() / ecg_fs))
            ppg_windows.append(window)
            hr_targets.append(hr)
            metadata_rows.append(
                (window_index, start_s, end_s, ppg_start, ppg_end, hr, len(window_rrs))
            )
        if not ppg_windows:
            raise ValueError(f"{subject}: no finite green PPG windows")
        ppg_save = np.asarray(ppg_windows, dtype=np.float32)
        hr_save = np.asarray(hr_targets, dtype=np.float32)
        out_dir = save_root / f"fs_{target_fs}" / dataset_name / subject
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "ppg.npy", ppg_save)
        np.save(out_dir / "hr.npy", hr_save)
        save_stats(
            out_dir,
            {
                "subject": subject,
                "ppg_mean": float(ppg_save.mean()),
                "ppg_std": float(ppg_save.std()),
                "native_ppg_fs": ppg_fs,
                "native_ecg_fs": ecg_fs,
                "target_fs": target_fs,
                "window_len_s": window_len_s,
                "stride_s": stride_s,
                "ppg_channel": "green",
            },
        )
        with (out_dir / "metadata.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "subject",
                    "window_idx",
                    "start_s",
                    "end_s",
                    "ppg_start",
                    "ppg_end",
                    "hr",
                    "n_rr_intervals",
                ]
            )
            for row in metadata_rows:
                writer.writerow([subject, *row])
        print(
            f"{subject}: saved {len(ppg_save)} green PPG windows ({np.isfinite(hr_save).sum()} finite HR labels)"
        )
        del participant
        gc.collect()


def find_raw_root(dataset, directory):
    """Find the dataset inside either a downloaded or user-extracted directory."""
    patterns = {
        "dalia": "S1/S1.pkl",
        "wesad": "S2/S2.pkl",
        "dreamt": "data_64Hz",
        "ecsmp": "scale.xlsx",
        "ppgbp": "PPG-BP dataset.xlsx",
        "bidmc": "bidmc_csv",
        "butppg": "subject-info.csv",
        "realworldppg": "train8.xlsx",
        "wildppg": "WildPPG_Part_*.mat",
    }
    directory = Path(directory).resolve()
    pattern = patterns[dataset]
    matches = sorted(directory.glob(pattern)) or sorted(directory.rglob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"{dataset}: cannot find {pattern} under {directory}. Run download or supply --raw-root."
        )
    root = (
        matches[0].parent.parent if dataset in ("dalia", "wesad") else matches[0].parent
    )
    return root


def preprocess(datasets, raw, output, rates, raw_root=None):
    import hashlib
    import shutil
    import tempfile

    from .datasets import DATASETS

    if raw_root and len(datasets) != 1:
        raise ValueError(
            "--raw-root is for one dataset; --raw contains dataset subdirectories"
        )
    for dataset in datasets:
        directory = Path(raw_root) if raw_root else Path(raw) / dataset
        root = find_raw_root(dataset, directory)
        raw_files = [
            (str(path.relative_to(root)), path.stat().st_size, path.stat().st_mtime_ns)
            for path in sorted(root.rglob("*"))
            if path.is_file()
            and path.suffix.lower()
            in {".pkl", ".csv", ".mat", ".xlsx", ".txt", ".dat", ".hea"}
        ]
        input_signature = hashlib.sha256(json.dumps(raw_files).encode()).hexdigest()
        parser_signature = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        for rate in rates:
            output = Path(output).resolve()
            output.mkdir(parents=True, exist_ok=True)
            destination = output / f"fs_{rate}" / DATASETS[dataset]
            config = {
                "inputs_sha256": input_signature,
                "parser_sha256": parser_signature,
                "common": {"target_fs": rate},
                dataset: {
                    "name": DATASETS[dataset],
                    "root": str(root),
                    "window_len_s": 8 if dataset in ("dalia", "wildppg") else 10,
                    "stride_s": 2
                    if dataset == "dalia"
                    else 8
                    if dataset == "wildppg"
                    else 10,
                    "low_f": -1,
                    "high_f": -1,
                    "flatline_std_frac": 0.05,
                },
            }
            marker = destination / "preprocessing.json"
            if destination.exists():
                if marker.exists() and json.loads(marker.read_text()) == config:
                    print(f"Already prepared: {destination}")
                    continue
                raise FileExistsError(
                    f"{destination} already exists without matching preprocessing metadata; choose a new --output"
                )
            with tempfile.TemporaryDirectory(
                prefix=".preproc-", dir=output
            ) as temporary:
                parser_config = {
                    "common": config["common"],
                    dataset: dict(config[dataset], save_root=temporary),
                }
                globals()[f"preproc{dataset}"](parser_config)
                prepared = Path(temporary) / f"fs_{rate}" / DATASETS[dataset]
                if not prepared.exists() or not list(prepared.glob("*/ppg.npy")):
                    raise ValueError(
                        f"No {dataset} recordings were prepared from {root}"
                    )
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(prepared), str(destination))
                marker.write_text(json.dumps(config, indent=2))
                print(f"Prepared: {destination}")
