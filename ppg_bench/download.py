"""Download the raw files used by the nine benchmark datasets."""

import base64
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import tarfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path


SOURCES = {
    "dalia": "https://archive.ics.uci.edu/dataset/495/ppg+dalia",
    "wesad": "https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html",
    "dreamt": "https://physionet.org/content/dreamt/2.1.0/",
    "ecsmp": "https://data.mendeley.com/datasets/vn5nknh3mn/2",
    "ppgbp": "https://figshare.com/articles/dataset/PPG-BP_Database_zip/5459299/5",
    "bidmc": "https://physionet.org/content/bidmc/1.0.0/",
    "butppg": "https://physionet.org/content/butppg/2.0.0/",
    "realworldppg": "https://data.mendeley.com/datasets/yynb8t9x3d/1",
    "wildppg": "https://siplab.org/projects/WildPPG",
}


def open_url(url, headers=None):
    return urllib.request.urlopen(
        urllib.request.Request(
            url, headers={"User-Agent": "PPG-Bench/0.1", **(headers or {})}
        ),
        timeout=120,
    )


def file_checksum(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch(url, destination, size=None, checksum=None, headers=None):
    """Resume partial transfers, validate publisher checksums, then rename."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and (size is None or destination.stat().st_size == size):
        if checksum is None or file_checksum(destination) == checksum:
            return destination
    partial = destination.with_name(destination.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    if size is not None and offset == size:
        if checksum is None or file_checksum(partial) == checksum:
            partial.replace(destination)
            return destination
        partial.unlink()
        offset = 0
    request_headers = dict(headers or {})
    if offset:
        request_headers["Range"] = f"bytes={offset}-"
    print(f"Downloading {destination}", flush=True)
    with open_url(url, request_headers) as response:
        if "text/html" in response.headers.get("Content-Type", ""):
            raise ValueError(f"Expected dataset bytes, received a web page: {url}")
        resumed = offset and response.status == 206
        total = size or (
            int(response.headers["Content-Length"]) + (offset if resumed else 0)
            if response.headers.get("Content-Length")
            else None
        )
        content_range = response.headers.get("Content-Range")
        if response.status == 206 and content_range:
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
            if not match or int(match[1]) != (offset if resumed else 0):
                raise ValueError(f"Unexpected download range: {content_range}")
            total = size or int(match[3])
        with partial.open("ab" if resumed else "wb") as stream:
            shutil.copyfileobj(response, stream, length=1024 * 1024)
    if total and partial.stat().st_size != total:
        raise ValueError(
            f"Incomplete download: {destination}. Run the command again to resume."
        )
    if checksum and file_checksum(partial) != checksum:
        partial.unlink()
        raise ValueError(f"Checksum mismatch: {destination}. Retry the download.")
    partial.replace(destination)
    return destination


def unpack(archive, destination):
    """Extract ZIP/TAR archives, including ZIP archives nested by publishers."""
    archive, destination = Path(archive), Path(destination).resolve()
    marker = archive.with_name(archive.name + ".extracted")
    if marker.exists():
        return
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as bundle:
            for entry in bundle.infolist():
                target = (destination / entry.filename).resolve()
                if (
                    not target.is_relative_to(destination)
                    or (entry.external_attr >> 16) & 0o170000 == 0o120000
                ):
                    raise ValueError(f"Unsafe archive entry: {entry.filename}")
            bundle.extractall(destination)
            nested = [
                destination / entry.filename
                for entry in bundle.infolist()
                if entry.filename.lower().endswith(".zip")
            ]
        for child in nested:
            unpack(child, child.parent)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as bundle:
            bundle.extractall(destination, filter="data")
    else:
        raise ValueError(
            f"Unsupported archive: {archive}; extract it with your archive tool"
        )
    marker.touch()


def mendeley_files(dataset_id, version, directory, folder="root", folders=None):
    base = f"https://data.mendeley.com/public-api/datasets/{dataset_id}"
    query = urllib.parse.urlencode({"folder_id": folder, "version": version})
    entries = []
    start = 0
    while True:
        with open_url(f"{base}/files?{query}&$start={start}&$limit=1000") as response:
            page = json.load(response)
        entries.extend(page)
        if len(page) < 1000:
            break
        start += 1000
    for entry in entries:
        if (
            dataset_id == "vn5nknh3mn"
            and entry["filename"] != "scale.xlsx"
            and folder == "root"
        ):
            continue
        if (
            dataset_id == "vn5nknh3mn"
            and "E4" in Path(directory).parts
            and entry["filename"] != "BVP.csv"
        ):
            continue
        details = entry["content_details"]
        filename = Path(entry["filename"]).name
        fetch(
            details["download_url"],
            Path(directory) / filename,
            details["size"],
            details.get("sha256_hash"),
        )
    if folders is None:
        with open_url(f"{base}/folders/{version}") as response:
            folders = json.load(response)
    for entry in folders:
        if (entry.get("parent_id") or "root") != folder:
            continue
        name = entry.get("name", entry.get("folder_name", ""))
        if (
            folder == "root"
            and dataset_id == "vn5nknh3mn"
            and name not in ("E4", "EEG_downsample")
        ):
            continue
        if Path(name).name != name or name in ("", ".", ".."):
            raise ValueError(f"Invalid repository folder: {name}")
        mendeley_files(
            dataset_id, version, Path(directory) / name, entry["id"], folders
        )


def download_one(dataset, directory, physionet_user=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if dataset in ("dalia", "wesad", "wildppg"):
        url = {
            "dalia": "https://archive.ics.uci.edu/static/public/495/ppg+dalia.zip",
            "wesad": "https://uni-siegen.sciebo.de/index.php/s/HGdUkoNlW1Ub0Gx/download",
            "wildppg": "https://polybox.ethz.ch/index.php/s/NWTuyNojU7aya1y/download",
        }[dataset]
        unpack(fetch(url, directory / f"{dataset}.zip"), directory)
    elif dataset == "ppgbp":
        with open_url(
            "https://api.figshare.com/v2/articles/5459299/versions/5"
        ) as response:
            entries = json.load(response)["files"]
        entry = next(item for item in entries if item["name"].endswith(".zip"))
        unpack(
            fetch(entry["download_url"], directory / entry["name"], entry["size"]),
            directory,
        )
    elif dataset in ("ecsmp", "realworldppg"):
        identifier, version = (
            ("vn5nknh3mn", 2) if dataset == "ecsmp" else ("yynb8t9x3d", 1)
        )
        mendeley_files(identifier, version, directory)
    elif dataset == "bidmc":
        base = "https://physionet.org/files/bidmc/1.0.0/bidmc_csv/"
        for record in range(1, 54):
            for suffix in ("Signals", "Numerics"):
                name = f"bidmc_{record:02d}_{suffix}.csv"
                fetch(base + name, directory / "bidmc_csv" / name)
    elif dataset == "butppg":
        base = "https://physionet.org/files/butppg/2.0.0/"
        for name in ("subject-info.csv", "quality-hr-ann.csv"):
            fetch(base + name, directory / name)
        with (directory / "quality-hr-ann.csv").open(encoding="utf-8-sig") as stream:
            records = [
                row["ID"] for row in csv.DictReader(stream) if int(row["Quality"]) == 1
            ]
        for record in records:
            for extension in ("dat", "hea"):
                name = f"{record}/{record}_PPG.{extension}"
                fetch(base + name, directory / name)
    elif dataset == "dreamt":
        if not physionet_user:
            raise ValueError(
                f"DREAMT needs approved PhysioNet access: {SOURCES[dataset]}. Then pass --physionet-user YOUR_USERNAME."
            )
        password = os.environ.get("PHYSIONET_PASSWORD")
        if not password:
            if not sys.stdin.isatty():
                raise ValueError(
                    "Set PHYSIONET_PASSWORD for noninteractive DREAMT downloads"
                )
            from getpass import getpass

            password = getpass("PhysioNet password: ")
        credentials = base64.b64encode(f"{physionet_user}:{password}".encode()).decode()
        headers = {"Authorization": f"Basic {credentials}"}
        base = "https://physionet.org/files/dreamt/2.1.0/data_64Hz/"
        with open_url(base, headers) as response:
            listing = response.read().decode()
        names = sorted(set(re.findall(r"S\d+_whole_df\.csv", listing)))
        if not names:
            raise ValueError(
                "No DREAMT files found; verify your PhysioNet access and signed data agreement"
            )
        for name in names:
            fetch(base + name, directory / "data_64Hz" / name, headers=headers)
    (directory / "source.json").write_text(
        json.dumps({"dataset": dataset, "source": SOURCES[dataset]}, indent=2)
    )


def download(datasets, output, physionet_user=None):
    errors = []
    for dataset in datasets:
        try:
            download_one(dataset, Path(output) / dataset, physionet_user)
        except Exception as error:
            errors.append(f"{dataset}: {error}")
            print(errors[-1], file=sys.stderr)
    if errors:
        raise RuntimeError(
            f"{len(errors)} dataset downloads need attention; completed downloads were retained"
        )
