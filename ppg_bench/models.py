"""An encoder adapter, loaded from a Python file or an installed module."""

import hashlib
import importlib
import importlib.util
import sys
import threading
from pathlib import Path

import yaml


_external_import_lock = threading.RLock()


def external_module(source_root, module):
    """Import an encoder from a user-provided upstream checkout."""
    if not source_root:
        raise ValueError("Set init.source_root to your downloaded model repository")
    root = Path(source_root).resolve()
    source = root.joinpath(*module.split(".")).with_suffix(".py")
    if not source.is_file():
        raise FileNotFoundError(f"Missing upstream model source: {source}")
    with _external_import_lock:
        sys.path.insert(0, str(root))
        try:
            loaded = importlib.import_module(module)
        finally:
            sys.path.remove(str(root))
        if Path(loaded.__file__).resolve() != source:
            raise ValueError(f"{module} is already loaded from another checkout; restart Python")
    return loaded


class ModelPort:
    def __init__(self, model_init_cfg):
        self.model_init_cfg = model_init_cfg

    def load_encoder(self):
        """Return a fresh torch.nn.Module with pretrained weights loaded."""
        raise NotImplementedError

    def prepare_batch(self, batch, model_cfg, device):
        """Apply your model's input formatting, filtering and normalization."""
        return batch["x"].float().to(device)


def load_models(config_path):
    path = Path(config_path).resolve()
    configs = yaml.safe_load(path.read_text())
    configs = configs if isinstance(configs, list) else [configs]
    result = []
    for config in configs:
        if not isinstance(config, dict):
            raise ValueError("Model YAML must contain a mapping or a list of mappings")
        unknown = set(config) - {"name", "port", "fs", "embedding_dim", "init"}
        if unknown:
            raise ValueError(f"Unknown model keys: {sorted(unknown)}")
        config = dict(config)
        for key in ("name", "port", "fs", "embedding_dim"):
            if key not in config:
                raise ValueError(f"Model YAML requires {key}")
        if any(
            type(config[key]) is not int or config[key] <= 0
            for key in ("fs", "embedding_dim")
        ):
            raise ValueError("fs and embedding_dim must be positive integers")
        if not isinstance(config["name"], str) or not config["name"].strip():
            raise ValueError("Model name must be a nonempty string")
        if config["name"] == "TrainMean" or config["name"].startswith("TrainMean/"):
            raise ValueError("TrainMean is reserved for the benchmark control")
        if not isinstance(config["port"], str) or ":" not in config["port"]:
            raise ValueError("port must be file.py:Class or installed_module:Class")
        module, attribute = config["port"].rsplit(":", 1)
        if module.endswith(".py"):
            source = (path.parent / module).resolve()
            config["port"] = f"{source}:{attribute}"
            config["port_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        if not isinstance(config.get("init", {}), dict):
            raise ValueError("init must be a mapping of adapter settings")
        config["init"] = dict(config.get("init", {}))
        if "source_root" in config["init"]:
            source_root = (path.parent / config["init"]["source_root"]).resolve()
            if not source_root.is_dir():
                raise FileNotFoundError(f"Download the upstream model source first: {source_root}")
            config["init"]["source_root"] = str(source_root)
            config["source_sha256"] = hashlib.sha256(b"".join(
                str(p.relative_to(source_root)).encode() + b"\0" + p.read_bytes()
                for p in sorted(source_root.rglob("*.py"))
            )).hexdigest()
        if "ckpt_path" in config["init"]:
            checkpoint = (path.parent / config["init"]["ckpt_path"]).resolve()
            if not checkpoint.exists():
                raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
            config["init"]["ckpt_path"] = str(checkpoint)
        config["init"]["target_fs"] = int(config["fs"])
        result.append(config)
    if len({item["name"] for item in result}) != len(result):
        raise ValueError("Model names must be unique")
    return result


def get_port(config):
    module, attribute = config["port"].rsplit(":", 1)
    if module.endswith(".py"):
        source = Path(module).read_bytes()
        digest = hashlib.sha256(source).hexdigest()
        if digest != config.get("port_sha256", digest):
            raise ValueError(
                "Adapter changed after configuration was loaded; reload the model configuration"
            )
        name = (
            "ppg_bench_port_"
            + hashlib.sha256((module + digest).encode()).hexdigest()[:12]
        )
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name, module)
            loaded = importlib.util.module_from_spec(spec)
            sys.modules[name] = loaded
            exec(compile(source, module, "exec"), loaded.__dict__)
        loaded = sys.modules[name]
    else:
        loaded = importlib.import_module(module)
    try:
        factory = getattr(loaded, attribute)
    except AttributeError as error:
        raise ValueError(f"Port {attribute} does not exist in {module}") from error
    port = factory(config["init"])
    if not all(
        callable(getattr(port, method, None))
        for method in ("load_encoder", "prepare_batch")
    ):
        raise ValueError("Port must implement load_encoder and prepare_batch")
    return port
