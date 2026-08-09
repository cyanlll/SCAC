from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml


ALIASES = {
    "act": "act",
    "activitynet": "act",
    "activitynet-captions": "act",
    "cha": "cha",
    "charades": "cha",
    "charades-sta": "cha",
    "didemo": "didemo",
}

DATASET_LOCALIZATION = {
    "act": {"stride": 50, "max_stride_factor": 1.0},
    "cha": {"stride": 23, "max_stride_factor": 0.5},
    "didemo": {"stride": 21, "max_stride_factor": 0.5},
}


def normalize_dataset_name(name: str) -> str:
    key = name.strip().lower()
    if key not in ALIASES:
        raise ValueError(f"Unsupported dataset: {name}. Choose from act/cha/didemo.")
    return ALIASES[key]


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg = _expand(cfg)

    dataset = normalize_dataset_name(cfg.get("dataset", {}).get("name", "act"))
    cfg.setdefault("dataset", {})["name"] = dataset
    defaults = DATASET_LOCALIZATION[dataset]
    cfg.setdefault("localization", {})
    cfg["localization"].setdefault("stride", defaults["stride"])
    cfg["localization"].setdefault("max_stride_factor", defaults["max_stride_factor"])
    cfg.setdefault("runtime", {}).setdefault("device", "cuda:0")
    cfg["runtime"].setdefault("top_k", 200)
    return cfg


def require_path(cfg: Dict[str, Any], section: str, key: str) -> Path:
    value = cfg.get(section, {}).get(key)
    if not value:
        raise ValueError(f"Missing config value: {section}.{key}")
    return Path(value)
