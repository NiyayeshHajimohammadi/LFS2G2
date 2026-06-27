"""Minimal YAML config with dotted attribute access and ``_base_`` inheritance.

Uses OmegaConf if installed (nicer CLI overrides); otherwise falls back to a small AttrDict so the
package has no hard dependency on it.
"""
from __future__ import annotations

import os
from typing import Any, List

import yaml


class AttrDict(dict):
    """dict with attribute access; nested dicts are wrapped recursively."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return AttrDict(value) if isinstance(value, dict) else value

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def get(self, key, default=None):
        value = super().get(key, default)
        return AttrDict(value) if isinstance(value, dict) else value


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_raw(path: str) -> dict:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    base = data.pop("_base_", None)
    if base:
        base_path = os.path.join(os.path.dirname(os.path.abspath(path)), base)
        data = _deep_merge(_load_raw(base_path), data)
    return data


def as_container(cfg) -> dict:
    """Convert any config (AttrDict / OmegaConf / dict) to a plain nested dict."""
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(cfg):
            return OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        pass
    return dict(cfg)


def as_config(obj):
    """Wrap a plain dict back into a usable config (OmegaConf if available, else AttrDict)."""
    try:
        from omegaconf import OmegaConf

        return OmegaConf.create(obj)
    except Exception:
        return AttrDict(obj)


def load_config(path: str, overrides: List[str] | None = None):
    """Load a config. ``overrides`` are ``dotted.key=value`` strings (parsed as YAML scalars)."""
    raw = _load_raw(path)
    for ov in overrides or []:
        key, _, val = ov.partition("=")
        node = raw
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    try:
        from omegaconf import OmegaConf

        return OmegaConf.create(raw)
    except Exception:
        return AttrDict(raw)
