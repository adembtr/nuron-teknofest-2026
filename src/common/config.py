#!/usr/bin/env python3
"""Merkezi konfigürasyon yükleyici. Tüm modüller buradan okur."""
import os
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG_PATH = os.path.join(ROOT, "config", "competition.yaml")


def load(path: str = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


CFG = load()


def model_root(modality: str) -> str:
    """DÜZ models/ dizininin mutlak yolu (tüm ağırlıklar tek klasörde, prefix'li).
    modality yalnız doğrulama için; ağırlık adları çağıran tarafta '{modality}_' önekiyle
    kurulur (ör. models/rgb_yolo.pt, models/termal_padim_uap.pth)."""
    assert modality in ("rgb", "termal"), f"geçersiz modalite: {modality}"
    return os.path.join(ROOT, CFG["paths"]["models_root"])


def abspath(rel: str) -> str:
    return rel if os.path.isabs(rel) else os.path.join(ROOT, rel)
