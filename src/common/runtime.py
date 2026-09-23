#!/usr/bin/env python3
"""
Çalışma-zamanı klasör düzeni ve kare yönetimi.

Düzen (NURON_DRONE kökünde):
  frames/new/     ← sunucudan gelen EN SON kare buraya düşer
  frames/old/     ← işlenen kareler buraya taşınır (SİLİNMEZ, birikir)
  referance/      ← oturum başı inen referans görselleri (reference_1.webp ...)
  crop/           ← referanslar bizim modelle (HQ-SAM) anında kırpılır → buraya
  logs/           ← sonuç/olay logları
  runtime/        ← GPS IPC (pozlar), geçici durum

Kural: eski kareler silinmez (kullanıcı isteği) → hata ayıklama/tekrar test için kalır.
"""
import os
import time
import shutil
import cv2
import numpy as np

from src.common.config import ROOT
from src.common import kosu as KOSU

# 2026-09-17 — KOŞU KLASÖRÜ: kare/referans klasörleri artık kosular/folder<N>/ altında.
# Böylece art arda koşularda birikme ve üzerine yazma olmaz (bkz. src/common/kosu.py).
# TEMBEL: yollar ilk KULLANIMDA çözülür — modülü import etmek klasör OLUŞTURMAZ
# (GPS worker bu modülü yalnızca RUNTIME sabiti için import eder, koşu klasörü açmamalı).
CROP_DIR = os.path.join(ROOT, "crop")
RUNTIME = os.path.join(ROOT, "runtime")       # worker ile dosya-IPC → SABİT kalmalı
LOGS = os.path.join(ROOT, "logs")


def frames_dir() -> str:
    return KOSU.alt("frames")


def new_dir() -> str:
    return os.path.join(frames_dir(), "new")


def old_dir() -> str:
    return os.path.join(frames_dir(), "old")


def ref_dir() -> str:
    return KOSU.alt("referance")              # kullanıcı yazımı (referance)


def ensure_dirs():
    # NOT: CROP_DIR (crop/) yalnızca eski orchestrator akışına aitti; sunucu (connect)
    # akışı kullanmıyor (referanslar _images/ + offline_data/'ya gider) → oluşturulmaz.
    for d in (new_dir(), old_dir(), ref_dir(), RUNTIME, LOGS):
        os.makedirs(d, exist_ok=True)


def _safe(name: str) -> str:
    """Dosya adı güvenli hale getir (url/benzersiz isim → dosya adı)."""
    keep = "-_.() "
    s = "".join(c if (c.isalnum() or c in keep) else "_" for c in str(name))
    return s.strip("_") or f"frame_{int(time.time()*1000)}"


def save_new_frame(image_bgr: np.ndarray, frame_key: str) -> str:
    """EN SON kareyi frames/new/ altına yaz. Bir önceki new kareyi old'a taşı.
    frame_key = benzersiz kare adı (sunucudaki video_name/url'den). Dönüş: yeni dosya yolu."""
    ensure_dirs()
    # önceki new içeriğini old'a taşı (birikir, silinmez)
    NEW, OLD = new_dir(), old_dir()
    for f in os.listdir(NEW):
        if f == ".gitkeep":
            continue
        src = os.path.join(NEW, f)
        if os.path.isfile(src):
            dst = os.path.join(OLD, f)
            if os.path.exists(dst):                      # isim çakışması → zaman ekle
                base, ext = os.path.splitext(f)
                dst = os.path.join(OLD, f"{base}_{int(time.time()*1000)}{ext}")
            shutil.move(src, dst)
    path = os.path.join(NEW, f"{_safe(frame_key)}.jpg")
    cv2.imwrite(path, image_bgr)
    return path


def list_references() -> list[str]:
    """referance/ altındaki referans görsel yolları (webp/png/jpg), sıralı."""
    ensure_dirs()
    exts = (".webp", ".png", ".jpg", ".jpeg")
    RD = ref_dir()
    fs = [os.path.join(RD, f) for f in sorted(os.listdir(RD))
          if f.lower().endswith(exts)]
    return fs


def clear_runtime():
    """Oturum başında GPS poz dosyalarını temizle (kareler DEĞİL)."""
    ensure_dirs()
    for f in os.listdir(RUNTIME):
        p = os.path.join(RUNTIME, f)
        if os.path.isfile(p):
            os.remove(p)
