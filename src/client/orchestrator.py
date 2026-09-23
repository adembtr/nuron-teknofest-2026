#!/usr/bin/env python3
"""
ORKESTRATÖR — kalıcı süreç (base env). Tüm G1/G3 modelleri GPU'da BİR KEZ yüklenir,
her karede tekrar kullanılır (yeniden yükleme YOK → kare başı ≤0.5sn hedefi).

Bir kare akışı:
  save_new_frame (frames/new, önceki→frames/old) →
  [G1] YOLO tespit + taşıt hareket (DetectionPipeline) →
  [G1] UAP/UAİ iniş (LandingEstimator, PaDiM) →
  [G3] referans eşleme (ReferenceSession) →
  [G2] GPS (GpsBridge → dpvo worker; AYRI süreç, paralel) →
  ResultPackage (şema) → JSON

VRAM (8GB): tüm modeller aynı anda sığmazsa görev sırası/aç-kapa `enable_*` ile ayarlanır.
GPS ayrı süreçte (dpvo env) olduğu için base env ile PARALEL koşar.
Kare başı her görevin süresi ölçülür (.last_timing) → sıralama/optimizasyon için.
"""
import os
import time
from typing import Optional
import numpy as np

from src.common.config import CFG
from src.common import runtime as RT
from src.common.schema import (FrameInfo, ResultPackage, DetectedObject,
                               DetectedTranslation, UndefinedObject)
from src.task1_detection.pipeline import DetectionPipeline
from src.task1_detection.landing import LandingEstimator, UAP_CLS, UAI_CLS, OBSTACLE_CLS
from src.task3_reference.g3_v2 import ReferenceSessionV2
ReferenceSession = ReferenceSessionV2   # v1 KALDIRILDI (2026-09-01)

# G3 v2 boru hatti (FastSAM + WebSSL + DAM4SAM takip). False -> eski DINOv3 yolu.
#   RGB    : rgb2    · esik 0.44 · 3 ardisik tespitte cipala
#   TERMAL : termal2 · gri banka · gonder 0.24 / cipa 0.45 / 4 ardisik
#            SADECE TAKIP (SAM cipalanmadan kutu gonderilmez)
REFERANS_V2 = True     # v1 yolu kaldirildi; tek yol v2 (rgb2 / termal2)

VEHICLE_CLS = 0


class Orchestrator:
    def __init__(self, modality: str,
                 enable_detection: bool = True,
                 enable_landing: bool = True,
                 enable_reference: bool = True,
                 enable_gps: bool = True,
                 gps_enable_worker: bool = True,
                 reference_schedule: Optional[list] = None,
                 landing_device: str = "cuda",       # PaDiM cihazı — VRAM darsa "cpu"
                 skip_eval_first: Optional[int] = None):
        assert modality in ("rgb", "termal")
        self.modality = modality
        self.enable_detection = enable_detection
        self.enable_landing = enable_landing
        self.enable_reference = enable_reference
        self.enable_gps = enable_gps
        self.skip_eval_first = (skip_eval_first if skip_eval_first is not None
                                else CFG["session"].get("skip_eval_first", 10))
        self.frame_idx = 0
        self.last_timing: dict = {}

        RT.ensure_dirs()
        RT.clear_runtime()

        # ---- G1: tespit + hareket (persistent) ----
        self.pipe = DetectionPipeline(modality) if enable_detection else None
        # ---- G1: iniş (PaDiM, persistent, lazy model; VRAM darsa CPU) ----
        self.landing = LandingEstimator(modality, device=landing_device) if enable_landing else None
        # ---- G3: referans (persistent) ----
        self.ref: Optional[ReferenceSession] = None
        if enable_reference:
            self._bootstrap_reference(reference_schedule)
        # ---- G2: GPS köprüsü (dpvo worker AYRI süreç) ----
        self.gps = None
        if enable_gps:
            from src.client.gps_bridge import GpsBridge
            self.gps = GpsBridge(modality, enable_worker=gps_enable_worker)

    # -------- oturum başı: referansları bizim modelle kırp → crop/, banka yükle --------
    def _bootstrap_reference(self, schedule):
        refs = RT.list_references()
        if not refs:
            print("[orchestrator] referance/ boş → G3 kapalı (referans yok).")
            self.enable_reference = False
            return
        bank_path = os.path.join(RT.CROP_DIR, "ref_bank.npz")   # crop'lar crop/ref_crops'a düşer
        Oturum = ReferenceSessionV2
        sur = "v2"
        print(f"[orchestrator] {len(refs)} referans → HQ-SAM ile kırpılıyor "
              f"(crop/, G3 {sur})...")
        Oturum.build_bank(RT.ref_dir(), out_path=bank_path, save_crops=True)
        self.ref = Oturum(self.modality, bank_path=bank_path, load_models=True)
        self.ref.load_bank(bank_path)
        self.ref.set_schedule(schedule)
        print(f"[orchestrator] referans bankası hazır ({sur}): {self.ref.ref_ids}")

    # -------- her kare --------
    def process_frame(self, info: FrameInfo, image_bgr: np.ndarray) -> dict:
        t_all = time.time()
        tm = {}
        RT.save_new_frame(image_bgr, info.video_name or f"frame_{self.frame_idx}")

        pkg = ResultPackage(frame_url=info.url, user_url="")

        # ---- G1: tespit + taşıt hareket ----
        objs: list = []
        vehicle_boxes, obstacle_boxes = [], []
        if self.pipe is not None:
            t0 = time.time()
            objs = self.pipe.process(image_bgr)          # DetectedObject (motion dolu)
            tm["detect"] = time.time() - t0
            for o in objs:
                box = (o.top_left_x, o.top_left_y, o.bottom_right_x, o.bottom_right_y)
                if o.cls == VEHICLE_CLS:
                    vehicle_boxes.append(box)
                if o.cls in OBSTACLE_CLS:
                    obstacle_boxes.append(box)

        # ---- G1: UAP/UAİ iniş ----
        if self.landing is not None and objs:
            t0 = time.time()
            for o in objs:
                if o.cls in (UAP_CLS, UAI_CLS):
                    box = (o.top_left_x, o.top_left_y, o.bottom_right_x, o.bottom_right_y)
                    o.landing_status = self.landing.status_for(image_bgr, box, o.cls, obstacle_boxes)
            tm["landing"] = time.time() - t0
        pkg.detected_objects = objs

        # ---- G3: referans eşleme ----
        if self.ref is not None:
            t0 = time.time()
            try:
                matches = self.ref.process(image_bgr, frame_idx=self.frame_idx)
                pkg.detected_undefined_objects = matches or []
            except Exception as e:
                print(f"[orchestrator] G3 hata: {e}")
            tm["reference"] = time.time() - t0

        # ---- G2: GPS ----
        if self.gps is not None:
            t0 = time.time()
            gt = None
            if info.gps_health_status == 1 and info.translation_x is not None:
                gt = (info.translation_x, info.translation_y, info.translation_z)
            world = self.gps.process(image_bgr, self.frame_idx, gt, info.gps_health_status)
            if world is not None:
                pkg.detected_translations = [DetectedTranslation(*world)]
            tm["gps"] = time.time() - t0

        tm["total"] = time.time() - t_all
        self.last_timing = tm
        self.frame_idx += 1

        # NOT: ilk 10 kare sunucuda DEĞERLENDİRİLMEZ ama biz yine de TAM sonucu gönderiyoruz
        # (kullanıcı kararı: "ilk 10 frame de de sonuç yollayalım"). Boş bırakmıyoruz.
        return pkg.to_json()

    def close(self):
        if self.gps is not None:
            self.gps.stop()
