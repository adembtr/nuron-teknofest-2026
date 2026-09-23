#!/usr/bin/env python3
"""
Görev 1 üst-seviye cephe (facade) — TEK giriş noktası.

Detector (YOLO) + MotionEstimator (hareket) tek çağrıda birleşir ve doğrudan
şema nesnesi (DetectedObject) üretir. RGB ve TERMAL için AYNI kod; fark yalnız
MOTION_PROFILE[modality] ve imgsz'dir (modalite dışarıdan verilir, oturum içi sabit).

Akış (her kare, nedensel/streaming):
  1) YOLO → tüm nesneler (0=tasit 1=insan 2=uap 3=uai).
  2) Yalnız TAŞIT (cls==0) kutuları MotionEstimator.update()'e verilir → 0/1.
     (insan/uap/uai harekete girmez; motion_status = -1.)
  3) DetectedObject listesi: taşıt → motion_status 0/1, diğerleri → -1.

motion_status: 0=hareketsiz, 1=hareketli, -1=taşıt değil (şartname Tablo 4).
landing_status burada -1 (iniş = ayrı modül); UAP/UAI iniş uygunluğu landing.py'de.
"""
from typing import List, Optional
import numpy as np

from src.common.config import model_root
from src.common.schema import DetectedObject
from src.task1_detection.detector import Detector
from src.task1_detection.motion import MotionEstimator

# YOLO çıkarım boyutu — model 1280px'de eğitildi → HER İKİ modalite 1280 (eğitim çözünürlüğü
# ile eşleşsin; termal native 640x512 olsa da model 1280'de eğitildiği için 1280'de daha iyi).
IMGSZ = {"rgb": 1280, "termal": 1280}
DEFAULT_CONF = 0.5
VEHICLE_CLS = 0

# SINIF-BAZLI güven eşiği (0=tasit 1=insan 2=uap 3=uai).
# termal: taşıt 0.8 (yanlış-pozitif çok → sıkı), insan/UAP/UAİ 0.5.
# rgb: None → tek eşik (DEFAULT_CONF); RGB modeli sonra ayarlanacak.
CLASS_CONF = {
    "termal": {0: 0.8, 1: 0.5, 2: 0.5, 3: 0.5},
    # RGB (2026-09-18): İNSAN tabanı 0.5 → 0.35. folder1'de 0.35-0.50 bandındaki, iz devamının
    # zaten almadığı 98 "yalnız" insan kutusuna tek tek bakıldı: hepsi gerçek insan (yeni giren /
    # yalnız duran kişiler). Diğer sınıflar 0.5 (araçta bu bantta yanlışlar var, dokunulmadı).
    "rgb": {0: 0.5, 1: 0.35, 2: 0.5, 3: 0.5},
}

# ── İZ DEVAMI (kırpışma önleyici, 2026-09-18) — bkz. detector.IzDevam ──
# Sınıf eşiğinin altında ama `alt` üstünde bir kutu, son k_maks karede görülmüş GÜVENLİ bir izi
# sürdürüyorsa kabul edilir. Kutu uydurulmaz, GPU maliyeti yok (YOLO tek geçiş, düşük eşik).
# v2: ego-hareket tahmini (drone kaymasını izlere uygular, kapı daha dar → şüpheli ekleme azalır).
# folder1 ölçümü: insan kırpışma %27,9 → %8,7 (alt 0,05); araç %7,2 → %4,8 (alt 0,25; %10'da
# depo/kubbe yanlışları var, 0,25 altına İNME); UAP/UAİ kırpışma zaten %1, alt 0,25 ile 6+3 kutu.
# En düşük güvenli eklemeler görsel olarak tek tek doğrulandı (kullanıcı onayı, ~/Desktop/yolo_insan).
# TERMAL: None → eski davranış birebir. Kapatma: NURON_IZ_DEVAM=0
IZ_DEVAM = {
    "rgb": dict(siniflar=(0, 1, 2, 3), alt={1: 0.05, 0: 0.25, 2: 0.25, 3: 0.25},
                k_maks=6, kat=2.0, ego=True, kat_ego=1.5),
    "termal": None,
}


class DetectionPipeline:
    """Streaming Görev-1 cephesi. Oturum boyunca tek örnek; her kare process()."""

    def __init__(self, modality: str, conf: float = DEFAULT_CONF,
                 imgsz: Optional[int] = None, device: str = "cuda"):
        assert modality in ("rgb", "termal")
        self.modality = modality
        self.detector = Detector(model_root(modality), modality, conf=conf,
                                 imgsz=imgsz or IMGSZ[modality], device=device,
                                 class_conf=CLASS_CONF.get(modality),
                                 devam=IZ_DEVAM.get(modality))
        self.motion = MotionEstimator(modality)

    def process(self, img: np.ndarray) -> List[DetectedObject]:
        """Tek kare → DetectedObject listesi (motion_status dolu)."""
        dets = self.detector.detect(img)

        # taşıt kutularını sırayı KORUYARAK ayır → MotionEstimator aynı sırada 0/1 döndürür
        v_idx = [i for i, d in enumerate(dets) if d.cls == VEHICLE_CLS]
        v_boxes = [(dets[i].x1, dets[i].y1, dets[i].x2, dets[i].y2) for i in v_idx]
        flags = self.motion.update(img, v_boxes)          # len == len(v_boxes)
        flag_by_det = {di: flags[k] for k, di in enumerate(v_idx)}

        out: List[DetectedObject] = []
        for i, d in enumerate(dets):
            mstat = flag_by_det.get(i, -1) if d.cls == VEHICLE_CLS else -1
            out.append(DetectedObject(
                cls=d.cls,
                top_left_x=d.x1, top_left_y=d.y1,
                bottom_right_x=d.x2, bottom_right_y=d.y2,
                motion_status=int(mstat),
            ))
        return out

    # görselleştirme/debug için hareket ayrıntısı (tid, hız, kenar, ratio)
    @property
    def motion_info(self) -> List[dict]:
        return self.motion.last_info
