#!/usr/bin/env python3
"""
Görev 3 model yolları — NURON_DRONE/models/ altına bakar.
Referans bankası (HQ-SAM + DINOv3) modaliteden bağımsız (renkli+gri saklanır).
Kare segmenter'i modaliteye göre: rgb→CropFormer, termal→SAM2.
"""
import os
from src.common.config import ROOT

# Tüm ağırlıklar DÜZ models/ altında (prefix'li). Ortak (iki modalite) prefix'siz;
# RGB'ye özel CropFormer 'rgb_' önekli; SAM2 iki yerde (RGB takip + termal segment) → ortak, prefix'siz.
_MODELS = os.path.join(ROOT, "models")

# --- Ortak (her iki modalite) — prefix yok ---
DINOV3_PATH = os.path.join(_MODELS, "dinov3_vitl16")
HQSAM_CKPT  = os.path.join(_MODELS, "sam_hq_vit_l.pth")
HQSAM_TYPE  = "vit_l"

# --- RGB kare segmenter: CropFormer (depo NURON_DRONE/third_party içinde) ---
CROPFORMER_CKPT = os.path.join(_MODELS, "rgb_CropFormer_swin_tiny_3x.pth")
CROPFORMER_REPO = os.path.join(ROOT, "third_party", "CropFormer")
CROPFORMER_CFG  = os.path.join(CROPFORMER_REPO,
    "configs/entityv2/entity_segmentation/cropformer_swin_tiny_3x.yaml")

# --- SAM2 kare segmenter (termal) + RGB takip (ref_tracker) — ORTAK, prefix yok ---
SAM2_CKPT = os.path.join(_MODELS, "sam2.1_hiera_large.pt")
SAM2_CFG  = "configs/sam2.1/sam2.1_hiera_l.yaml"

# --- Referans bankası çıktısı ---
REF_BANK = os.path.join(ROOT, "offline_data", "ref_bank.npz")

# --- Ayarlar ---
DEVICE  = "cuda"
EMB_SIZE = 224
MAXSIDE  = 1024
MAXSIDE_SAM = 1536
MATCH_THRESH        = 0.60   # RGB cosine eşiği (ref DOGRUDAN yollanmaz, SAM2'yi cogunlukla besler)
MATCH_THRESH_TERMAL = 0.40   # termal cosine eşiği (karar 2026-07-10: 0.45 sınırdaki
                             # gerçek nesneyi kaçırdı, örn. biçer 0.416; set_active
                             # aralık kısıtıyla 0.40 güvenli)
# Termal oturumda referans varyantları: renkli(direkt) + düz gri. Frame-adaptif
# ton-eşleme KAPALI (karar 2026-07-10: normal gri yeterli; ThermalGen de elendi).
USE_THERMAL_TONE = False
# Kenar/siluet (saf şekil) varyantları: DENENDI, ELENDI (2026-07-10) — DINOv3'te
# ayırt edicilik çöküyor (çalı→UAP 0.87 gibi sahte eşleşmeler). Kod duruyor, kapalı.
USE_SHAPE_VARIANTS = False

# CropFormer
CF_MIN_SIZE_TEST = 512
CF_MAX_SIZE_TEST = 1024
CF_SCORE_THRESH  = 0.50
# SAM2
SAM2_POINTS_PER_SIDE = 16   # 32→16 (2026-07-11): ~3.4x hızlı (SAM2 2.0s→0.6s, referans 2.2s→~0.75s,
                            #  tam koşu 82dk→~28dk). Büyük nesne aynı; çok küçük nesnede az kayıp riski.
SAM2_PRED_IOU = 0.82
SAM2_STABILITY = 0.88
SAM2_MIN_AREA = 200
