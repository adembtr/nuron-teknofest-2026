#!/usr/bin/env python3
"""
Görev 2 (GPS-siz pozisyon) — YOLLAR + MODALİTE PROFİLLERİ.   [v4 YAMA 2026-09-11: termal K 2 → 1]

DPVO AYRI conda ortamında koşar (dpvo, Python 3.11 — derlenmiş CUDA eklentileri).
Orkestratör base ortamda. DPVO ayrı worker süreçtir; dpvo_runner.py o worker'da çalışır.
aligner/z_* modülleri base'de (saf numpy/opencv).

╔══════════════════════════════════════════════════════════════════════════════╗
║  RGB GPS ve TERMAL GPS TAMAMEN AYRI PROFİL — KARIŞTIRMA.                        ║
║  Her şey aşağıdaki GPS_PROFILE[modality] tablosunda. Kod bunu okur.             ║
║  DPVO ağı (dpvo.pth) ortaktır; FARK ön-işleme + config + Z yönteminde.          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
import os
from src.common.config import ROOT

# --- DPVO worker (ayrı ortam, her iki modalite ortak altyapı) ---
# Taşınabilir: DPVO_ENV_PY ortam değişkeni varsa onu, yoksa ~/miniconda3/envs/dpvo yolunu kullan.
# Conda başka yerdeyse:  export DPVO_ENV_PY="$(conda info --base)/envs/dpvo/bin/python"
DPVO_ENV_PY = os.environ.get(
    "DPVO_ENV_PY",
    os.path.join(os.path.expanduser("~"), "miniconda3", "envs", "dpvo", "bin", "python"),
)
DPVO_REPO   = os.path.join(ROOT, "third_party", "dpvo_repo")   # NURON fix: dpvo.py:427 ones_like
DPVO_CFG    = os.path.join(ROOT, "models", "dpvo_default.yaml")   # ortak (prefix yok)

# Modaliteye göre AĞIRLIK (düz models/ + modalite öneki): models/rgb_dpvo.pth, models/termal_dpvo.pth
DPVO_CKPT = lambda m: os.path.join(ROOT, "models", f"{m}_dpvo.pth")
CALIB     = lambda m: os.path.join(ROOT, "config", "calibration",
                                   "calib_rgb1080.txt" if m == "rgb" else "calib_termal.txt")

# --- Görev/oturum sabitleri (ortak) ---
GT_FRAMES = 450        # ilk 450 kare health=1 → kalibrasyon; sonra kendi VO
FPS = 7.5              # sunucu kare hızı (30fps videonun her 4. karesi = stride-4)
BUFFER_SIZE = 8192     # DPVO poz tamponu

# ══════════════════════════════════════════════════════════════════════════════
#  GPS MODALİTE PROFİLLERİ — RGB ve TERMAL YAN YANA, AYRI
# ══════════════════════════════════════════════════════════════════════════════
#
#            │ RGB (kolay: bol feature)      │ TERMAL (zor: zayıf feature)
#  ──────────┼───────────────────────────────┼──────────────────────────────────
#  Ön-işleme │ ham, 0.5x küçültme            │ CLAHE+unsharp (sharpen) + tam ölçek
#  DPVO cfg  │ varsayılan (patch96, kf15)    │ tune (patch160, kf8, ow15) düşük-doku
#  XY        │ DPVO doğrudan                 │ DPVO + sharpen (domain gap kapatır)
#  Z ekseni  │ FINAL-11 3-kaynak füzyon (final_kestirici.py) │ FINAL-TERMAL 3-kaynak füzyon (termal_kestirici.py, 2026-09-04)
#  Kalibr.   │ ilk-450 + akıllı pencere      │ ilk-450 + akıllı pencere (durgun atla)
#  Sonuç     │ 1.21/0.74/1.33 m ✅            │ ot4 1.18/2.15/2.20 · 2026 2.3/15.8/2.3 (GPS yok) — SONUC.md
#
GPS_PROFILE = {
    "rgb": dict(
        scale = 640/1920,      # 1080p → 640×360 (DPVO NATIVE ~640px eğitildi; robust)
        prep  = "none",        # ham RGB, ön-işleme yok
        z_method = "final11",         # final_kestirici.py: ORB oran + DPVO z + DPVO derinlik, eğim Kalmanı (2026-09-03)
        cfg_tune = None,       # varsayılan DPVO config (patch96/kf15)
    ),
    "termal": dict(
        scale = 1.0,           # tam ölçek (640×512)
        prep  = "sharp",       # CLAHE(3.0) + unsharp [[0,-1,0],[-1,5,-1],[0,-1,0]] — domain adaptation
        z_method = "object_size",     # z_termal.ObjectSizeZ (feature çift-mesafe = irtifa)
        cfg_tune = dict(              # düşük-doku için DPVO tune
            PATCHES_PER_FRAME=160,    # 96 → 160 (çok tutunma noktası)
            KEYFRAME_THRESH=8.0,      # 15 → 8 (sık keyframe)
            OPTIMIZATION_WINDOW=15,   # 10 → 15 (geniş BA penceresi)
            REMOVAL_WINDOW=30,
            MOTION_DAMPING=1.0,       # FINAL-TERMAL (2026-09-04): tekrar-kare boşluğunda hareket modeli boşluğu TAM ekstrapole etsin (0.5 ile yarım → ölçek modu dallanıyor / çöküyor)
        ),
        # FINAL-TERMAL: eşzamanlı DPVO örneği sayısı (sağlam ensemble; K=1 tek örnek). 8 GB GPU: 2×~1.8 GB
        # 2026-09-05 E2E: hibrit G3 (4.9 GB) + YOLO + K=2 worker (2.9 GB) 8 GB'a SIĞMADI (G1/G3 OOM) ->
        # NURON_TERMAL_K ortam değişkeniyle ezilebilir.
        # ═══ v4 (2026-09-11, adim5 hakem kararı §6.3): VARSAYILAN 2 → 1 ═══
        #   (a) K=2 ÖLÇÜLMÜŞ BOZULMA: sanal K=2/K=3 ensemble ot4 GPS-yok Y'yi 2.09 → 4.21/4.27 m yapıyor
        #       (aynı reçeteli çiftte de tekrarlanıyor) — denetim_nedensellik + adim5 RAPOR §3/13.
        #   (b) K=2 hiç uçtan uca doğrulanmadı; 8/8 canlı GPU replay koşusu K=1 ile yapıldı (tepe 1757 MB).
        #   (c) K=2 + hibrit G3 canlı E2E'de OOM aldı.
        #   NOT: config/gps_termal_final*.json'daki "dpvo": {"K": ...} bloğu HİÇBİR KOD TARAFINDAN OKUNMUYOR
        #        (yalnız belge); K'nın TEK GERÇEK KAYNAĞI bu satır ve NURON_TERMAL_K ortam değişkenidir.
        K = int(os.environ.get("NURON_TERMAL_K", "1")),
        tekrar_esik = 1.5,            # FINAL-TERMAL: ardışık ön-işlenmiş gri kare ort |fark| < eşik → video TEKRAR karesi → DPVO'ya verilmez
    ),
}

# --- LOOP CLOSURE: her iki modalitede AÇIK (ek model gerekmez, edges_loop) ---
# Yarışma kareleri seyrek (2250 kare = geniş baseline). Termalde tracking için önemli.
DPVO_LOOP_CLOSURE = False   # KESIN KAPALI (kullanici karari): havadan/deniz duz sahnede sahte loop-closure jump. Zaten dpvo_runner._build_cfg cfg.LOOP_CLOSURE=False ile hardcoded.

# --- Termal Z (nesne-boyutu) sabitleri ---
TERMAL_F = 731.8       # termal odak (piksel) — GSD h0 başlangıç yüksekliği için

# --- Geriye-uyum (eski kod bu isimleri kullanıyorsa) ---
DPVO_SCALE = {m: GPS_PROFILE[m]["scale"] for m in GPS_PROFILE}
DPVO_PREP  = {m: GPS_PROFILE[m]["prep"]  for m in GPS_PROFILE}
TERMAL_CFG_TUNE = GPS_PROFILE["termal"]["cfg_tune"]
