#!/usr/bin/env python3
"""
Görev 1 — UAP/UAİ iniş durumu (landing_status ∈ {1=uygun, 0=uygun değil, -1=iniş alanı değil}).

Şartname (Tablo 5, Şekil 9-11): daire TEMİZ + tamamı karede + üstünde/yanında cisim yoksa → 1;
üstünde/yanında taşıt/insan/cisim varsa (açı yüzünden üstündeymiş gibi görünse bile) → 0;
daire kareye tam sığmıyorsa → 0. (landing_status yalnız UAP=2 / UAİ=3 için; diğerleri -1.)

İKİ-KATMANLI KARAR (SUMMARY/padim.md tasarımı):
  1) KADRAJ: kutu kare kenarına değiyor/taşıyorsa (daire tam görünmüyor) → 0.
  2) ENGEL ÖRTÜŞME (birincil): tespit edilmiş TAŞIT/İNSAN kutusu daireyle çakışıyorsa → 0.
  3) PaDiM (ikincil): YOLO'nun kaçırdığı sınıfsız cisim → daire crop'u Mahalanobis anomali
     skoru > eşik ise → 0. Aksi halde → 1 (uygun).

╔══════════════════════════════════════════════════════════════════════════════╗
║  ÇOKLU PaDiM — "BİRİ UYGUN DERSE UYGUN"  (2026-09-18)                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
Sorun: eski RGB PaDiM'i GÖLGELİ UAP/UAİ'de yanılıyor, temiz ama gölge düşmüş daireye
"uygun değil" diyordu. Eski modelin üzerine eğitmek onu unutturacağı için ESKİ MODEL
AYNEN KALDI; yanına yeni veriyle eğitilmiş İKİNCİ bir model kondu.

  Bir sınıf (uap/uai) için N model çalışır. Karar:
      modellerden EN AZ BİRİ "uygun" derse            → 1 (uygun)
      HEPSİ birden "uygun değil" derse                → 0 (uygun değil)
  Yani anomali kararı OY BİRLİĞİ ister; tek model varken davranış ESKİSİYLE BİREBİR.

MODEL YOLU (2026-09-18 değişti):
  * KLASÖR varsa  models/{modality}_padim_{uap,uai}/*.pth  → içindeki TÜM modeller
    (ad sırasına göre) yüklenir. RGB böyle: rgb_padim_uap/{rgb_padim_uap.pth,
    rgb_padim_uap2.pth}.
  * Klasör yoksa eski DÜZ dosya  models/{modality}_padim_{uap,uai}.pth  kullanılır.
    TERMAL bu yoldan gider → termal davranışı HİÇ DEĞİŞMEDİ (tek model, aynı karar).

ResNet18 öznitelik çıkarıcı artık modeller arasında PAYLAŞILIR (cihaz başına tek
kopya): aynı ağırlık, aynı eval modu, aynı girdi → öznitelikler birebir aynı, yalnız
bellek ve süre iki katına çıkmıyor.

PaDiM modeli: models/{modality}_padim_{uap,uai}[/*].pth (ResNet18, 448×196, Mahalanobis).
"""
import os
from typing import List, Tuple, Optional
import numpy as np

from src.common.config import model_root

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

UAP_CLS, UAI_CLS = 2, 3
OBSTACLE_CLS = (0, 1)          # taşıt, insan → daire üstünde olursa iniş uygun değil

# Karar ayarları
EDGE_MARGIN = 3.0              # kutu kenara bu kadar px yakınsa "kadraj dışı" say
OVERLAP_MIN = 0.05            # engel kutusu daire alanının bu oranını örterse → engel var
                              # 2026-09-18: 0.02 -> 0.05. %2 cok hassasti: yandaki aracin/insanin
                              # KUTUSU daireye degdigi icin, cisim dairenin USTUNDE olmasa bile
                              # PaDiM hic calismadan 0 gidiyordu. Olculdu (925 kutu, folder0+1):
                              # %2 -> %5 bandindaki 57 kutunun cogunda cisim dairenin DISINDA.
                              # %10 denendi ve ELENDI: o bantta dairenin UZERINDE duran insanlar var.
                              # Kapi acilinca karar PaDiM e gecer, o da bu kutularin yarisini reddediyor.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def _iou_area_frac(a, b) -> float:
    """b (engel) kutusunun a (daire) ile kesişiminin, a alanına oranı."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(1e-6, (ax2 - ax1) * (ay2 - ay1))
    return inter / area_a


# ── PAYLAŞILAN ÖZNİTELİK ÇIKARICI (2026-09-18) ────────────────────────────────
# Aynı sınıf için birden çok PaDiM modeli çalıştığından ResNet18 cihaz başına BİR KEZ
# yüklenir. Ağırlık/eval modu/girdi aynı olduğu için üretilen öznitelik, her modelin
# kendi ResNet'i olduğu eski sürümle BİREBİR aynıdır; yalnız bellek ve süre tasarrufu.
_OMURGA: dict = {}


def _omurga(device: str):
    """(net, feat_dict) — device başına tek kopya (lazy)."""
    if device not in _OMURGA:
        import torch
        from torchvision.models import resnet18, ResNet18_Weights
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device).eval()
        feat: dict = {}
        net.layer1.register_forward_hook(lambda m, i, o: feat.__setitem__("l1", o))
        net.layer2.register_forward_hook(lambda m, i, o: feat.__setitem__("l2", o))
        net.layer3.register_forward_hook(lambda m, i, o: feat.__setitem__("l3", o))
        _OMURGA[device] = (net, feat)
    return _OMURGA[device]


class _PadimModel:
    """Tek PaDiM modeli (uap veya uai) — ResNet18 feature + Mahalanobis skor."""

    def __init__(self, pth_path: str, device: str = "cuda"):
        import torch
        import torch.nn.functional as F
        self._torch = torch
        self._F = F
        self.device = device if torch.cuda.is_available() else "cpu"
        self.path = pth_path
        self.name = os.path.splitext(os.path.basename(pth_path))[0]
        d = torch.load(pth_path, map_location=self.device, weights_only=False)
        self.mean = d["mean"].to(self.device)                 # (448,196)
        self.cov_inv = d["cov_inv"].to(self.device)           # (196,448,448)
        self.threshold = float(d["threshold"])
        # feature çıkarıcı (donuk, PAYLAŞILAN)
        self.net, self._feat = _omurga(self.device)

    def score(self, crop_bgr: np.ndarray) -> float:
        torch, F = self._torch, self._F
        if crop_bgr is None or crop_bgr.size == 0:
            return 0.0
        im = cv2.resize(crop_bgr, (224, 224))
        im = (cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        t = torch.from_numpy(im).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            self.net(t)
            l1, l2, l3 = self._feat["l1"], self._feat["l2"], self._feat["l3"]
            s = l3.shape[-2:]
            e = torch.cat([F.interpolate(l1, s, mode="bilinear", align_corners=False),
                           F.interpolate(l2, s, mode="bilinear", align_corners=False), l3], 1)
            e = e.squeeze(0).reshape(448, -1)                 # (448,196)
            d = (e - self.mean).permute(1, 0)                 # (196,448)
            m = torch.einsum("pi,pij,pj->p", d, self.cov_inv, d)
            return float(torch.sqrt(m.clamp(min=0)).max())

    def anomali(self, crop_bgr: np.ndarray) -> bool:
        """True = bu model 'inişe uygun DEĞİL' diyor."""
        return self.score(crop_bgr) > self.threshold


class LandingEstimator:
    """UAP/UAİ iniş uygunluğu. status_for(...) tek işaretçi için 0/1 döndürür.
    PaDiM ağır (ResNet18) → lazy yükle; VRAM darsa use_padim=False ile kapatılabilir."""

    def __init__(self, modality: str, device: str = "cuda", use_padim: bool = True):
        assert modality in ("rgb", "termal")
        self.modality = modality
        self.device = device
        self.use_padim = use_padim
        self._root = model_root(modality)
        self._models: dict[int, list] = {}                    # cls → [model, ...] (lazy)

    def _yollar(self, cls: int) -> List[str]:
        """Bu sınıf için yüklenecek .pth yolları.
        KLASÖR varsa içindeki tüm .pth (ad sırasıyla) → çoklu model (RGB).
        Klasör yoksa eski düz dosya → tek model (TERMAL: davranış değişmez)."""
        taban = (f"{self.modality}_padim_uap" if cls == UAP_CLS
                 else f"{self.modality}_padim_uai")
        klasor = os.path.join(self._root, taban)
        if os.path.isdir(klasor):
            return sorted(os.path.join(klasor, f) for f in os.listdir(klasor)
                          if f.endswith(".pth"))
        duz = os.path.join(self._root, taban + ".pth")
        return [duz] if os.path.exists(duz) else []

    def _model(self, cls: int) -> List[_PadimModel]:
        """Bu sınıfın PaDiM modelleri (lazy). Boş liste → PaDiM katmanı kapalı."""
        if not self.use_padim or cv2 is None:
            return []
        if cls not in self._models:
            yuklu = []
            for p in self._yollar(cls):
                try:
                    yuklu.append(_PadimModel(p, self.device))
                except Exception as e:                      # bozuk/eksik dosya akışı durdurmaz
                    print(f"[NURON] PaDiM yuklenemedi ({p}): {e}", flush=True)
            if yuklu:
                print(f"[NURON] PaDiM {'UAP' if cls == UAP_CLS else 'UAI'}: "
                      f"{len(yuklu)} model -> " +
                      ", ".join(f"{m.name}(esik {m.threshold:.1f})" for m in yuklu) +
                      ("  [karar: BIRI uygun derse UYGUN]" if len(yuklu) > 1 else ""),
                      flush=True)
            else:
                # 2026-09-20: DOSYA HIC YOKSA eski kod tamamen SESSIZ kaliyordu -- _yollar()
                # bos liste donunce yukleme dongusu hic donmuyor, except tetiklenmiyor, ustteki
                # "if yuklu" atlaniyordu. Sonuc: status_for() asagida `return 1` ile HER daireye
                # "inis uygun" der ve bunu kimse fark etmez. Olcum/karar mantigina DOKUNULMADI,
                # sadece gorunurluk: models/ bozulursa/tasinirsa artik ekranda gorulur.
                print(f"[NURON] UYARI: PaDiM {'UAP' if cls == UAP_CLS else 'UAI'} modeli "
                      f"BULUNAMADI ({self._root}) -> anomali katmani KAPALI, "
                      f"kadraj+engel gecen TUM daireler 'uygun (1)' gidecek!", flush=True)
            self._models[cls] = yuklu
        return self._models[cls]

    def status_for(self, frame_bgr: np.ndarray, marker_box: Tuple[float, float, float, float],
                   marker_cls: int, obstacle_boxes: List[Tuple[float, float, float, float]]) -> int:
        """Tek UAP/UAİ işaretçisi için landing_status (0/1)."""
        H, W = frame_bgr.shape[:2]
        x1, y1, x2, y2 = marker_box
        # 1) KADRAJ: daire tam karede değilse → uygun değil
        if x1 <= EDGE_MARGIN or y1 <= EDGE_MARGIN or x2 >= W - EDGE_MARGIN or y2 >= H - EDGE_MARGIN:
            return 0
        # 2) ENGEL ÖRTÜŞME: taşıt/insan daireyle çakışıyorsa → uygun değil
        for ob in obstacle_boxes:
            if _iou_area_frac(marker_box, ob) > OVERLAP_MIN:
                return 0
        # 3) PaDiM anomali: YOLO'nun kaçırdığı cisim → uygun değil
        #    ÇOKLU MODEL (2026-09-18): modellerden BİRİ bile "uygun" derse UYGUN sayılır;
        #    "uygun değil" kararı OY BİRLİĞİ ister. Tek model varken (termal) eski davranış.
        mdls = self._model(marker_cls)
        if mdls:
            cx1, cy1 = max(0, int(x1)), max(0, int(y1))
            cx2, cy2 = min(W, int(x2)), min(H, int(y2))
            crop = frame_bgr[cy1:cy2, cx1:cx2]
            for m in mdls:
                if not m.anomali(crop):        # bu model "uygun" dedi → kimseye sorma
                    return 1
            return 0                            # hepsi "uygun değil" dedi
        return 1

    # ------------------------------------------------------------------
    def teshis(self, frame_bgr: np.ndarray, marker_box, marker_cls: int,
               obstacle_boxes: List[Tuple[float, float, float, float]]) -> dict:
        """Aynı kararı verir ama HER katmanın ve HER modelin ne dediğini de döndürür.
        Yalnız ölçüm/görselleştirme içindir; status_for'un kararını DEĞİŞTİRMEZ."""
        H, W = frame_bgr.shape[:2]
        x1, y1, x2, y2 = marker_box
        out = {"kadraj_disi": False, "engel": False, "skorlar": {}, "esikler": {},
               "model_karar": {}, "durum": 1}
        if x1 <= EDGE_MARGIN or y1 <= EDGE_MARGIN or x2 >= W - EDGE_MARGIN or y2 >= H - EDGE_MARGIN:
            out["kadraj_disi"] = True; out["durum"] = 0; return out
        for ob in obstacle_boxes:
            if _iou_area_frac(marker_box, ob) > OVERLAP_MIN:
                out["engel"] = True; out["durum"] = 0; return out
        mdls = self._model(marker_cls)
        if not mdls:
            return out
        cx1, cy1 = max(0, int(x1)), max(0, int(y1))
        cx2, cy2 = min(W, int(x2)), min(H, int(y2))
        crop = frame_bgr[cy1:cy2, cx1:cx2]
        for m in mdls:
            s = m.score(crop)
            out["skorlar"][m.name] = s
            out["esikler"][m.name] = m.threshold
            out["model_karar"][m.name] = 0 if s > m.threshold else 1   # 1 = uygun
        out["durum"] = 1 if any(v == 1 for v in out["model_karar"].values()) else 0
        return out
