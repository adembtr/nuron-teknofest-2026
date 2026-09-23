#!/usr/bin/env python3
"""
Görev 1 — Nesne Tespiti. YOLO sarmalayıcı.
models/{modality}_yolo.pt yüklenir (0=tasit 1=insan 2=uap 3=uai).
Çıktı: her nesne için sınıf + bbox (piksel). motion/landing ayrı modüllerde eklenir.

╔══════════════════════════════════════════════════════════════════════════════╗
║  İZ DEVAMI — kırpışma önleyici  (2026-09-18)                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
SORUN: YOLO insanı bulup 1-2 kare sonra bırakıyor, sonra yine buluyor. folder1'de
ölçüldü (2250 kare): insan izlerinin karelerinin %27,9'u boş. Kaçan karelerin %83'ünde
YOLO doğru yerde bir kutu ÜRETMİŞ ama güveni 0,5 eşiğinin altında kalmış (0,10-0,49);
yalnız %17'sinde hiç kutu yok. Yani dedektör kör değil, eşik sert + kareler arası bağ yok.

NEDEN HAZIR İZLEYİCİ DEĞİL: insan kutusu ~30 px ve drone hareketinden ardışık karede
kendi boyunun 0,8 katı kayıyor (ardışık IoU medyanı 0,06). Ultralytics ByteTrack/BoT-SORT
bir izi ancak ardışık eşleşince YAYINLADIĞI için tespit KAYBEDİYOR (ByteTrack 1215 güvenli
kutunun 843'ünü, kamera telafili BoT-SORT 213'ünü düşürdü). SAM ise GPU'ya sığmaz.

ÇÖZÜM (bu dosya, `IzDevam`): üretim çıktısı AYNEN kalır, üstüne küçük bir kural:
  1) YOLO TEK geçişte daha düşük eşikle koşar (GPU maliyeti aynı).
  2) Sınıf eşiğini (0,5) geçen her kutu bugünkü gibi gider — hiçbir şey değişmez.
  3) Eşik altı ama `alt` üstü (0,10) bir kutu, YALNIZ son `k_maks` (6) karede görülmüş
     GÜVENLİ bir izin beklenen yerine yakınsa kabul edilir (mesafe kapısı = kat × kutu
     boyu × kayıp kare sayısı; drone kayması birikir).
  4) Yeni iz yalnız güvenli kutuyla doğar; kutu UYDURULMAZ, eklenen her kutu YOLO'nun kendi kutusu.
  5) Güvenli görülmeyeli k_maks kareyi geçen iz düşer → tek başına düşük kutuyla sonsuza dek sürmez.

ÖLÇÜLDÜ (folder1, çevrimdışı benzetim, insan): v1 (alt 0,10) +607 kutu, 451 boşluğun 297'si
doldu, kırpışma %27,9 → %9,5. Eklenen kutuların EN DÜŞÜK güvenli 20'si ve iz-sonrası kuyruk
eklemelerinin en düşük 20'si tek tek bakıldı: hepsi gerçek insan (kullanıcı onayı, video ile).
Kalan boşlukların çoğu dedektörün gerçekten kör olduğu kareler (%17).

v2 (aynı gece, "biraz daha güçlendir"): (a) alt 0,05 → 311 boşluk; (b) EGO-HAREKET TAHMİNİ:
güvenli eşleşen izlerin ortak kaymasının medyanı eşleşmeyen izlere uygulanır, kapı √k ile
büyüyen dar kapı → aynı doldurma, şüpheli ekleme 607 → 561; (c) sınıf başına alt eşik.
ELENENLER: iz ömrünü uzatmak (k_maks 10/15) hiç boşluk doldurmuyor, yalnız kuyruk ekliyor;
"hayalet kutu" (tespit yokken tahmine kutu koymak) konumu medyan 0,69 kutu boyu kayık ve
119 hayaletin 77'si iz bittikten sonra → REDDEDİLDİ, kutu uydurulmaz.

KAPSAM: `pipeline.IZ_DEVAM` — RGB'de dört sınıf (insan alt 0,05; araç/UAP/UAİ alt 0,25).
TERMAL: None → kod yolu ESKİSİYLE BİREBİR (tek predict, aynı eşik). Kapatma: NURON_IZ_DEVAM=0
"""
import os
from dataclasses import dataclass
from typing import List, Optional, Dict
import numpy as np


@dataclass
class Detection:
    cls: int
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float


def _iou(a: "Detection", b: "Detection") -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = (a.x2 - a.x1) * (a.y2 - a.y1)
    ab = (b.x2 - b.x1) * (b.y2 - b.y1)
    return inter / (aa + ab - inter + 1e-9)


def dedup_same_class(dets: List["Detection"], iou_thr: float = 0.8) -> List["Detection"]:
    """Mükerrer kutu cezası (Örnek 4 / Tablo 11): aynı sınıftan iki kutu IoU>eşik ise
    (ör. insan-insan) düşük güvenli olan atılır. YOLO NMS'ine ek güvenlik katı."""
    order = sorted(range(len(dets)), key=lambda i: dets[i].conf, reverse=True)
    keep, dropped = [], set()
    for i in order:
        if i in dropped:
            continue
        keep.append(i)
        for j in order:
            if j == i or j in dropped:
                continue
            if dets[j].cls == dets[i].cls and _iou(dets[i], dets[j]) > iou_thr:
                dropped.add(j)          # aynı tür + yüksek örtüşme → düşük güvenliyi ele
    return [dets[i] for i in sorted(keep)]


# ══════════════════════════════════════════════════════════════════════════════
#  İZ DEVAMI (2026-09-18) — saf Python, GPU yok, kare başına mikrosaniyeler
# ══════════════════════════════════════════════════════════════════════════════
class IzDevam:
    """Eşik altı kutuyu, yakın geçmişteki GÜVENLİ bir iz sürdürüyorsa kabul eder.

    siniflar : bu kurala giren sınıflar (ör. (1,) = yalnız insan)
    alt      : bu güvenin altındaki kutu HİÇ değerlendirilmez
    k_maks   : iz, güvenli görülmeyeli en fazla bu kadar kare yaşar
    kat      : mesafe kapısı = kat × max(kutu boyu) × (kayıp kare sayısı)
    ego/kat_ego : ego-hareket tahmini (v2, bkz. __init__)
    Durum: izler = [dict(cls, kutu, son, hiz, tahmin[, ham])]  son = son GÜVENLİ görülme karesi."""

    def __init__(self, siniflar=(1,), alt=0.10, k_maks=6, kat=2.0, ego=True, kat_ego=1.5):
        self.siniflar = set(int(c) for c in siniflar)
        # alt: tek sayı (tüm sınıflar) ya da {cls: alt} (sınıf başına; eksik sınıf → en düşük değer)
        if isinstance(alt, dict):
            self.alt_sinif = {int(c): float(v) for c, v in alt.items()}
            self.alt = min(self.alt_sinif.values())
        else:
            self.alt_sinif = {}
            self.alt = float(alt)
        self.k_maks = int(k_maks)
        self.kat = float(kat)
        # EGO-HAREKET TAHMİNİ (v2): bu karede güvenli eşleşen izlerin ORTAK kaymasının medyanı
        # (drone hareketi bütün kutulara ortak) eşleşmeyen izleri ilerletir; tek iz varsa kendi
        # son hızı. İlerletilmiş iz için kapı daha DAR (kat_ego × boy × √k) → yanlış ize atlama
        # azalır. Ölçüldü (folder1 insan): aynı boşluk doldurma, şüpheli ekleme 607 → 561.
        self.ego = bool(ego)
        self.kat_ego = float(kat_ego)
        self.izler: List[dict] = []
        self.kare = 0
        self.eklenen = 0            # toplam kabul edilen düşük kutu (telemetri)

    def alt_esik(self, cls: int) -> float:
        return self.alt_sinif.get(int(cls), self.alt)

    @staticmethod
    def _kutu(d: Detection):
        return (d.x1, d.y1, d.x2, d.y2)

    @staticmethod
    def _boy(b):
        return max(b[2] - b[0], b[3] - b[1], 1.0)

    @staticmethod
    def _mesafe(a, b):
        return float(np.hypot((a[0] + a[2] - b[0] - b[2]) / 2.0, (a[1] + a[3] - b[1] - b[3]) / 2.0))

    def _en_yakin_iz(self, b, cls, kullanildi, yalniz_eski):
        """b kutusuna kapı içinde en yakın iz indeksi (yoksa None)."""
        en, end = None, float("inf")
        for ti, t in enumerate(self.izler):
            if ti in kullanildi or t["cls"] != cls:
                continue
            k = self.kare - t["son"]
            if k == 0:                          # bu karede güvenli kutuyla doğmuş/güncellenmiş iz: başka kutu almaz
                continue
            d = self._mesafe(b, t["kutu"])
            boy = max(self._boy(b), self._boy(t["kutu"]))
            if self.ego and t.get("tahmin"):
                kapi = self.kat_ego * boy * float(np.sqrt(max(k, 1)))   # konum zaten ilerletildi
            else:
                kapi = self.kat * boy * max(k, 1)
            if d <= kapi and d < end:
                en, end = ti, d
        return en

    def adim(self, guvenli: List[Detection], dusuk: List[Detection]) -> List[Detection]:
        """guvenli: sınıf eşiğini geçenler (tüm sınıflar; yalnız `siniflar` iz tutar)
           dusuk  : alt ≤ conf < eşik, `siniflar` içinden. -> kabul edilen düşük kutular"""
        self.kare += 1
        i = self.kare
        self.izler = [t for t in self.izler if i - t["son"] <= self.k_maks]
        kullanildi = set()
        kaymalar = []                                       # ego-hareket: ardışık güvenli eşleşmelerin kaymaları
        # 1) güvenli kutular izleri günceller / yeni iz açar
        for d in guvenli:
            if d.cls not in self.siniflar:
                continue
            b = self._kutu(d)
            en = self._en_yakin_iz(b, d.cls, kullanildi, yalniz_eski=False)
            if en is None:
                self.izler.append(dict(cls=d.cls, kutu=b, son=i, hiz=(0.0, 0.0), tahmin=False))
                kullanildi.add(len(self.izler) - 1)
            else:
                t = self.izler[en]
                k = max(i - t["son"], 1)
                onceki = t.get("ham", t["kutu"])            # son GERÇEK gözlem (tahminle kaymamış)
                dx = ((b[0] + b[2]) - (onceki[0] + onceki[2])) / 2.0
                dy = ((b[1] + b[3]) - (onceki[1] + onceki[3])) / 2.0
                if k == 1:
                    kaymalar.append((dx, dy))
                t["hiz"] = (dx / k, dy / k)
                t.update(kutu=b, son=i, tahmin=False)
                t.pop("ham", None)
                kullanildi.add(en)
        # 1b) EGO-HAREKET: eşleşmeyen izleri ortak kaymayla (yoksa kendi hızıyla) ilerlet
        if self.ego:
            g = None
            if len(kaymalar) >= 2:
                arr = np.asarray(kaymalar, dtype=float)
                g = (float(np.median(arr[:, 0])), float(np.median(arr[:, 1])))
            for ti, t in enumerate(self.izler):
                if ti in kullanildi or t["son"] == i:
                    continue
                dx, dy = g if g is not None else t.get("hiz", (0.0, 0.0))
                if "ham" not in t:
                    t["ham"] = t["kutu"]
                kb = t["kutu"]
                t["kutu"] = (kb[0] + dx, kb[1] + dy, kb[2] + dx, kb[3] + dy)
                t["tahmin"] = True
        # 2) düşük kutu: yalnız bu karede güvenli eşleşmesi OLMAYAN eski bir izi sürdürüyorsa
        kabul: List[Detection] = []
        for d in sorted(dusuk, key=lambda x: -x.conf):      # güçlü aday önce iz alır
            if d.conf < self.alt_esik(d.cls):
                continue
            b = self._kutu(d)
            en = self._en_yakin_iz(b, d.cls, kullanildi, yalniz_eski=True)
            if en is not None:
                kullanildi.add(en)
                t = self.izler[en]
                t["kutu"] = b                               # konum güncel, 'son güvenli' DEĞİŞMEZ
                t["tahmin"] = False
                t.pop("ham", None)
                kabul.append(d)
        self.eklenen += len(kabul)
        return kabul


class Detector:
    def __init__(self, model_root: str, modality: str, conf: float = 0.25, imgsz: int = 1280,
                 device: str = "cuda", class_conf: Optional[Dict[int, float]] = None,
                 devam: Optional[dict] = None):
        """
        class_conf: SINIF-BAZLI güven eşiği {cls: eşik}. Verilirse YOLO en düşük eşikte
        koşulur ve her tespit kendi sınıf eşiğine göre elenir (ör. termal: taşıt 0.8,
        insan/UAP/UAİ 0.5). None → tek eşik (conf) tüm sınıflara uygulanır.
        devam: İZ DEVAMI ayarları (dict: siniflar, alt, k_maks, kat) ya da None (kapalı,
        eski davranış birebir). Bkz. dosya başı ve pipeline.IZ_DEVAM.
        """
        from ultralytics import YOLO
        # Düz models/ dizini + modalite öneki: models/rgb_yolo.pt, models/termal_yolo.pt.
        assert modality in ("rgb", "termal"), f"geçersiz modalite: {modality}"
        weight = os.path.join(model_root, f"{modality}_yolo.pt")
        assert os.path.exists(weight), f"{modality}_yolo.pt yok: {weight}"
        self.model = YOLO(weight)
        self.class_conf = class_conf
        # sınıf-bazlı eşikte YOLO'yu en düşük eşikte koştur, sonra sınıfa göre filtrele
        self.conf = min(class_conf.values()) if class_conf else conf
        self.imgsz = imgsz
        self.device = device
        self.names = self.model.names   # {0:'tasit',1:'insan',2:'uap',3:'uai'}
        # İZ DEVAMI: NURON_IZ_DEVAM=0 ile kapatılır
        kapali = os.environ.get("NURON_IZ_DEVAM", "1").strip().lower() in ("0", "false", "kapali")
        self.devam = IzDevam(**devam) if (devam and not kapali) else None
        if self.devam is not None:
            adlar = [f"{self.names.get(c, c)}>={self.devam.alt_esik(c):.2f}" for c in sorted(self.devam.siniflar)]
            print(f"[NURON] Iz devami AKTIF ({modality}): {adlar} · k_maks {self.devam.k_maks} · "
                  f"kapi x{self.devam.kat}{' · ego-hareket x%.1f' % self.devam.kat_ego if self.devam.ego else ''}"
                  f"  (kirpisma onleyici, detector.py)", flush=True)

    def _esik(self, cls: int) -> float:
        return self.class_conf.get(cls, self.conf) if self.class_conf is not None else self.conf

    def detect(self, img: np.ndarray) -> List[Detection]:
        # İz devamı açıksa YOLO daha düşük eşikle koşar (tek geçiş, GPU maliyeti aynı);
        # kapalıysa eskisi gibi self.conf.
        kos_esik = min(self.conf, self.devam.alt) if self.devam is not None else self.conf
        r = self.model.predict(img, conf=kos_esik, imgsz=self.imgsz,
                               device=self.device, verbose=False)[0]
        out: List[Detection] = []
        dusuk: List[Detection] = []
        if r.boxes is None:
            if self.devam is not None:
                self.devam.adim([], [])           # kare sayacı / iz yaşlandırma ilerlesin
            return out
        for b in r.boxes:
            cls = int(b.cls[0])
            cf = float(b.conf[0])
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            if cf >= self._esik(cls):
                out.append(Detection(cls=cls, conf=cf, x1=x1, y1=y1, x2=x2, y2=y2))
            elif (self.devam is not None and cls in self.devam.siniflar and cf >= self.devam.alt_esik(cls)):
                dusuk.append(Detection(cls=cls, conf=cf, x1=x1, y1=y1, x2=x2, y2=y2))
            # aksi halde: sınıf eşiğinin altında → ele (eski davranış)
        if self.devam is not None:
            out = out + self.devam.adim(out, dusuk)
        return dedup_same_class(out, iou_thr=0.8)   # aynı sınıf mükerrer kutu → düşük güvenliyi ele
