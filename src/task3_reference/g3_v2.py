#!/usr/bin/env python3
"""GOREV 3 v2 — ORKESTRATOR ADAPTORU (RGB + TERMAL tek giris).

Eski `session.ReferenceSession` ile AYNI arayuzu sunar, ama altta secilen
v2 boru hatlarini calistirir:

    modality "rgb"    -> rgb2.RgbOturum      (FastSAM + WebSSL + DAM4SAM, esik 0.44)
    modality "termal" -> termal3.hibrit.TermalOturumHibrit  (2026-09-05)
                           referans GRI   -> termal2.TermalOturum (WebSSL gri banka, 0.45/4, SAM2.1-B, SADECE TAKIP)
                           referans RENKLI-> termal3.TermalOturum3 (C-RADIOv3-H, FastSAM temas, ALAN_MAKS,
                                             pencereli kilit, SAM2.1-L)
                         NURON_TERMAL_HIBRIT=0 -> eski termal2.TermalOturum

Eski yol (`session.ReferenceSession`) SILINMEDI; geri donmek icin orchestrator'da
`REFERANS_V2 = False` yapmak yeter.

Kullanim (orchestrator ile ayni):
    ReferenceSessionV2.build_bank(ref_dir, out_path=..., save_crops=True)
    s = ReferenceSessionV2(modality).load_bank(bank_path)
    s.set_schedule([{"object_id": 4, "start": 100, "end": 300}, ...])
    objs = s.process(frame_bgr, frame_idx=137)      # -> [UndefinedObject]
"""
import os, re, glob
import numpy as np, cv2
from src.common.schema import UndefinedObject


def _oid(ad, sira=0):
    m = re.search(r"(\d+)", ad)
    return int(m.group(1)) if m else sira + 1


def _ref_dosyalari(reference_dir):
    """Klasordeki OKUNABILIR referans goruntuleri, sabit (alfabetik) sirada."""
    return sorted(p for p in glob.glob(reference_dir + "/*")
                  if os.path.isfile(p) and cv2.imread(p) is not None)


def stem_oid(reference_dir):
    """dosya-adi(stem) -> banka object_id. build_bank ile AYNI kural, TEK kaynak:
    addaki ilk sayi; sayi yoksa sira; iki dosya ayni sayiya duserse ikincisine bos
    bir numara verilir (kirpimlar birbirinin ustune yazilmasin).
    Aralik <-> referans baglantisi bunun USTUNDE kurulur: sunucu ayni goruntuyu
    2-3 ayri aralikta (ayri url) isteyebilir -> cok url, tek oid (2026-09-02)."""
    harita, dolu = {}, set()
    for i, p in enumerate(_ref_dosyalari(reference_dir)):
        ad = os.path.splitext(os.path.basename(p))[0]
        oid = _oid(ad, i)
        while oid in dolu:
            oid = max(dolu) + 1
        dolu.add(oid)
        harita[ad] = oid
    return harita


class ReferenceSessionV2:
    def __init__(self, modality: str, bank_path: str | None = None,
                 maxside: int | None = None, load_models: bool = True, **_):
        assert modality in ("rgb", "termal"), f"gecersiz modalite: {modality}"
        self.modality = modality
        self.bank_path = bank_path
        self.schedule: list[dict] = []
        self._aktif_oid = None
        self.kirpimlar: dict[int, tuple] = {}      # oid -> (crop_bgr, maske)
        self.son_skor = {}
        if modality == "termal":
            # HIBRIT (2026-09-05, nuron_referance_termal/YONTEM_v3.md): referans GRI ise termal2,
            # RENKLI ise termal3 (RADIO-H + pencereli kilit). 2026 gercek nesne araliklarinda
            # BULMA %64.4 -> %83.3, kesinlik %97.7. Eski yola donmek icin NURON_TERMAL_HIBRIT=0.
            if os.environ.get("NURON_TERMAL_HIBRIT", "1").strip().lower() not in ("0", "false", "kapali"):
                from src.task3_reference.termal3.hibrit import TermalOturumHibrit
                self.o = TermalOturumHibrit()
            else:
                from src.task3_reference.termal2.oturum import TermalOturum
                self.o = TermalOturum()
        else:
            from src.task3_reference.rgb2.oturum import RgbOturum
            self.o = RgbOturum()
        if load_models:
            self.o.yukle()

    # ── oturum basi: referanslari HQ-SAM ile kirp ──────────────
    @staticmethod
    def build_bank(reference_dir: str, out_path: str | None = None,
                   save_crops: bool = True):
        """Referans goruntulerini HQ-SAM ile kirpip <crop_dir>/ref_crops/<oid>.png
        olarak yazar. v2 bankasi kare kare degil, bu kirpimlardan kurulur.
        Zaten kirpilmis gelenler (ref_crops/*_cut.png) YENIDEN KIRPILMAZ."""
        from src.task3_reference.segmenters import HQSAMRef, mask_bbox
        kok = os.path.dirname(out_path or reference_dir)
        crop_dir = os.path.join(kok, "ref_crops")
        os.makedirs(crop_dir, exist_ok=True)
        yollar = _ref_dosyalari(reference_dir)
        oidler = stem_oid(reference_dir)          # url<->id eslemesiyle AYNI kaynak
        hq = None
        for p in yollar:
            ad = os.path.splitext(os.path.basename(p))[0]
            oid = oidler[ad]
            img = cv2.imread(p)
            h, w = img.shape[:2]
            s = 1536/max(h, w) if max(h, w) > 1536 else 1.0
            if s < 1.0:
                img = cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
            if ad.endswith("_cut"):                     # zaten kirpik
                mask = img.max(2) > 10
            else:
                if hq is None:
                    hq = HQSAMRef()
                mask, _, _ = hq.best_mask(img)
            x1, y1, x2, y2 = mask_bbox(mask)
            kes = img[y1:y2, x1:x2].copy()
            if kes.size == 0 or (x2-x1) < 8 or (y2-y1) < 8 or not mask.any():
                # HQ-SAM bos/bozuk maske dondurdu (olculdu: 11 referanstan 2'sinde).
                # NESNEYI KAYBETME: tum goruntuyu kirpim say, banka yine kurulur.
                kes = img.copy()
                print(f"[G3] {ad}: HQ-SAM maskesi bos -> tam goruntu kirpim olarak alindi")
            else:
                kes[~mask[y1:y2, x1:x2]] = 0
            if not cv2.imwrite(os.path.join(crop_dir, f"{oid}.png"), kes):
                print(f"[G3] UYARI: {ad} kirpimi YAZILAMADI (oid {oid})")
        del hq
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return crop_dir

    def load_bank(self, bank_path: str | None = None):
        """<crop_dir>/ref_crops/<oid>.png kirpimlarini okuyup gomme bankasini kurar."""
        yol = bank_path or self.bank_path
        crop_dir = yol if os.path.isdir(yol) else os.path.join(os.path.dirname(yol),
                                                               "ref_crops")
        refs = {}
        for p in sorted(glob.glob(crop_dir + "/*.png")):
            ad = os.path.splitext(os.path.basename(p))[0]
            oid = _oid(ad)
            c, m = self.o.kirpim_oku(p) if hasattr(self.o, "kirpim_oku") else (cv2.imread(p), None)
            refs[oid] = (c, m)
            self.kirpimlar[oid] = (c, m)
        if not refs:
            raise FileNotFoundError(f"referans kirpimi yok: {crop_dir}")
        self.o.banka_kur(refs)
        self.bank_path = crop_dir
        return self

    @property
    def ref_ids(self):
        return sorted(self.o.banka.keys())

    # ── aktif referans plani ───────────────────────────────────
    def set_schedule(self, schedule):
        self.schedule = list(schedule) if schedule else []

    def aralik_degisti(self):
        """Sunucu penceresi degisti: bir sonraki process() aktif(oid)'i YENIDEN cagirir,
        takip sifirlanir. Gerekce: ayni referans nesnesi 2-3 ayri aralikta istenebilir
        (2026: Nesne_10 -> 97-135 ve 382-427); oid ayni kaldigi icin process() icindeki
        `oid != _aktif_oid` kontrolu yeni araligi tek basina GOREMEZ."""
        self._aktif_oid = None

    def active_ids(self, frame_idx):
        if not self.schedule or frame_idx is None:
            return None
        return [s["object_id"] for s in self.schedule
                if int(s.get("start", 0)) <= frame_idx <= int(s.get("end", 10**9))]

    # ── kare kare ──────────────────────────────────────────────
    def process(self, frame_bgr, frame_idx=None, active_ids="auto"):
        if active_ids == "auto":
            active_ids = self.active_ids(frame_idx)
        oid = None
        if active_ids:
            oid = next((i for i in active_ids if i in self.o.banka), None)
        elif len(self.o.banka) == 1:
            oid = next(iter(self.o.banka))
        if oid is None:
            return []
        if oid != self._aktif_oid:          # ARALIK DEGISTI -> takip sifirlanir
            self.o.aktif(oid)
            self._aktif_oid = oid
        kutu = self.o.kare(frame_bgr)
        if kutu is None:
            return []
        x1, y1, x2, y2 = kutu
        return [UndefinedObject(object_id=int(oid),
                                top_left_x=float(x1), top_left_y=float(y1),
                                bottom_right_x=float(x2), bottom_right_y=float(y2))]
