#!/usr/bin/env python3
"""
Görev 1 — Taşıt hareket durumu v2 (0=hareketsiz, 1=hareketli). RGB ve TERMAL AYNI kod.

motion.py (v1, NURON üretim) üzerine yapılan değişiklikler — amaç: kararı ERKENE almak (kullanıcı: "araç
kadraja girdikten 2-3 kare sonra çalışıyor"), termalde çalışır hale getirmek, sahte pozitifleri azaltmak.
Hepsi elle etiketlenmiş GT ile ölçüldü (rgb/2026 4002 kutu-kare, termal/2026 1572; ayrıntı: OKU.md).

  (A) KENAR-GÜVENLİ ÇIPA: kutu kadraj kenarına değiyorsa merkez ZIPLAR (yarım kutu büyür); v1 kenarda hiç
      residual üretmiyordu. v2 kesilen eksende İÇ kenar noktasını (görünen gerçek silüet ucu) çıpa alır,
      DİĞER ekseni yalnız kutunun o eksendeki boyu iki karede sabitse (≤%10) kullanır — boy değişmişse görünen
      parça değişmiştir, merkezi kayar (ölçüldü: eğik araçta 10-15 px) → sıfırlanır; köşede (iki eksen kesik)
      çıpa üretmez. (Boy-sabitlik kuralı GT'de FN'i %25 azalttı, FP değişmedi.) Kenar residual'ında eşikler ×edge_floor_mult.
      Çıpa tanımsızsa v1'deki çoğunluk oyu yedeği.
  (B) HIZLI KARAR: son fast_k(2) residual GÜÇLÜ (eşik×fast_mult) ve YÖN-TUTARLI (cos>fast_cos) → hemen 1;
      TEK residual eşiğin ×fast_k1_mult katıysa → hemen 1. Gerçek hareket yön-tutarlı ve büyük (RGB'de
      hareketli residual medyanı 40-50 px, durgun p99 5-10 px); jitter/yanlış eşleşme bunu üretmez.
  (C) BOŞLUK KORUMASI: YOLO aracı kaçırıp yeniden bulunca residual üretilmez — H tek kare adımıdır, boşlukta
      hesaplanan residual anlamsız ve büyüktü (ölçüldü: 5 kare boşlukta 60 px → sahte hareketli).
  (D) OY TUTARLILIĞI: oy kuralı (pencerede ≥vote_min hit) ayrıca |Σr|/Σ|r| ≥ oy_tutarlilik ister (büyük
      kutulu durgun araçta jitter 4 hit toplayabiliyordu).
  (E) BOYUT KAPISI: eşleşmede aday kutu alanı track kutusunun boyut_orani katı dışındaysa eşlenmez (komşu
      araca atlama → 70 px sahte residual).
  (F) MODALİTEYE ÖZEL ÖLÇÜLMÜŞ EŞİKLER + SABİT ÇALIŞMA ÇÖZÜNÜRLÜĞÜ: termal 640 px karede durgun along p99
      3.5 px, hareketli 5.5-8 px → mutlak 10/4 px eşik (v1) termalde hiçbir hareketi yakalamıyordu (TP 0).
      Gelen kare ve kutular is_w genişliğine ölçeklenir (termal 640, rgb 960) → davranış girdi çözünürlüğünden
      bağımsız (1080p ve 4K aynı yoldan geçer), homografi/LK yükü 1080p'de 4 kat, 4K'da 16 kat düşer.
      960/1280/1920 GT ile karşılaştırıldı: 1080p'de doğruluk aynı (0.985), 4K'da 960 en iyi.
  (G) Pencere 'win' profilden kullanılır (v1'de deque(maxlen=10) sabitti).

Geri kalan her şey (arka plan homografisi RANSAC, merkez residual, epipolar perp/along ayrıştırma, hız-tahminli
iki aşamalı eşleşme, çıkışta çoğunluk oyu yedeği) v1 ile AYNI. Arayüz aynı: MotionEstimator(modality)
.update(img, boxes) → [0/1], .last_info. Ek isteğe bağlı update(H_ext=, shape=) yalnız çevrimdışı hızlı
tarama için (degerlendir.py --hizli); üretimde verilmez.

Sonuç (7 oturum, elle etiketlenmiş GT, 23.679 kutu-kare): rgb/2026 doğruluk 0.946→0.985, sahte pozitif 25→5,
gecikme medyan 5→1 kare; termal/2026 0.985→0.999 (hareketli araç 0/24→22/24); termal/2025_oturum_4 0.957→0.984;
termal ornek_1 0.800→0.931; ornek_2 0.605→0.918; 4K rgb/2025_ot2 0.832→0.921 (FP 1001→35); ot3 0.757→0.939.
NURON'a 2026-09-05 21:41'de entegre edildi (yedek: motion.py.YEDEK_20260905_2141). Ayrıntı ve ölçüm araçları:
/home/adem/Desktop/yolo_padim_arab-hareketli/OKU.md
"""
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


@dataclass
class _Track:
    tid: int
    cx: float
    cy: float
    box: tuple = (0.0, 0.0, 0.0, 0.0)
    vx: float = 0.0
    vy: float = 0.0
    res: deque = field(default_factory=lambda: deque(maxlen=6))   # (rx,ry,ex,ey,kenar)
    mov_n: int = 0
    dec_n: int = 0
    missed: int = 0
    f_last: int = 0       # son eşleştiği kare numarası (boşluk kontrolü)
    bayat: int = 0        # ardışık residual üretilemeyen kare sayısı (H yok / boşluk)
    son_karar: int = 0    # önceki karedeki karar (histerezis seçeneği için)


# SABİT ÇALIŞMA ÇÖZÜNÜRLÜĞÜ: kare ne boyutta gelirse gelsin is_w genişliğine ölçeklenir (kutular da), tüm hesap orada
# yapılır → davranış girdi çözünürlüğünden bağımsız (kullanıcı isteği: termal 640'a kilit, RGB en iyi genişliğe kilit).
# Piksel eşikleri taban_w için ölçüldü; is_w farklıysa is_w/taban_w ile ölçeklenir (perp/along/edge_margin/match_gate).
#   rgb : is_w=960 SEÇİLDİ (960/1280/1920 GT ile karşılaştırıldı: 1080p oturumda doğruluk aynı 0.985, 4K oturumlarda en iyi,
#         hesap yükü en düşük). 1920'de ölçülen durgun along p99 9.6 / perp p99 5.1 px → 960'ta 4.8 / 2.6; eşik perp 3.0, along 6.0.
#   termal: is_w=640 (doğal). Durgun along p99 3.5 px, perp p99 1.3 | hareketli along 5.5-8.2, perp ~1 → perp 2.5, along 4.0.
MOTION_PROFILE = {
    "rgb":    dict(is_w=960, taban_w=960, win=6, min_win=3, match_px=80.0, match_gate=36.0, clahe=False,
                   bg_corners=800, car_corners=40, box_dilate=1.20,
                   perp_floor=3.0, along_floor=6.0, vote_min=4,
                   edge_margin=6.0, vel_ema=0.6,
                   ransac_px=3.0, min_bg=25,
                   edge_floor_mult=1.5, fast_k=2, fast_k_edge=2, fast_mult=1.5, fast_cos=0.5,
                   oy_tutarlilik=0.5,      # oy kuralı: |Σres| / Σ|res| ≥ bu değer (jitter oyları eler; 0 = kapalı)
                   fast_k1_mult=2.5,       # TEK residual eşiğin ×bu katı ise hemen hareketli (None = kapalı)
                   boyut_orani=2.5,        # eşleşme: kutu alan oranı bu katın dışındaysa aday değil (araç atlaması)
                   kenar_diger_eksen="boyut",  # kenarda kesilmeyen eksen: kutu boyu sabitse kullan ('sifir' = hiç kullanma)
                   kenar_boyut_tol=0.10,
                   birikim=0.40,           # YAVAŞ araç: |Σartık| ≥ birikim × Σ|ego| ise hareketli (None = kapalı)
                   birikim_n=4, birikim_win=6, birikim_tutarlilik=0.95, birikim_taban=0.0,
                   cipa="merkez",          # 'oznitelik' = artık kutu içi öznitelik akışından (kutu merkezi yerine)
                   cipa_ic_oran=0.75, cipa_min_nokta=4),      # "sabit" toleransı (oran)
    "termal": dict(is_w=640, taban_w=640, win=7, min_win=4, match_px=80.0, match_gate=45.0, clahe=True,
                   bg_corners=500, car_corners=30, box_dilate=1.25,
                   perp_floor=2.5, along_floor=4.0, vote_min=4,
                   edge_margin=10.0, vel_ema=0.6,
                   ransac_px=3.0, min_bg=18,
                   edge_floor_mult=1.5, fast_k=2, fast_k_edge=2, fast_mult=1.25, fast_cos=0.5,
                   oy_tutarlilik=0.5, fast_k1_mult=2.5, boyut_orani=2.5,
                   kenar_diger_eksen="boyut", kenar_boyut_tol=0.10,
                   birikim=0.50, birikim_n=4, birikim_win=7, birikim_tutarlilik=0.95, birikim_taban=0.0,
                   cipa="merkez", cipa_ic_oran=0.75, cipa_min_nokta=4),
}
_LK = dict(winSize=(21, 21), maxLevel=3, criteria=(3, 30, 0.01))
_UNSET = object()      # update(H_ext=...) verilmedi işareti


class MotionEstimator:
    """Streaming: update(img, vehicle_boxes) → her araç için 0/1. `.last_info` araç başına ayrıntı."""

    def __init__(self, modality: str = "rgb", profile: Optional[dict] = None):
        assert modality in ("rgb", "termal")
        self.modality = modality
        self.p = dict(MOTION_PROFILE[modality]); self.p.update(profile or {})
        k = float(self.p.get("is_w") or self.p.get("taban_w") or 1) / float(self.p.get("taban_w") or 1)
        if abs(k - 1.0) > 1e-6:                      # eşikler taban_w'de ölçüldü → çalışma genişliğine taşı
            for ad in ("perp_floor", "along_floor", "edge_margin", "match_gate"):
                self.p[ad] = self.p[ad] * k
        self.p["_olcek"] = k
        self.win = self.p["win"]; self.min_win = self.p["min_win"]
        # artık tamponu: oy penceresi (win) ile birikim penceresi (birikim_win) uzunundan
        self.tampon = max(self.win, int(self.p.get("birikim_win") or 0))
        self._clahe = cv2.createCLAHE(3.0, (8, 8)) if (cv2 and self.p["clahe"]) else None
        self.prev_gray: Optional[np.ndarray] = None
        self.tracks: list[_Track] = []
        self._next_id = 0
        self.last_info: list[dict] = []
        self._gercek = True
        self.last_H: Optional[np.ndarray] = None
        self.last_bg = None            # (p0, p1) arka plan LK çiftleri — son kare
        self._bg_res = None            # arka plan noktalarının H altındaki artığı (yerel düzeltme)
        self._frame_no = 0

    # ---------------- görüntü / akış / homografi (v1 ile aynı) ----------------
    def _prep(self, img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        return self._clahe.apply(g) if self._clahe is not None else g

    def _flow(self, pts):
        if pts is None or len(pts) == 0:
            return None, None
        p0 = pts.reshape(-1, 1, 2).astype(np.float32)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, self._cur_gray, p0, None, **_LK)
        if p1 is None:
            return None, None
        st = st.reshape(-1).astype(bool)
        return p0.reshape(-1, 2)[st], p1.reshape(-1, 2)[st]

    def _bg_homography(self, boxes):
        H, W = self.prev_gray.shape[:2]
        mask = np.full((H, W), 255, np.uint8)
        d = self.p["box_dilate"]
        for (x1, y1, x2, y2) in boxes:
            bw, bh = (x2 - x1), (y2 - y1)
            mx, my = bw * (d - 1) / 2, bh * (d - 1) / 2
            ax1, ay1 = max(0, int(x1 - mx)), max(0, int(y1 - my))
            ax2, ay2 = min(W, int(x2 + mx)), min(H, int(y2 + my))
            mask[ay1:ay2, ax1:ax2] = 0
        pts = cv2.goodFeaturesToTrack(self.prev_gray, self.p["bg_corners"], 0.01, 7, mask=mask)
        if pts is None or len(pts) < self.p["min_bg"]:
            return None
        p0, p1 = self._flow(pts.reshape(-1, 2))
        if p0 is None or len(p0) < self.p["min_bg"]:
            return None
        Hm, inl = cv2.findHomography(p0, p1, cv2.RANSAC, self.p["ransac_px"])
        if Hm is None or int(inl.sum()) < self.p["min_bg"]:
            return None
        self.last_bg = (p0.astype(np.float32), p1.astype(np.float32))     # tüm LK çiftleri (yerel düzeltme için)
        return Hm

    @staticmethod
    def _warp_pt(pt, Hm):
        return cv2.perspectiveTransform(np.array([[[pt[0], pt[1]]]], np.float32), Hm).reshape(2)

    def _at_edge(self, box):
        H, W = self._cur_gray.shape[:2]
        m = self.p["edge_margin"]
        return (box[0] <= m or box[1] <= m or box[2] >= W - m or box[3] >= H - m)

    # ---------------- v2: kenar-güvenli çıpa ----------------
    def _anchor_pair(self, pb, cb):
        """Önceki kutu pb ve şimdiki kutu cb için ORTAK görünür çıpa noktaları + güvenilir eksen maskesi.
        Dönüş: (prev_pt, cur_pt, tür, maske) veya (None, None, tür, None).
          tür  : 'c' merkez (kenarda değil), 'k' kenar çıpası.
          maske: (mx, my) 1/0 — residual'ın hangi bileşeni güvenilir.
        Kenarda kesilen eksende İÇ kenar (görünen gerçek silüet ucu) güvenilirdir; DİĞER eksende hiçbir
        şey güvenilir değil (eğik araçta görünen parçanın merkezi/uçları kesim ilerledikçe kayar → ölçüldü:
        alt kenarda çıkan durgun araçta x residual 10-15 px). İki eksen de kesikse (köşe) çıpa yok."""
        H, W = self._cur_gray.shape[:2]
        m = self.p["edge_margin"]
        right = pb[2] >= W - m or cb[2] >= W - m
        left = pb[0] <= m or cb[0] <= m
        bottom = pb[3] >= H - m or cb[3] >= H - m
        top = pb[1] <= m or cb[1] <= m
        xk, yk = (left or right), (top or bottom)
        if (left and right) or (top and bottom):
            return None, None, "yok", None
        if xk and yk:                                   # KÖŞE: iki eksen de kesik
            if not self.p.get("kose_cipa", False):
                return None, None, "yok", None
            ax = (pb[0], cb[0]) if right else (pb[2], cb[2])
            ay = (pb[1], cb[1]) if bottom else (pb[3], cb[3])
            return (ax[0], ay[0]), (ax[1], ay[1]), "k", (1.0, 1.0)
        if right:   ax = (pb[0], cb[0])                              # görünen SOL kenar
        elif left:  ax = (pb[2], cb[2])                              # görünen SAĞ kenar
        else:       ax = ((pb[0] + pb[2]) / 2, (cb[0] + cb[2]) / 2)
        if bottom:  ay = (pb[1], cb[1])
        elif top:   ay = (pb[3], cb[3])
        else:       ay = ((pb[1] + pb[3]) / 2, (cb[1] + cb[3]) / 2)
        # kesilmeyen eksen: 'sifir' → hiç kullanma; 'boyut' → kutunun o eksendeki boyu iki karede sabitse
        # (görünen parça değişmemiş → merkezi güvenilir) kullan, boy değişmişse sıfırla.
        tol = self.p.get("kenar_boyut_tol", 0.10); kip = self.p.get("kenar_diger_eksen", "boyut")
        if xk:
            h0, h1 = pb[3] - pb[1], cb[3] - cb[1]
            y_ok = kip == "boyut" and abs(h1 - h0) <= max(2.0, tol * h0)
            tur, maske = "k", (1.0, 1.0 if y_ok else 0.0)
        elif yk:
            w0, w1 = pb[2] - pb[0], cb[2] - cb[0]
            x_ok = kip == "boyut" and abs(w1 - w0) <= max(2.0, tol * w0)
            tur, maske = "k", (1.0 if x_ok else 0.0, 1.0)
        else:    tur, maske = "c", (1.0, 1.0)
        return (ax[0], ay[0]), (ax[1], ay[1]), tur, maske

    def _yerel_sapma(self, pt, box):
        """Aracın çevresindeki arka plan noktalarının H altındaki MEDYAN artığı (yerel misfit: distorsiyon,
        zemin düzlem sapması). Yeterli nokta yoksa None. Yarıçap: yerel_R_kat × kutu köşegeni (sınırlı)."""
        if self._bg_res is None:
            return None
        p0, r = self._bg_res
        W = self._cur_gray.shape[1]
        diag = float(np.hypot(box[2] - box[0], box[3] - box[1]))
        R = min(max(self.p.get("yerel_R_kat", 2.0) * diag, 0.04 * W), 0.25 * W)
        d = np.hypot(p0[:, 0] - pt[0], p0[:, 1] - pt[1])
        sel = d < R
        if int(sel.sum()) < self.p.get("yerel_min", 6):
            return None
        return np.median(r[sel], axis=0)

    def _arac_akisi(self, kutular_onceki, hazir=None, ham=False):
        """ARAÇ ÇIPASI = kutunun İÇİNDEKİ öznitelikler (kutu merkezi değil).
        Kutu merkezi, drone araç üstünden geçerken aracın görünen silüeti değiştiği için durgun araçta bile
        ego ile orantılı SİSTEMATİK kayar (ölçüm: durgun |Σartık|/Σ|ego| p90 0.13-0.29). İçerideki nokta
        öznitelikleri aracın yüzeyini takip eder, bu kaymayı taşımaz.
        Dönüş: track sırasıyla [(p0, p1) veya None]."""
        if not kutular_onceki: return []
        if hazir is None and self.prev_gray is None: return []
        g0, g1 = self.prev_gray, self._cur_gray
        H, W = self._cur_gray.shape[:2]
        mask = np.zeros((H, W), np.uint8); kut = []
        for b in kutular_onceki:
            cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
            f = float(self.p.get("cipa_ic_oran", 0.75))
            hw, hh = (b[2] - b[0]) / 2.0 * f, (b[3] - b[1]) / 2.0 * f
            q = (int(max(0, cx - hw)), int(max(0, cy - hh)), int(min(W, cx + hw)), int(min(H, cy + hh)))
            kut.append(q)
            if q[2] - q[0] >= 4 and q[3] - q[1] >= 4: mask[q[1]:q[3], q[0]:q[2]] = 255
        if hazir is not None:                       # çevrimdışı önbellek: noktalar hazır, yalnız kutulara dağıt
            a0, a1 = np.asarray(hazir[0], np.float32), np.asarray(hazir[1], np.float32)
            if len(a0) == 0: return [None] * len(kutular_onceki)
            enaz0 = int(self.p.get("cipa_min_nokta", 4)); cik = []
            for q in kut:
                m = (a0[:, 0] >= q[0]) & (a0[:, 0] < q[2]) & (a0[:, 1] >= q[1]) & (a0[:, 1] < q[3])
                cik.append((a0[m], a1[m]) if int(m.sum()) >= enaz0 else None)
            return cik
        if not mask.any():
            return (np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)) if ham else [None] * len(kutular_onceki)
        nk = int(self.p.get("car_corners", 40)) * max(1, len(kutular_onceki))
        p0 = cv2.goodFeaturesToTrack(g0, maxCorners=nk, qualityLevel=0.01, minDistance=3, mask=mask, blockSize=5)
        bos = (np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)) if ham else [None] * len(kutular_onceki)
        if p0 is None or len(p0) < 1: return bos
        p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None, **_LK)
        if p1 is None: return bos
        p0b, st2, _ = cv2.calcOpticalFlowPyrLK(g1, g0, p1, None, **_LK)
        if p0b is None: return bos
        a0, a1, ab = p0.reshape(-1, 2), p1.reshape(-1, 2), p0b.reshape(-1, 2)
        ok = (st.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm(a0 - ab, axis=1) < 1.0)
        a0, a1 = a0[ok], a1[ok]
        if ham: return a0, a1                       # önbellek üretimi: ham nokta çiftleri
        enaz = int(self.p.get("cipa_min_nokta", 4))
        out = []
        for q in kut:
            m = (a0[:, 0] >= q[0]) & (a0[:, 0] < q[2]) & (a0[:, 1] >= q[1]) & (a0[:, 1] < q[3])
            out.append((a0[m], a1[m]) if int(m.sum()) >= enaz else None)
        return out

    def _oznitelik_residual(self, cift, Hm):
        """Araç noktalarının H altındaki MEDYAN artığı (aykırı noktalara dayanıklı)."""
        p0, p1 = cift
        w = cv2.perspectiveTransform(p0.reshape(-1, 1, 2).astype(np.float32), Hm).reshape(-1, 2)
        r = np.median(p1 - w, axis=0)
        ego = np.median(w - p0, axis=0)
        emag = float(np.hypot(*ego))
        eu = ego / emag if emag > 1.0 else np.array([0.0, 0.0])
        return float(r[0]), float(r[1]), float(eu[0]), float(eu[1]), emag

    def _residual(self, prev_pt, cur_pt, Hm):
        exp = self._warp_pt(prev_pt, Hm)
        res = np.array(cur_pt) - exp
        ego = exp - np.array(prev_pt)
        emag = float(np.hypot(*ego))
        eu = ego / emag if emag > 1.0 else np.array([0.0, 0.0])
        return float(res[0]), float(res[1]), float(eu[0]), float(eu[1]), emag

    # ---------------- ana döngü ----------------
    def update(self, img, boxes_xyxy, H_ext=_UNSET, shape=None, bg_ext=None, arac_ext=None) -> list[int]:
        """img: BGR kare. H_ext verilirse (np.ndarray veya None) arka plan homografisi HESAPLANMAZ, o kullanılır;
        bu durumda img None olabilir (shape=(H,W) verilir) → görüntü okumasız hızlı tarama."""
        if cv2 is None:
            return [0] * len(boxes_xyxy)
        # --- SABİT ÇALIŞMA ÇÖZÜNÜRLÜĞÜ: gelen kare is_w genişliğine ölçeklenir, kutular da ---
        W_in = img.shape[1] if img is not None else shape[1]
        is_w = self.p.get("is_w")
        s = (float(is_w) / float(W_in)) if is_w else 1.0
        if abs(s - 1.0) < 1e-3: s = 1.0
        self._s = s
        self._gercek = img is not None       # gerçek görüntü var mı (öznitelik çıpası yalnız o zaman)
        if img is not None:
            if s != 1.0:
                img = cv2.resize(img, (int(is_w), max(1, int(round(img.shape[0] * s)))),
                                 interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR)
            self._cur_gray = self._prep(img)
        else:
            sh = (max(1, int(round(shape[0] * s))), int(is_w)) if s != 1.0 else tuple(shape)
            self._cur_gray = np.zeros(sh, np.uint8)             # yalnız boyut için
        if s != 1.0:
            if H_ext is not _UNSET and H_ext is not None:       # dış H girdi (doğal) koordinatındaysa çalışma koordinatına taşı
                S = np.diag([s, s, 1.0]); H_ext = S @ np.asarray(H_ext, dtype=np.float64) @ np.linalg.inv(S)
            if bg_ext is not None:
                bg_ext = (np.asarray(bg_ext[0], np.float32) * s, np.asarray(bg_ext[1], np.float32) * s)
            if arac_ext is not None:
                arac_ext = (np.asarray(arac_ext[0], np.float32) * s, np.asarray(arac_ext[1], np.float32) * s)
        self._frame_no += 1
        boxes_xyxy = [tuple(float(v) * s for v in b) for b in boxes_xyxy]
        centers = [((x1 + x2) / 2, (y1 + y2) / 2) for (x1, y1, x2, y2) in boxes_xyxy]

        if self.prev_gray is None:
            self.tracks = []
            for (cx, cy), b in zip(centers, boxes_xyxy):
                tr = _Track(self._next_id, cx, cy, b); tr.res = deque(maxlen=self.tampon); tr.f_last = self._frame_no
                self.tracks.append(tr); self._next_id += 1
            self.prev_gray = self._cur_gray
            self.last_info = [dict(tid=t.tid, speed=0.0, dir=None, n=0, Hok=False, edge=False, ratio=0.0, anchor="-")
                              for t in self.tracks]
            return [0] * len(centers)

        self.last_bg = None
        Hm = self._bg_homography(boxes_xyxy) if H_ext is _UNSET else H_ext
        if bg_ext is not None:
            self.last_bg = bg_ext
        self.last_H = Hm
        # YEREL DÜZELTME hazırlığı: arka plan noktalarının H altındaki artığı (misfit alanı)
        self._bg_res = None
        if Hm is not None and self.last_bg is not None and self.p.get("yerel", False):
            p0, p1 = self.last_bg
            if len(p0) >= self.p.get("yerel_min", 6):
                w0 = cv2.perspectiveTransform(p0.reshape(-1, 1, 2), Hm).reshape(-1, 2)
                self._bg_res = (p0, p1 - w0)

        pred_c = []
        for tr in self.tracks:
            base = self._warp_pt((tr.cx, tr.cy), Hm) if Hm is not None else np.array([tr.cx, tr.cy])
            pred_c.append(base + np.array([tr.vx, tr.vy]))

        Wpx = self._cur_gray.shape[1]
        gate1 = self.p["match_gate"]
        gate2 = max(1.8 * gate1, 0.05 * Wpx)
        assign = [None] * len(centers); tused = set(); dused = set()

        alan = [max(1.0, (b[2]-b[0]) * (b[3]-b[1])) for b in boxes_xyxy]
        bo = self.p.get("boyut_orani", None)

        def _greedy(gate, i_pool, j_pool):
            pairs = []
            for i in i_pool:
                cx, cy = centers[i]
                for j in j_pool:
                    dd = np.hypot(cx - pred_c[j][0], cy - pred_c[j][1])
                    if dd >= gate:
                        continue
                    if bo:                                   # farklı boyutta kutu = başka araç → eşleme
                        tb = self.tracks[j].box; ta = max(1.0, (tb[2]-tb[0]) * (tb[3]-tb[1]))
                        oran = alan[i] / ta
                        if oran > bo or oran < 1.0 / bo:
                            continue
                    pairs.append((dd, i, j))
            pairs.sort()
            for dd, i, j in pairs:
                if i in dused or j in tused:
                    continue
                assign[i] = j; dused.add(i); tused.add(j)

        _greedy(gate1, range(len(centers)), range(len(self.tracks)))
        rem_i = [i for i in range(len(centers)) if i not in dused]
        rem_j = [j for j in range(len(self.tracks)) if j not in tused]
        if rem_i and rem_j:
            _greedy(gate2, rem_i, rem_j)

        # araç öznitelik akışı (çıpa='oznitelik'): TÜM araç kutuları için tek goodFeatures + tek LK
        arac_akis = []
        if self.p.get("cipa") == "oznitelik" and self.tracks and (self._gercek or arac_ext is not None):
            arac_akis = self._arac_akisi([tr.box for tr in self.tracks], arac_ext)
        self._arac_akis_ham = arac_akis
        result, info, alive = [], [], set()
        for i, ((cx, cy), box) in enumerate(zip(centers, boxes_xyxy)):
            j = assign[i]
            edge = self._at_edge(box)
            if j is None:
                tr = _Track(self._next_id, cx, cy, box); tr.res = deque(maxlen=self.tampon); tr.f_last = self._frame_no
                self._next_id += 1
                self.tracks.append(tr); j = len(self.tracks) - 1
                result.append(0)
                info.append(dict(tid=tr.tid, speed=0.0, dir=None, n=0, Hok=Hm is not None, edge=edge, ratio=0.0, anchor="-"))
                alive.add(j); continue
            tr = self.tracks[j]; alive.add(j); tr.missed = 0

            res, tur = None, "-"
            bosluk = self._frame_no - tr.f_last          # 1 = ardışık kare; >1 = YOLO arada kaçırdı
            tr.f_last = self._frame_no
            if Hm is not None and bosluk == 1 and j < len(arac_akis) and arac_akis[j] is not None:
                rx, ry, ex, ey, emag = self._oznitelik_residual(arac_akis[j], Hm)
                res = (rx, ry, ex, ey, 1 if edge else 0, emag); tur = "o"
            elif Hm is not None and bosluk == 1:         # H yalnız ardışık kare çifti için geçerli → boşlukta residual YOK
                pp, cp, tur, maske = self._anchor_pair(tr.box, box)
                if pp is not None:
                    rx, ry, ex, ey, emag = self._residual(pp, cp, Hm)
                    if self._bg_res is not None:
                        sap = self._yerel_sapma(pp, tr.box)
                        if sap is not None:
                            rx, ry = rx - float(sap[0]), ry - float(sap[1])
                    res = (rx * maske[0], ry * maske[1], ex, ey, 1 if tur == "k" else 0, emag)
            if res is not None:
                a = self.p["vel_ema"]
                tr.vx = (1 - a) * tr.vx + a * res[0]
                tr.vy = (1 - a) * tr.vy + a * res[1]
                tr.res.append(res); tr.bayat = 0
            else:
                tr.bayat += 1
                if tr.bayat > self.win:            # uzun süre residual yok (H kurulamıyor) → bayat pencereyi boşalt
                    tr.res.clear(); tr.vx = tr.vy = 0.0
            tr.cx, tr.cy, tr.box = cx, cy, box

            mov, sp, ang, why = _decide(tr.res, self.min_win, self.p, float(np.hypot(box[2]-box[0], box[3]-box[1])))
            hz = self.p.get("histerezis")
            if hz and mov == 0 and tr.son_karar == 1 and why and why[0] in ("yok", "oy-tutarsiz") \
                    and len(tr.res) >= self.min_win and why[1] >= self.p["vote_min"] - hz:
                mov, why = 1, ("histerezis", why[1])
            tr.son_karar = mov
            if len(tr.res) >= 2:
                tr.dec_n += 1; tr.mov_n += mov
            ratio = (tr.mov_n / tr.dec_n) if tr.dec_n > 0 else 0.0
            if bosluk > 1: tur = f"bosluk{bosluk}"
            if res is None and edge and Hm is not None and bosluk == 1:
                # çıpa tanımsız (kutu karşılıklı kenarlara değiyor) → görülme boyunca çoğunluk oyu
                out = 1 if (tr.dec_n > 0 and 2 * tr.mov_n > tr.dec_n) else 0
                why = ("cogunluk",)
            else:
                out = mov
            result.append(int(out))
            info.append(dict(tid=tr.tid, speed=sp, dir=ang, n=len(tr.res), Hok=Hm is not None, edge=edge,
                             ratio=ratio, why=why, anchor=tur))

        for j, tr in enumerate(self.tracks):
            if j not in alive:
                tr.missed += 1
        self.tracks = [tr for tr in self.tracks if tr.missed < 6]
        self.prev_gray = self._cur_gray
        self.last_info = info
        return result


def _decide(res, min_win, p, diag=0.0):
    """(A) oy: pencerede ≥vote_min hit (n≥min_win). (B) hızlı yol: son fast_k residual güçlü + yön-tutarlı.
    (C) birikim: pencere boyunca biriken yer değiştirme egoya oranla büyük + yön-tutarlı (yavaş araç).
    Dönüş: (0/1, öz hız px/kare, yön derece, neden)."""
    tum = list(res)                       # birikim penceresi (uzun olabilir)
    res = tum[-p["win"]:]                 # oy/hızlı yollar: HER ZAMAN son win artık
    n = len(res)
    pf, af = p["perp_floor"], p["along_floor"]
    # BOYUTA GÖRE EŞİK: paralaks ve kutu jitter'ı görünen boyutla büyür → eşik = max(taban, kat × köşegen)
    pf = max(pf, (p.get("boyut_kat_perp") or 0.0) * diag)
    af = max(af, (p.get("boyut_kat_along") or 0.0) * diag)
    k_par = p.get("k_par", 0.0) or 0.0       # paralaks ∝ ego akışı: along eşiği en az k_par·|ego| (0 = kapalı)
    if n < 2:
        k1 = p.get("fast_k1_mult")
        if n == 1 and k1:                                # tek ama ÇOK güçlü residual → hemen hareketli
            rx, ry, ex, ey, kenar, emag = res[-1]
            mult = (p["edge_floor_mult"] if kenar else 1.0) * k1
            af_e = max(af * mult, k_par * emag * k1)
            r = np.array([rx, ry]); e = np.array([ex, ey])
            if np.hypot(ex, ey) > 1e-6:
                along = abs(float(r @ e)); perp = float(np.hypot(*(r - (r @ e) * e)))
            else:
                along, perp = 0.0, float(np.hypot(rx, ry))
            if perp > pf * mult or along > af_e:
                return 1, float(np.hypot(rx, ry)), float(np.degrees(np.arctan2(ry, rx))), ("tek", 1)
        return 0, 0.0, None, ("n<2",)
    hits, strong, vecs, net = 0, [], [], np.zeros(2)
    ego_top = 0.0                                # pencere boyunca ego akışının toplamı (paralaks ölçeği)
    perp_isaretli = []                           # ego'ya DİK bileşen (işaretli) — yavaş-tutarlı yol için
    for rx, ry, ex, ey, kenar, emag in res:
        ego_top += float(emag)
        mult = p["edge_floor_mult"] if kenar else 1.0
        af_e = max(af * mult, k_par * emag)
        r = np.array([rx, ry]); e = np.array([ex, ey]); net += r
        if np.hypot(ex, ey) > 1e-6:
            along = abs(float(r @ e)); perp = float(np.hypot(*(r - (r @ e) * e)))
            perp_isaretli.append(float(ex * ry - ey * rx))
        else:
            along, perp = 0.0, float(np.hypot(rx, ry))
            perp_isaretli.append(float(np.hypot(rx, ry)) * (1.0 if (rx + ry) >= 0 else -1.0))
        hits += int(perp > pf * mult or along > af_e)
        strong.append(perp > p["fast_mult"] * pf * mult or along > p["fast_mult"] * af_e)
        vecs.append(r)
    speed = float(np.hypot(*net)) / n
    ang = float(np.degrees(np.arctan2(net[1], net[0])))
    if n >= min_win and hits >= p["vote_min"]:
        top = sum(float(np.linalg.norm(v)) for v in vecs)
        tut = float(np.linalg.norm(net)) / top if top > 1e-6 else 0.0
        if tut >= p.get("oy_tutarlilik", 0.0):          # yön-tutarlı oy: gerçek hareket; jitter oyları elenir
            return 1, speed, ang, ("oy", hits)
        return 0, speed, ang, ("oy-tutarsiz", hits)
    k = p["fast_k"]
    if any(r[4] for r in list(res)[-k:]):          # son residual'larda kenar var -> daha uzun kanıt iste
        k = p.get("fast_k_edge", 3)
    if n >= k and all(strong[-k:]):
        ok = True
        for a, b in zip(vecs[-k:], vecs[-k + 1:]):
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na < 1e-6 or nb < 1e-6 or float(a @ b) / (na * nb) < p["fast_cos"]:
                ok = False; break
        if ok:
            return 1, speed, ang, ("hizli", k)
    # BİRİKİM: yavaş ama yön-tutarlı hareket. Kare başına artık eşiğin altında kalsa da pencere boyunca
    # biriken yer değiştirme büyükse araç gerçekten kayıyordur. Eşik EGO AKIŞINA oranlıdır: paralaks ve H
    # hatası ego hareketiyle büyür, gerçek araç hareketi büyümez. Ölçüm: durgun net/ego p90 0.13-0.29,
    # kaçırılan hareketli p50 0.23-0.41 (sonuclar/birikim.json, 4 RGB oturumu, 35k araç-kare).
    bir = p.get("birikim")
    if bir and len(tum) >= p.get("birikim_n", 4):
        net_b = np.zeros(2); top_m = 0.0; ego_b = 0.0
        for rx, ry, ex, ey, kenar, emag in tum:
            net_b += np.array([rx, ry]); top_m += float(np.hypot(rx, ry)); ego_b += float(emag)
        net_m = float(np.linalg.norm(net_b)); ego_top = ego_b
        kat = p["edge_floor_mult"] if tum[-1][4] else 1.0
        if (top_m > 1e-6 and ego_top > 1e-6 and net_m >= bir * kat * ego_top
                and net_m / top_m >= p.get("birikim_tutarlilik", 0.95)
                and net_m >= p.get("birikim_taban", 0.0)):
            return 1, speed, ang, ("birikim", round(net_m, 1))
    yk = p.get("yavas_kat")
    if yk and n >= p.get("yavas_n", 4):             # YAVAŞ ama yön-tutarlı dik hareket: birikimli dik yer değiştirme
        top_p = sum(abs(v) for v in perp_isaretli); net_p = abs(sum(perp_isaretli))
        if top_p > 1e-6 and net_p > yk * pf and net_p / top_p >= p.get("yavas_tutarlilik", 0.8):
            return 1, speed, ang, ("yavas", round(net_p, 1))
    return 0, speed, ang, ("yok", hits)
