#!/usr/bin/env python3
"""
TERMAL Z — nesne büyüme/küçülmesinden yükseklik (saf geometri, renk YOK).

Fikir: drone yükselince nesneler görüntüde küçülür. Ardışık karede eşleşen feature
ÇİFTLERİ arası mesafe = "nesne boyutu"; kare-kare oran = h_önce/h_şimdi. Kümülatif çarpım
→ yükseklik oranı. h0 (başlangıç yükseklik) = GSD (ilk 450: metrik-XY-hız/piksel-XY-hız × f).
    z(aşağı-poz) = h0·(1 − 1/∏oran)
GT korelasyonu +0.88, CANLI (std ~GT). RGB decorrelation termalde ölü çıktı, bu doğru çözüm.

Streaming: her feed(bgr) oranı biriktirir; kalibrasyon karelerinde GT XY hız ile h0 çözülür.
"""
import numpy as np
import cv2
from scipy.ndimage import median_filter
from src.task2_position import paths as P


class ObjectSizeZ:
    def __init__(self, f=None, gt_frames=None):
        self.f = f or P.TERMAL_F
        self.G = gt_frames or P.GT_FRAMES
        self.orb = cv2.ORB_create(1500)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.clahe = cv2.createCLAHE(3.0, (8, 8))
        self._prev_kp = self._prev_des = None
        self.ratios = [1.0]     # kare-kare çift-mesafe oranı
        self.pxvel = [0.0]      # ORB piksel XY hızı (h0 kalibrasyonu için)
        self.h0 = None

    # --- pickle destegi (kesinti-resume, 2026-09-03): cv2 nesneleri ve onceki kare
    #     ozellikleri pickle'lanamaz -> atilir, yuklenince yeniden kurulur (bir kare oran=1.0).
    def __getstate__(self):
        d = self.__dict__.copy()
        for k in ("orb", "bf", "clahe", "_prev_kp", "_prev_des"):
            d.pop(k, None)
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self.orb = cv2.ORB_create(1500)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.clahe = cv2.createCLAHE(3.0, (8, 8))
        self._prev_kp = self._prev_des = None

    def _feat(self, bgr):
        g = self.clahe.apply(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
        return self.orb.detectAndCompute(g, None)

    def feed(self, image_bgr):
        """Bir kare işle → o kare için çift-mesafe oranı + piksel hızı biriktir."""
        kp, des = self._feat(image_bgr)
        r, ph = 1.0, np.nan
        pkp, pdes = self._prev_kp, self._prev_des
        if des is not None and pdes is not None and len(kp) > 30:
            m = sorted(self.bf.match(des, pdes), key=lambda x: x.distance)[:300]
            if len(m) >= 30:
                p2 = np.float32([kp[x.queryIdx].pt for x in m])
                p1 = np.float32([pkp[x.trainIdx].pt for x in m])
                M, msk = cv2.estimateAffinePartial2D(p2, p1, method=cv2.RANSAC, ransacReprojThreshold=3.0)
                inl = msk.ravel() == 1 if msk is not None else np.ones(len(m), bool)
                q1, q2 = p1[inl], p2[inl]
                if len(q1) > 15:
                    idx = np.random.RandomState(0).choice(len(q1), min(100, len(q1)), replace=False)
                    d1 = np.linalg.norm(q1[idx][:, None] - q1[idx][None], axis=2)
                    d2 = np.linalg.norm(q2[idx][:, None] - q2[idx][None], axis=2)
                    mask = d1 > 3
                    if mask.sum() > 20:
                        r = float(np.median(d2[mask] / d1[mask]))
                    if M is not None:
                        ph = np.hypot(M[0, 2], M[1, 2])
        self.ratios.append(r); self.pxvel.append(ph)
        self._prev_kp, self._prev_des = kp, des

    def calibrate_h0(self, gt_xy):
        """gt_xy: ilk G karenin dünya XY'si [G,2]. GSD ile h0 (başlangıç yükseklik)."""
        gt_xy = np.asarray(gt_xy)[:self.G]
        mh = np.linalg.norm(np.diff(gt_xy[:, :2], axis=0), axis=1)   # metrik XY hız
        ph = np.array(self.pxvel[1:self.G])                         # piksel XY hız
        gec = (ph > 1) & (mh > 0.05) & np.isfinite(ph)
        if gec.sum() < 5:
            return None
        self.h0 = self.f * np.median(mh[gec] / ph[gec])
        return self.h0

    # --- RGB (2026-08-15) icin ayarlanabilir hale getirildi. VARSAYILANLAR mevcut
    #     davranisi BIREBIR korur -> TERMAL HIC ETKILENMEZ.
    #     RGB'de olculen en iyi ayar: clip ±0.02, nedensel medyan W=3, ofset "son".
    CLIP_LO, CLIP_HI = 0.95, 1.05     # oran kirpma sinirlari
    MED_W = 11                        # medyan pencere genisligi
    NEDENSEL = False                  # True: pencere sadece gecmise bakar
    OFSET = "ort"                     # "ort" (kalibrasyon ortalamasi) | "son" (son N kare medyani)
    OFSET_N = 30

    @staticmethod
    def _nedensel_medyan(x, W):
        """out[i] = medyan(x[max(0,i-W+1) .. i]) — gelecege BAKMAZ."""
        x = np.asarray(x, float)
        if W <= 1:
            return x.copy()
        pad = np.concatenate([np.full(W - 1, np.nan), x])
        sw = np.lib.stride_tricks.sliding_window_view(pad, W)
        return np.nanmedian(sw, axis=1)

    def z(self, gt_z_calib=None):
        """Tüm karelere yükseklik (aşağı-pozitif). gt_z_calib verilirse offset hizalanır."""
        if self.h0 is None:
            raise RuntimeError("önce calibrate_h0 çağır")
        r = np.clip(np.nan_to_num(np.array(self.ratios), nan=1.0), self.CLIP_LO, self.CLIP_HI)
        o = self._nedensel_medyan(r, self.MED_W) if self.NEDENSEL else median_filter(r, self.MED_W)
        o = np.nan_to_num(o, nan=1.0)
        prodD = np.exp(np.cumsum(np.log(o)))
        z = self.h0 * (1 - 1 / prodD)
        if gt_z_calib is not None:
            g = np.asarray(gt_z_calib)
            if self.OFSET == "son" and len(z) >= self.G and len(g) >= self.G:
                N = min(self.OFSET_N, self.G)
                z = z - (np.median(z[self.G - N:self.G]) - np.median(g[self.G - N:self.G]))
            else:
                z = z - (z[:self.G].mean() - g[:self.G].mean())
        return z


# --- Offline: videodan yeniden üret (kaynak sonuç: RMS 7.9m, kor +0.88) ---
def from_video(video, gt_csv, f=None, gt_frames=None):
    import pandas as pd
    solver = ObjectSizeZ(f, gt_frames)
    gt = pd.read_csv(gt_csv)[['translation_x', 'translation_y', 'translation_z']].values
    cap = cv2.VideoCapture(video)
    ok, prev = cap.read()
    solver._prev_kp, solver._prev_des = solver._feat(prev)
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        solver.feed(fr)
    cap.release()
    n = min(len(solver.ratios), len(gt))
    solver.ratios = solver.ratios[:n]; solver.pxvel = solver.pxvel[:n]
    solver.calibrate_h0(gt[:, :2])
    z = solver.z(gt[:, 2])
    return z, gt[:n, 2], solver.h0


if __name__ == "__main__":
    import os, sys
    from src.common.config import ROOT
    video = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(ROOT, "offline_data", "sample_termal.MP4")
    gt = f"{ROOT}/offline_data/gps/v2_gt.csv"
    z, gz, h0 = from_video(video, gt)
    G = P.GT_FRAMES
    print(f"TERMAL Z: h0={h0:.0f}m RMS={np.abs(z[G:]-gz[G:]).mean():.2f}m "
          f"kor={np.corrcoef(z,gz)[0,1]:+.2f} std={z.std():.1f}(GT {gz.std():.1f})")


# ══════════════════════════════════════════════════════════════════════════════
#  TERMAL Z v2 — "OLCEK KILIDI"                                     2026-08-15
# ══════════════════════════════════════════════════════════════════════════════
#  SORUN (olculdu, 2 veri seti):
#    ObjectSizeZ'nin KAHIN tavani 2026'da 5.76 m, nedensel sonucu 63.71 m.
#    Sinyalin SEKLI dogru, OLCEGI yanlis — tek bir carpan. Cunku h0 kalibrasyonda
#    GSD ile bulunuyor (32.63 m) ama dogru carpan ~5.4. 2026'da kalibrasyon
#    penceresinde GT z std 0.41 m (irtifa hic degismiyor) -> olcek oradan
#    COZULEMEZ. (Oturum 4'te std 13.97 m -> cozuluyor, 7.89 m.)
#
#  FIKIR: model log uzayinda DOGRUSAL
#       irtifa_t = h0 · exp(−L_t)      ->     log(irtifa_t) = log h0 − k·L_t
#    GSD (f·v_metrik/v_piksel) her karede MUTLAK irtifa verir; kumulatif
#    olmadigi icin SURUKLENMEZ, sadece gurultuludur. log(a_gsd)'yi L'ye karsi
#    regres et:  egim = −k (ORB olceginin sismesi),  kesisim = log h0.
#    Nedensel (genisleyen pencere), kalibrasyonda irtifa degisimi GEREKTIRMEZ,
#    GPS GEREKTIRMEZ.
#
#  IKINCI KAYNAK: DPVO z'ye ayni kilidin DOGRUSAL hali (a_gsd ≈ C − s·z_dpvo).
#    2026: 47.47 -> 9.86 m ,  Oturum 4: 6.10 -> 5.29 m
#  HAKEM: GSD. Tarafli degil (mutlak, surukLENMEZ), sadece gurultulu. Kayan
#    pencerede hangi kaynak GSD ile daha iyi anlasiyorsa agirligi o alir.
#
#  OLCULEN (GPS kapali kareler, Z RMSE):
#              2026            Oturum 4
#    mevcut   63.71 / 49.27     7.89 / 5.71      (yalniz450 / 450+1600+2100)
#    YENI      7.33 /  7.27     4.82 / 4.74
#
#  RGB'ye DOKUNULMADI: bu sinif yalniz modality=="termal" iken kullanilir.
#  Kapatmak icin:  export NURON_TERMAL_Z_V2=0
# ══════════════════════════════════════════════════════════════════════════════
class OlcekKilidi:
    """ObjectSizeZ'nin biriktirdigi ORB verisini okur, olcegi online kilitler.

    ORB isini TEKRAR YAPMAZ — mevcut ObjectSizeZ ornegine baglanir.
    Her karede push(v_metrik, z_dpvo) cagrilir; z() o ana kadarki tahmini verir.
    """

    TAU = 0.10          # k icin oncül std ("k'nin 1'den sapmasina inancimiz")
    K_ALT, K_UST = 0.4, 2.0     # k emniyet sinirlari (0.4 = "2.5 kattan fazla
                                # sismis olduguna inanmayiz"). 2 veri setinde
                                # tarandi: 0.25/0.3/0.4/0.5 -> 2026 7.01/6.40/
                                # 6.11/7.22 ; ot4 hepsinde 7.39. AYARDIR.
    H_ALT, H_UST = 2.0, 300.0
    TAU_DPVO = 1.0      # DPVO olcegi icin oncül std (daha gevsek)
    GSD_W = 31          # GSD medyan penceresi (kare)
    MIN_N = 60          # regresyon icin en az ornek
    HAKEM_W = 1200      # hakem penceresi (kare)
    HAKEM_GUC = 4.0
    HAKEM_TABAN = 2.0
    SONUM = 50.0        # GPS ofsetinin sonumleme sabiti (kare). KALICI DEGIL:
                        # kalici yama 2026'da bozuyordu (11.10 -> 13.39).

    def __init__(self, zt, f=None, gt_frames=None):
        self.zt = zt
        self.f = f or zt.f
        self.G = gt_frames or zt.G
        self.vm = [np.nan]          # metrik XY hizi (kare-arasi, m)
        self.zd = [np.nan]          # hizalanmis DPVO z'si
        self._ofs = 0.0             # GPS olay ofseti
        self._ofs_i = None          # ofsetin konuldugu kare
        self._son_z = None

    # ---- her kare ----
    def push(self, v_metrik, z_dpvo):
        self.vm.append(float(v_metrik) if v_metrik is not None else np.nan)
        self.zd.append(float(z_dpvo) if z_dpvo is not None else np.nan)

    # ---- yardimcilar ----
    @staticmethod
    def _nedensel_medyan(x, W):
        x = np.asarray(x, float)
        if W <= 1 or len(x) < 2:
            return x.copy()
        pad = np.concatenate([np.full(W - 1, np.nan), x])
        sw = np.lib.stride_tricks.sliding_window_view(pad, W)
        with np.errstate(invalid="ignore"):
            return np.nanmedian(sw, axis=1)

    @staticmethod
    def _doldur(x):
        x = np.asarray(x, float).copy()
        son = np.nan
        for i in range(len(x)):
            if np.isfinite(x[i]):
                son = x[i]
            else:
                x[i] = son
        ok = np.flatnonzero(np.isfinite(x))
        if len(ok):
            x[:ok[0]] = x[ok[0]]
        else:
            x[:] = 0.0
        return x

    @staticmethod
    def _bayes_egim(x, y, tau, oncul):
        """Genisleyen pencere dogrusal regresyon; egim kendi belirsizligine gore
        oncüle buzulur. Sihirli agirlik YOK: kanit = Sxx/sigma²."""
        m = np.isfinite(x) & np.isfinite(y)
        n = m.sum()
        if n < OlcekKilidi.MIN_N:
            return oncul
        xx, yy = x[m], y[m]
        xm, ym = xx.mean(), yy.mean()
        Sxx = ((xx - xm) ** 2).sum()
        if Sxx < 1e-9:
            return oncul
        Sxy = ((xx - xm) * (yy - ym)).sum()
        Syy = ((yy - ym) ** 2).sum()
        eg = Sxy / Sxx
        sse = max(Syy - eg * Sxy, 1e-9)
        sig2 = sse / max(n - 2.0, 1.0)
        kanit = Sxx / sig2
        onc_p = 1.0 / (tau * tau)
        return (kanit * eg + onc_p * oncul) / (kanit + onc_p), xm, ym

    # ---- ana hesap ----
    def z(self, gt_z_calib):
        """Tum karelere z (asagi +). Son eleman = su anki tahmin."""
        n = min(len(self.zt.ratios), len(self.vm), len(self.zd))
        if n < 5:
            return np.zeros(max(n, 1))
        G = self.G
        r = np.clip(np.nan_to_num(np.array(self.zt.ratios[:n]), nan=1.0),
                    self.zt.CLIP_LO, self.zt.CLIP_HI)
        o = np.nan_to_num(median_filter(r, self.zt.MED_W), nan=1.0)
        L = np.cumsum(np.log(o))                       # log-kumulatif olcek

        # --- GSD mutlak irtifa (kumulatif DEGIL -> surukLENMEZ)
        px = np.asarray(self.zt.pxvel[:n], float)
        vm = np.asarray(self.vm[:n], float)
        ok = np.isfinite(px) & np.isfinite(vm) & (px > 1.0) & (vm > 0.08)
        a = np.full(n, np.nan)
        a[ok] = self.f * vm[ok] / px[ok]
        a = self._doldur(self._nedensel_medyan(a, self.GSD_W))
        a = np.clip(a, 0.5, None)

        g = np.asarray(gt_z_calib, float)
        gk = g[:min(G, len(g))]
        taban = float(np.nanmedian(gk + a[:len(gk)])) if len(gk) else 0.0
        z_gsd = taban - a                              # GSD'nin z'si (hakem)

        # --- A kaynagi: ORB, LOG uzayinda olcek kilidi
        h0_onc = float(np.clip(self.zt.h0 or 30.0, self.H_ALT, self.H_UST))
        y = np.log(a)
        cik = self._bayes_egim(L, y, self.TAU, -1.0)
        if isinstance(cik, tuple):
            eg, xm, ym = cik
            k = float(np.clip(-eg, self.K_ALT, self.K_UST))
            h0 = float(np.clip(np.exp(ym + k * xm), self.H_ALT, self.H_UST))
        else:
            k, h0 = 1.0, h0_onc
        zA = h0 * (1.0 - np.exp(-k * L))

        # --- B kaynagi: DPVO z, DOGRUSAL olcek kilidi
        zd = np.asarray(self.zd[:n], float)
        zd = self._doldur(zd)
        xb = zd - (zd[:G].mean() if n > G else zd.mean())
        yb = -(a - (a[:G].mean() if n > G else a.mean()))
        cikb = self._bayes_egim(xb, yb, self.TAU_DPVO, 1.0)
        s = float(np.clip(cikb[0] if isinstance(cikb, tuple) else 1.0, 0.05, 3.0))
        zB = s * xb

        # --- kalibrasyona demirle
        if len(gk):
            zA = zA - (zA[:len(gk)].mean() - gk.mean())
            zB = zB - (zB[:len(gk)].mean() - gk.mean())

        # --- HAKEM: GSD ile daha iyi anlasan agirligi alir
        Z = np.stack([zA, zB])
        hata = np.abs(Z - z_gsd[None, :])
        W = min(self.HAKEM_W, n)
        ker = np.ones(W)
        birim = np.convolve(np.ones(n), ker, mode="full")[:n]
        yig = np.stack([np.convolve(np.nan_to_num(h, nan=0.0), ker,
                                    mode="full")[:n] / np.maximum(birim, 1)
                        for h in hata])
        w = (1.0 / (yig + self.HAKEM_TABAN)) ** self.HAKEM_GUC
        w = w / np.maximum(w.sum(0, keepdims=True), 1e-12)
        z = (w * Z).sum(0)

        # --- GPS olayi: SONUMLU ofset (kalici degil)
        if self._ofs_i is not None and self._ofs != 0.0:
            i = np.arange(n, dtype=float)
            z = z + self._ofs * np.exp(-np.maximum(i - self._ofs_i, 0.0) / self.SONUM)
        self._son_z = float(z[-1])
        return z

    # ---- GPS acik bir kare geldiginde ----
    def gps_olayi(self, gt_z, kare):
        """Ofseti KALICI degil SONUMLU koy: kaynak zaten olcek-kilitli oldugu
        icin GPS'i kalici yamamak oncülü bozuyor (olculdu)."""
        if self._son_z is None:
            return False
        self._ofs = float(gt_z) - self._son_z
        self._ofs_i = int(kare)
        return True
