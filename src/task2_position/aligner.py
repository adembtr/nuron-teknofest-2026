#!/usr/bin/env python3
"""
DPVO kamera-çerçevesi (ölçeksiz) trajeyi DÜNYA çerçevesine hizalar.

Eksen: GT = dünya (x=Doğu, y=Kuzey, z=AŞAĞI, SOL-el). Reçete:
    GT z'yi çevir (sol→sağ-el) → Umeyama (det=+1) → çözümün z'sini geri çevir.
İlk GT_FRAMES kare (health=1) ile kalibrasyon. GT tekrar gelirse (health=1 döner)
yeniden kalibre → drift sıfırlanır.

Saf numpy — base ortamda çalışır (DPVO worker'dan gelen pozları tüketir).
"""
import os

import numpy as np
from src.task2_position import paths as P


def umeyama(src, dst):
    """dst ≈ s·R·src + t (3B benzerlik, det=+1 zorlar). R, s, t döner."""
    ms, md = src.mean(0), dst.mean(0)
    S, D = src - ms, dst - md
    C = D.T @ S / len(src)
    U, sv, Vt = np.linalg.svd(C)
    E = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        E[2, 2] = -1
    R = U @ E @ Vt
    s = np.trace(np.diag(sv) @ E) / (S ** 2).sum() * len(src)
    t = md - s * R @ ms
    return R, s, t


def fit_recipe(pred, gt, G=None):
    """pred (kamera çerç, ölçeksiz) [N,3], gt (dünya, sol-el) [N,3] → (R, s, t).
    İlk G kareyle kalibre eder (None → hepsi). z-çevir reçetesi uygulanır."""
    G = G or len(pred)
    gr = gt.copy(); gr[:, 2] *= -1            # sol-el → sağ-el
    return umeyama(pred[:G], gr[:G])


def apply_recipe(pred, R, s, t):
    """Ölçeksiz pozları dünya çerçevesine taşı (z geri sol-el)."""
    al = (s * (R @ np.asarray(pred).reshape(-1, 3).T)).T + t
    al[:, 2] *= -1
    return al


MOVE_START_M = 3.0   # kalkistan bu kadar (m) uzaklasinca "hareket basladi" sayilir


def movement_start_idx(gt, min_move_m=MOVE_START_M):
    """DURGUN KALKIS KAPISI (saf matematik, GPU yok).
    Kalkis noktasindan (ilk GT karesi) XY uzakligi ilk kez min_move_m'yi gecen kare index'i.
    Durgun karede kameralar arasi baseline yok -> DPVO olcegi cikmaz, Umeyama'yi bozar;
    o yuzden scale kalibrasyonuna SADECE bu index'ten sonraki (hareketli) kareler girer.
    GT kalibrasyon doneminde (health=1) zaten mevcut -> ek maliyet ~0."""
    gt = np.asarray(gt, float)
    d = np.linalg.norm(gt[:, :2] - gt[0, :2], axis=1)
    idx = np.where(d > min_move_m)[0]
    return int(idx[0]) if len(idx) else 0


def fit_recipe_smart(pred, gt, G=None, min_len=200, move_thresh=0.05):
    """AKILLI kalibrasyon (2026-07-10): ilk G karenin TAMAMI yerine, drone'un gerçekten
    hareket ettiği alt-pencereyle kalibre → durgun kalkış karelerini atla.
    Termalde 3B RMSE 46→32m (test edildi). RGB'yi de bozmaz (zaten hareket erken başlar).
    Döner: (R, s, t, (w0, w1)).

    Kriter: GT hareketinin başladığı kareyi bul (durgun kalkışı geç), oradan G'ye kadar
    en düşük iç-tutarlılık (residual/yol) veren pencereyi seç."""
    G = G or len(pred)
    pred = np.asarray(pred, float); gt = np.asarray(gt, float)
    gstep = np.linalg.norm(np.diff(gt[:G, :2], axis=0), axis=1)   # GT kare-arası XY hareket
    # hareket başlangıcı: 20-karelik pencerenin %60'ı eşik üstü olan ilk yer
    w0_start = 0
    for i in range(len(gstep)):
        if gstep[i:i + 20].mean() > move_thresh and (gstep[i:i + 20] > move_thresh).mean() > 0.5:
            w0_start = i; break
    gr = gt.copy(); gr[:, 2] *= -1
    best = None
    for w0 in range(max(0, w0_start - 20), G - min_len, 25):
        for w1 in (G, G - 50):
            if w1 - w0 < min_len:
                continue
            idx = np.arange(w0, w1)
            R, s, t = umeyama(pred[idx], gr[idx])
            res = np.linalg.norm((s * (R @ pred[idx].T)).T + t - gr[idx], axis=1).mean()
            gt_yol = np.linalg.norm(gt[idx][-1, :2] - gt[idx][0, :2]) + 1e-6
            skor = res / gt_yol      # düşük = iyi (yola göre normalize residual)
            if best is None or skor < best[0]:
                best = (skor, w0, w1, R, s, t)
    if best is None:   # güvenli fallback
        R, s, t = fit_recipe(pred, gt, G)
        return R, s, t, (0, G)
    return best[3], best[4], best[5], (best[1], best[2])


# ══════════════════════════════════════════════════════════════════════════════
#  OLCEK EGILIMI DUZELTMESI (XY)                                    2026-08-16
# ══════════════════════════════════════════════════════════════════════════════
#  TESHIS (olculdu, 2 termal veri seti):
#    DPVO yerel olarak MUKEMMEL — 150 karelik pencerede kahin hatasi 0.75 m
#    (ot4: 0.32 m). Butun XY hatasi GLOBAL donme+olcek suruklenmesi.
#    Kayan pencerede optimal olcek 2026'da: kalibrasyonda 0.988, oturum
#    ortasinda 1.186, sonda 1.089. Kalibrasyonda DONAN olcek sonraya uymuyor.
#
#  "Kalibrasyon olcegi yanlis mi hesaplaniyor?" -> HAYIR. Olcek optimumu
#  KESKIN: %10 sapmanin kalibrasyon artigina maliyeti +4.49 m (taban 2.45 m).
#  O pencere icin 0.988 GERCEKTEN dogru. Sorun tahmin degil, SURUKLENME.
#
#  ANCAK: suruklenme daha kalibrasyon ICINDE basliyor ve egimi TEMIZ:
#    2026 alt-pencereler: 0.9881 0.9948 0.9982 1.0090 1.0134
#         egim +0.0866/1000 kare,  artik 0.0014,  t-ist 11.4   -> GERCEK
#    ot4  alt-pencereler: 0.9771 0.9567 1.0116 1.0173 0.9690
#         egim +0.0592/1000 kare,  artik 0.0231,  t-ist  0.5   -> GURULTU
#  Egim ILERI TASINIR, ama kor korune degil: KENDI ISTATISTIKSEL
#  ANLAMLILIGINA gore (t-istatistigi). Kanit zayifsa carpan 1.0 kalir.
#
#  OLCULEN (ort XY hatasi, GPS kapali kareler):
#                       2026            Oturum 4
#    mevcut          16.30 / 15.39      3.60 / 2.91   (yalniz450 / 450+1600+2100)
#    YENI             9.60 /  7.56      3.60 / 2.91   (ot4 kapi kapali, AYNEN)
#  5 farkli alt-pencere duzeniyle sinandi: hepsi 2026'yi iyilestirdi
#  (9.60-12.36), hepsinde ot4 kapisi kapali kaldi.
#
#  Kapatmak icin: export NURON_XY_OLCEK=0
# ══════════════════════════════════════════════════════════════════════════════
OLCEK_PENCERELERI = [(0.0, 1/3), (1/6, 1/2), (1/3, 2/3), (1/2, 5/6), (2/3, 1.0)]
OLCEK_T0, OLCEK_T1 = 4.0, 12.0     # t-ist: T0 altinda hic uygulama, T1 ustu tam
OLCEK_DOYUM = 0.25                 # |carpan-1| en fazla (emniyet)


def olcek_egilimi(pred, gt):
    """Kalibrasyon ICINDE olcek egilimi + istatistiksel anlamlilik.
    Doner: (egim_kare_basina, t_istatistigi, merkez_kare)"""
    n = len(pred)
    if n < 200:
        return 0.0, 0.0, 0.0
    ss, xs = [], []
    for f0, f1 in OLCEK_PENCERELERI:
        a, b = int(n * f0), int(n * f1)
        if b - a < 60:
            continue
        gr = gt[a:b].copy(); gr[:, 2] *= -1
        _, s_, _ = umeyama(pred[a:b], gr)
        ss.append(s_); xs.append((a + b) / 2.0)
    if len(ss) < 4:
        return 0.0, 0.0, 0.0
    gr = gt.copy(); gr[:, 2] *= -1
    _, s_ref, _ = umeyama(pred, gr)
    if not np.isfinite(s_ref) or s_ref <= 0:
        return 0.0, 0.0, 0.0
    y = np.asarray(ss) / s_ref
    x = np.asarray(xs)
    xm, ym = x.mean(), y.mean()
    Sxx = ((x - xm) ** 2).sum()
    if Sxx <= 0:
        return 0.0, 0.0, 0.0
    egim = ((x - xm) * (y - ym)).sum() / Sxx
    art = y - (egim * (x - xm) + ym)
    sig2 = (art ** 2).sum() / max(len(x) - 2, 1)
    se = np.sqrt(sig2 / Sxx) if sig2 > 0 else 0.0
    t_ist = abs(egim) / se if se > 0 else 0.0
    return float(egim), float(t_ist), float(xm)


def olcek_carpani(pred, gt, ufuk):
    """Egilimden, anlamlilikla agirliklandirilmis SABIT olcek carpani.
    ufuk: tahmin bolgesinin ORTASI (kare). Zamanla degisen surum denendi ve
    DAHA KOTU cikti (13.38 vs 9.60) — dogrusal uzatma oturum sonunda asiyor."""
    egim, t_ist, merkez = olcek_egilimi(pred, gt)
    g = float(np.clip((t_ist - OLCEK_T0) / (OLCEK_T1 - OLCEK_T0), 0.0, 1.0))
    k = 1.0 + g * egim * (ufuk - merkez)
    return float(np.clip(k, 1.0 - OLCEK_DOYUM, 1.0 + OLCEK_DOYUM)), t_ist, g


class Aligner:
    """Akış (streaming) hizalayıcı.
    - push_gt(pred, gt): kalibrasyon karesi ekle (health=1).
    - calibrate(): biriken GT çiftleriyle R,s,t çöz.
    - transform(pred): tek/çok pozu dünya çerçevesine çevir.
    GT tekrar gelirse yeni pencereyle calibrate() → drift reset.
    """
    def __init__(self, gt_frames=None, smart=True, modality=None, toplam_kare=2250):
        self.modality = modality
        self.toplam_kare = toplam_kare          # oturum uzunlugu (ufuk hesabi icin)
        self.olcek_carpani = 1.0                # uygulanan olcek duzeltmesi (rapor)
        self.olcek_t_ist = 0.0
        self.G = gt_frames or P.GT_FRAMES
        self.smart = smart   # akıllı pencere (durgun kalkışı atla) — termalde 46→32m
        self._pred = []      # kalibrasyon: ölçeksiz pozlar
        self._gt = []        # kalibrasyon: dünya GT
        self.R = self.s = self.t = None
        self.window_used = None   # akıllı seçilen (w0, w1)
        self.move_start_idx = 0   # hareket kapısının attığı durgun kare sayısı

    @property
    def ready(self):
        return self.R is not None

    def push_gt(self, pred_xyz, gt_xyz):
        self._pred.append(np.asarray(pred_xyz, float).reshape(3))
        self._gt.append(np.asarray(gt_xyz, float).reshape(3))

    def calibrate(self, window=None):
        """Biriken (pred, gt) çiftleriyle reçeteyi çöz. window: son N çift (drift reset).
        smart=True (varsayılan) → akıllı pencere (durgun kalkışı atla)."""
        if len(self._pred) < 3:
            return False
        pred = np.stack(self._pred); gt = np.stack(self._gt)
        if window:
            pred, gt = pred[-window:], gt[-window:]
        else:
            # HAREKET KAPISI: kalkistan >MOVE_START_M uzaklasana kadarki durgun kareleri
            # scale'e KATMA (baseline yok -> scale cikmaz). health=0'a kadarki hareketli
            # kareler kalir. Yeterli hareketli kare kalirsa uygula, yoksa hepsini kullan.
            s0 = movement_start_idx(gt, MOVE_START_M)
            if len(gt) - s0 >= 100:
                pred, gt = pred[s0:], gt[s0:]
            self.move_start_idx = s0
        if self.smart and len(pred) >= 250 and not window:
            self.R, self.s, self.t, self.window_used = fit_recipe_smart(pred, gt, G=len(pred))
        else:
            self.R, self.s, self.t = fit_recipe(pred, gt, G=len(pred))

        # --- OLCEK EGILIMI DUZELTMESI (bkz. dosya basindaki blok) ---
        # Yalniz TERMAL; RGB'de olculmedi. Kapatma: NURON_XY_OLCEK=0
        if (self.modality == "termal" and not window and
                os.environ.get("NURON_XY_OLCEK", "1").strip() != "0"):
            try:
                G_ = self.G or len(pred)
                ufuk = G_ + (self.toplam_kare - G_) / 2.0      # tahmin bolgesinin ORTASI
                k, t_ist, g = olcek_carpani(pred, gt, ufuk)
                self.olcek_t_ist, self.olcek_carpani = t_ist, k
                if abs(k - 1.0) > 1e-6:
                    # Carpan, KALIBRASYONUN SON karesi etrafinda uygulanir ->
                    # kare 450'de sicrama OLMAZ, sapma ileriye dogru buyur.
                    p_ref = pred[-1]
                    self.t = self.t + (self.s - self.s * k) * (self.R @ p_ref)
                    self.s = self.s * k
                print(f"[GPS] olcek egilimi: t-ist {t_ist:.1f} -> guven {g:.2f} "
                      f"-> carpan {k:.4f}"
                      + ("" if abs(k - 1.0) > 1e-6 else "   (kanit zayif, DOKUNULMADI)"),
                      flush=True)
            except Exception as e:
                print(f"[GPS] olcek egilimi hesaplanamadi ({e}) -> carpan 1.0", flush=True)
        return True

    def transform(self, pred_xyz):
        """Ölçeksiz poz(lar) → dünya çerçevesi [.,3]. Kalibre değilse None."""
        if not self.ready:
            return None
        return apply_recipe(pred_xyz, self.R, self.s, self.t)


# --- Offline doğrulama (kaynak sonuçlarını yeniden üret) ---
def _selftest(traj_path, gt_csv, G=None):
    import pandas as pd
    G = G or P.GT_FRAMES
    t = np.loadtxt(traj_path); kare = t[:, 0].astype(int); pred = t[:, 1:4]
    gt_all = pd.read_csv(gt_csv)[['translation_x', 'translation_y', 'translation_z']].values
    kare = kare[kare < len(gt_all)]; pred = pred[:len(kare)]; gt = gt_all[kare]
    R, s, tt = fit_recipe(pred, gt, G)
    al = apply_recipe(pred, R, s, tt)
    exy = np.linalg.norm(al[G:, :2] - gt[G:, :2], axis=1).mean()
    print(f"  XY tahmin={exy:.2f}m  (Doğu {np.abs(al[G:,0]-gt[G:,0]).mean():.2f} / "
          f"Kuzey {np.abs(al[G:,1]-gt[G:,1]).mean():.2f})  kalib={np.linalg.norm(al[:G,:2]-gt[:G,:2],axis=1).mean():.2f}")
    return kare, al, gt


if __name__ == "__main__":
    import os
    d = os.path.join(P.ROOT if hasattr(P, "ROOT") else ".", "")
    from src.common.config import ROOT
    g = os.path.join(ROOT, "offline_data", "gps")
    print("[RGB] v1:");    _selftest(f"{g}/v1_rgb_traj.txt", f"{g}/v1_gt.csv")
    print("[TERMAL] v2:"); _selftest(f"{g}/termal_cfg_sharp_traj.txt", f"{g}/v2_gt.csv")
