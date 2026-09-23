#!/usr/bin/env python3
"""
DPVO koşucusu — Görev 2 XY (ölçeksiz kamera trajesi).

!!! AYRI ORTAM: `dpvo` conda env (Python 3.11). Base ortam BUNU IMPORT ETMEZ.
    Orkestratör bu modülü dpvo env'inde ayrı süreç olarak çağırır.

İki kullanım:
  1) DPVOTracker  — STREAMING: feed(bgr) → o ana kadarki kamera konumu (ölçeksiz).
                    Yarışma akışı (kare kare sunucudan) için.
  2) run_video()  — BATCH: video → traj.txt (kanıtlanmış çözümü yeniden üretir, offline test).

Modaliteye göre ön-işleme:
  RGB   : ham, 0.5x küçültme.
  Termal: CLAHE + unsharp (sharpen) → domain-adaptation (3145m→32m), tam ölçek + config-tune.
"""
import os
import sys
import numpy as np
import cv2

# --- DPVO deposu (taşındı: NURON_DRONE/third_party/dpvo_repo) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
DPVO_REPO = os.path.join(_ROOT, "third_party", "dpvo_repo")
sys.path.insert(0, DPVO_REPO)

import torch
from dpvo.config import cfg
from dpvo.dpvo import DPVO
from dpvo.lietorch import SE3

DEFAULT_CFG = os.path.join(_ROOT, "models", "dpvo_default.yaml")
_SHARP = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)


# ---------------- ön-işleme ----------------
# Modaliteye özel ayarlar TEK KAYNAK: paths.GPS_PROFILE (rgb/termal ayrı).
try:
    from src.task2_position import paths as _P
    _PROFILE = _P.GPS_PROFILE
except Exception:   # dpvo env'de src erişilemezse güvenli varsayılan
    _PROFILE = {
        "rgb":    dict(scale=0.5, prep="none", cfg_tune=None),
        "termal": dict(scale=1.0, prep="sharp",
                       cfg_tune=dict(PATCHES_PER_FRAME=160, KEYFRAME_THRESH=8.0,
                                     OPTIMIZATION_WINDOW=15, REMOVAL_WINDOW=30)),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  GIRIS BOYUTU NORMALIZASYONU                                      2026-08-16
# ══════════════════════════════════════════════════════════════════════════════
#  ESKI: "scale" SABIT BIR CARPANDI (termal 1.0, rgb 0.3333). Kareler kac piksel
#  gelirse gelsin intrinsics degismiyordu. K piksel biriminde oldugu icin
#  (fx = "bir radyan kac piksel"), goruntu boyutu degisince fx/fy/cx/cy de
#  degismek ZORUNDA. Degismezse:
#     * fx yariya duserse -> her hareket 2 KAT buyuk gorunur -> yorunge sisar
#     * cx yanlis kalirsa -> projeksiyon sadece olceklenmez, EGILIR (duzeltilemez)
#     * undistort da bozulur (dist katsayilari normalize, ama K uyusmali)
#  Ornek: termal calib 640 icin fx=731.80. Kareler 1280 gelirse gercek fx 1463.6.
#
#  YENI: HEDEF GENISLIGE normalize et; DPVO'ya verilen K SABIT olsun:
#         oran = W_HEDEF / W_gelen
#         K_dpvo = K_calib * (W_HEDEF / W_CALIB)     <- W_gelen HIC GECMIYOR
#  Sira onemli: (1) GELEN boyutta undistort (K gelen boyuta olceklenmis),
#               (2) TEK oranla resize, (3) sabit K ile DPVO.
#
#  Neden dolgu (letterbox) YOK: DPVO tam evrisimli, sabit tuval istemez —
#  yalnizca 16'nin katina kirpar. Dolgu sahte kenar/kose uretir, patch
#  cikarici oralara tutunur. YOLO'da dolgu sart cunku ag sabit tensor bekler.
#  Bu yuzden TEK oranla olcekleyip yuksekligi serbest birakiyoruz.
#  En-boy orani ASLA bozulmaz (fx ve fy icin AYRI oran K'yi gecersiz kilar).
#
#  Kucultme INTER_AREA (blok ortalar, termal gurultusunu de azaltir),
#  buyutme INTER_LINEAR (bilgi uretmez ama aga alisik oldugu olcegi verir;
#  cok kucuk goruntude 16'lik izgarada patch yeri kalmaz, takip zayiflar).
#
#  W_CALIB: kalibrasyon dosyasinin cekildigi goruntu genisligi.
#  W_HEDEF: DPVO'nun gorecegi genislik. Ikisi de 640 — DPVO ~640px'te egitildi.
#  Termalde bugun gelen zaten 640 -> oran 1.0, DAVRANIS DEGISMEZ (saf sigorta).
#  RGB'de 1920 -> 640 = 0.3333, yani eski davranisin BIREBIR aynisi.
# ══════════════════════════════════════════════════════════════════════════════
W_CALIB = {"rgb": 1920.0, "termal": 640.0}
# RGB 960 SECILDI (2026-08-16). Olculdu: 4 oturum x {640, 960}, GPS yalniz ilk 450.
#   oturum          640 XY   960 XY  |  640 3B   960 3B  |  kahin2B 640 -> 960
#   2026              6.56     5.62  |    9.07     8.67  |   3.33 -> 3.24
#   2025 Oturum 2     4.62     2.83  |    7.05     5.65  |   1.63 -> 1.39
#   2025 Oturum 3     6.85     4.23  |   11.35     8.16  |   1.18 -> 1.00
#   ORTALAMA          6.01     4.22  |    9.15     7.49        (%30 / %18 kazanc)
# Kazanc GERCEK: kahin2B (trajenin ic tutarliligi) her oturumda dustu, yani
# 960 sadece hizalamayi sansli kilmiyor, YORUNGE daha dogru cikiyor.
# Z neredeyse degismiyor (ORB'dan gelir, DPVO cozunurlugunden bagimsiz).
#
# 2025 Oturum 1 HARIC: ilk ~500 kare DENIZ ustunde, doku yok, DPVO her iki
# cozunurlukte de cokuyor (287 ve 1074 m). Veri ozelligi, yontem sonucu degil.
# DIKKAT: suda 960 DAHA KOTU (287 -> 1074 m) — yuksek cozunurluk dalgalardan
# SAHTE doku uretiyor, DPVO ona tutunuyor. Su ustu ucus riski, bkz. README.
#
# 1280 denendi: XY ortalamasi daha kotu (6.48) ama RMSE/kahin daha iyi -> tek
# veride belirsiz, alinmadi.  tune (160/8/15/30) RGB'de COKUYOR: 960'ta 143.85,
# 1280'de 260.02 m. Termal ayari RGB'ye TASINMAZ.
W_HEDEF = {"rgb": float(os.environ.get("NURON_RGB_W", 960)),
           "termal": float(os.environ.get("NURON_TERMAL_W", 640))}


class Preproc:
    """Kareyi DPVO'ya vermeden önce hazırla + intrinsics'i ölçekle.
    prep MODALİTEYE GÖRE paths.GPS_PROFILE'dan gelir (rgb ≠ termal, karışmaz).
    Boyut artık SABIT CARPAN degil, HEDEF GENISLIK ile normalize edilir."""
    def __init__(self, calib, modality, hedef_w=None):
        assert modality in ("rgb", "termal")
        c = np.loadtxt(calib, delimiter=" ")
        self.fx, self.fy, self.cx, self.cy = c[:4]
        self.dist = c[4:] if len(c) > 4 else None
        self.K = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])
        prof = _PROFILE[modality]
        self.modality = modality
        self.w_calib = W_CALIB[modality]
        self.w_hedef = float(hedef_w or W_HEDEF[modality])
        self.scale = self.w_hedef / self.w_calib     # DPVO'ya giden SABIT oran
        self.prep = prof["prep"]     # rgb none · termal sharp
        self._clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self._uyari = False

    def intrinsics(self):
        """DPVO'ya verilen K. GELEN boyuttan BAGIMSIZ — hep ayni sabitler."""
        s = self.scale
        return np.array([self.fx * s, self.fy * s, self.cx * s, self.cy * s], np.float32)

    def __call__(self, image_bgr):
        img = image_bgr
        h_in, w_in = img.shape[:2]

        # (1) UNDISTORT — GELEN boyutta, K o boyuta olceklenmis olarak.
        if self.dist is not None:
            k = w_in / self.w_calib                  # calib -> gelen
            K_in = self.K * np.array([[k, 1.0, k], [1.0, k, k], [1.0, 1.0, 1.0]])
            K_in[2, 2] = 1.0
            img = cv2.undistort(img, K_in, self.dist)

        # (2) ON-ISLEME (termal: CLAHE + unsharp)
        if self.prep in ("clahe", "sharp"):
            g = self._clahe.apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            if self.prep == "sharp":
                g = cv2.filter2D(g, -1, _SHARP)
            img = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

        # (3) HEDEF GENISLIGE NORMALIZE — TEK oran, en-boy korunur, dolgu YOK
        oran = self.w_hedef / w_in
        if abs(oran - 1.0) > 1e-6:
            yeni = (int(round(w_in * oran)), int(round(h_in * oran)))
            img = cv2.resize(img, yeni,
                             interpolation=cv2.INTER_AREA if oran < 1.0
                             else cv2.INTER_LINEAR)
            if not self._uyari and abs(w_in - self.w_calib) > 1:
                print(f"[dpvo] giris {w_in}x{h_in} (calib {self.w_calib:.0f}) -> "
                      f"{yeni[0]}x{yeni[1]} normalize edildi, oran {oran:.4f}",
                      flush=True)
                self._uyari = True

        # (4) DPVO 16'nin katini ister. Sag/alttan kirpma orijini TASIMAZ,
        #     cx/cy gecerli kalir.
        h, w = img.shape[:2]
        return img[:h - h % 16, :w - w % 16]


def _build_cfg(modality, cfg_file=None):
    """DPVO config'i MODALİTEYE GÖRE kur. cfg_tune paths.GPS_PROFILE'dan (rgb None · termal tune)."""
    assert modality in ("rgb", "termal")
    cfg.merge_from_file(cfg_file or DEFAULT_CFG)
    cfg.BUFFER_SIZE = 8192
    # LOOP CLOSURE — her iki modalitede AÇIK (edges_loop, ek model gerekmez).
    # Yarışma kareleri seyrek (2250 kare = geniş baseline) → drift/kopma; loop closure toparlar.
    cfg.LOOP_CLOSURE = False   # KAPATILDI (2026-07-15): havadan duz sahnede sahte loop-closure
    # frame ~1270'te X+Y'yi teleport etti (3B hata 48.6m). LC kapali = drift artabilir ama
    # ani jump YOK -> daha kararli. Test: Ornek_Veri_1 + translation.csv.
    # NOT: patch derinlik init'i dpvo.py:427'de NURON fix (rand_like→ones_like) ile deterministik.
    tune = _PROFILE[modality].get("cfg_tune")   # rgb: None (varsayılan) · termal: düşük-doku tune
    if tune:
        for k, v in tune.items():
            setattr(cfg, k, v)
    return cfg


# ---------------- STREAMING ----------------
class DPVOTracker:
    """Kare kare besle → o ana kadarki (ölçeksiz) kamera konumu.
    Orkestratör: her sunucu karesinde feed(bgr) çağırır, dönen xyz'yi Aligner'a verir."""
    def __init__(self, modality, ckpt, calib, cfg_file=None):
        assert modality in ("rgb", "termal")
        self.modality = modality
        self.pre = Preproc(calib, modality)
        self.intr = torch.from_numpy(self.pre.intrinsics()).cuda()
        self.cfg = _build_cfg(modality, cfg_file)
        self.net = ckpt
        self.slam = None
        self.t = 0

    # ══════════ FINAL-TERMAL (2026-09-04): TEKRAR KARE + GERÇEK ZAMAN DAMGASI ══════════
    #  Kaynak termal video her ~61 s'de ~6 kare DONUYOR (ardışık kareler birebir aynı) ve sonra tek karede sıçrıyor; GPS ise sürekli.
    #  DPVO bu sıçramada tohuma bağlı olarak ölçek modunu değiştiriyor / çöküyordu (150–544 m). Çare (7/7 tohumda doğrulandı):
    #   (1) tekrar kareyi DPVO'ya VERME (tekrar_mi), (2) DPVO'yu GERÇEK kare indeksiyle çağır (tstamp → DAMPED_LINEAR modeli boşluğu ölçekler),
    #   (3) MOTION_DAMPING=1.0 (paths.GPS_PROFILE termal cfg_tune). Ayrıntı: gps/thermal_gps/work/SONUC.md §4.
    def tekrar_mi(self, image_bgr, esik):
        """Ardışık HAM gri kare (hedef genişliğe küçültülmüş, keskinleştirme ÖNCESİ) ort |fark| < esik → video tekrar karesi.
        Eşik 1.5 ham kareye göre ölçüldü (2026: tekrar 0.0–0.8, normal 11–15); keskinleştirilmiş kare gürültüyü büyütür, o yüzden ham."""
        g = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        w = int(self.pre.w_hedef)
        if g.shape[1] != w:
            g = cv2.resize(g, (w, int(round(g.shape[0] * w / g.shape[1]))), interpolation=cv2.INTER_AREA)
        g = g.astype(np.float32)
        onceki = getattr(self, "_son_gri", None); self._son_gri = g
        return bool(onceki is not None and onceki.shape == g.shape and float(np.abs(g - onceki).mean()) < esik)

    @torch.no_grad()
    def feed(self, image_bgr, tstamp=None, seed=None):
        """tstamp: GERÇEK kare indeksi (atlanan kareler dahil) — None ise iç sayaç. seed: kare başına RNG tohumu (K örnek için farklı yama seçimi)."""
        img = self.pre(image_bgr)
        ten = torch.from_numpy(img).permute(2, 0, 1).cuda()
        if self.slam is None:
            _, H, W = ten.shape
            self.slam = DPVO(self.cfg, self.net, ht=H, wd=W, viz=False)
        if seed is not None:
            torch.manual_seed(int(seed)); torch.cuda.manual_seed_all(int(seed))
        self.slam(self.t if tstamp is None else int(tstamp), ten, self.intr)
        self.t += 1
        return self.current_xyz()

    # ══════════ EK ÇIKTI (FINAL-11, 2026-09-03): yama derinliği + yer-düzlemi ══════════
    #  d_med : yeni karenin yamalarının ters-derinlik medyanının tersi (DPVO ölçeğinde "irtifa").
    #          Z kaynağı C = −s·d_med (s: Umeyama ölçeği). Birikimsiz, mutlak.
    #  n_cam : yama 3B noktalarına oturtulan yer-düzlemi normali (KAMERA çerçevesi), rms göreli.
    #          Kalibrasyonda "kamera nadir mi, arazi düz mü" kararı için (açı<15°, rms<0.08).
    #  Yamalar DPVO'da RES=4 özellik çözünürlüğünde tutulur → intrinsics/4 ile geri-yansıtma.
    @staticmethod
    def _duzlem(pts):
        """pts [m,3] → (normal birim [3], göreli rms, mesafe). 3 tur budamalı SVD."""
        if len(pts) < 12:
            return np.full(3, np.nan), np.nan
        m = np.ones(len(pts), bool); nrm = np.array([0, 0, 1.0]); rms = np.nan
        for _ in range(3):
            p = pts[m]; c = p.mean(0)
            _, _, vt = np.linalg.svd(p - c, full_matrices=False); nrm = vt[2]
            d = (pts - c) @ nrm; rms = float(np.sqrt(np.mean(d[m] ** 2)))
            m = np.abs(d) < max(2.5 * rms, 1e-6)
            if m.sum() < 12:
                break
        if nrm[2] < 0:
            nrm = -nrm
        return nrm, rms

    @torch.no_grad()
    def ek_cikti(self):
        """(d_med, n_cam[3], plan_rms) — yoksa (nan, [nan]*3, nan)."""
        bos = (np.nan, np.full(3, np.nan), np.nan)
        n = self.slam.pg.n if self.slam else 0
        if n < 1:
            return bos
        try:
            pt = self.slam.pg.patches_[n - 1].float().cpu().numpy()      # [P,3,p,p]: x, y, ters-derinlik
            px, py, di = pt[:, 0, 1, 1], pt[:, 1, 1, 1], pt[:, 2, 1, 1]
            ok = np.isfinite(di) & (di > 1e-6)
            if ok.sum() < 12:
                return bos
            fxr, fyr, cxr, cyr = self.pre.intrinsics() / 4.0
            Z = 1.0 / di[ok]; X = (px[ok] - cxr) / fxr * Z; Y = (py[ok] - cyr) / fyr * Z
            d_med = float(np.median(Z))
            nrm, rms = self._duzlem(np.column_stack([X, Y, Z]))
            return d_med, nrm, (rms / d_med if np.isfinite(rms) and d_med > 0 else np.nan)
        except Exception:
            return bos

    def current_xyz(self):
        """En son keyframe'in ölçeksiz kamera konumu (DPVO çerçevesi). Yoksa None."""
        n = self.slam.pg.n if self.slam else 0
        if n < 1:
            return None
        pose = self.slam.pg.poses_[n - 1]          # world→cam SE3
        cam = SE3(pose).inv().data.cpu().numpy()   # cam→world → konum
        return cam[:3].astype(np.float64)

    @torch.no_grad()
    def finalize(self):
        """Oturum sonu: final global BA + tüm kareler için traje (offline değerlendirme)."""
        poses, tstamps = self.slam.terminate()
        return tstamps.astype(int), poses[:, :3]


# ---------------- BATCH (offline, kanıtlanmış) ----------------
@torch.no_grad()
def run_video(video, calib, modality, ckpt, out_txt=None, cfg_file=None, log_every=500):
    """Video → traj.txt (kare tx ty tz qx qy qz qw). Kanıtlanmış çözümü yeniden üretir."""
    from multiprocessing import Process, Queue
    pre = Preproc(calib, modality)
    intr = pre.intrinsics()
    conf = _build_cfg(modality, cfg_file)

    def stream(q):
        cap = cv2.VideoCapture(video); t = 0
        while True:
            ok, im = cap.read()
            if not ok:
                break
            q.put((t, pre(im))); t += 1
        q.put((-1, None)); cap.release()

    q = Queue(maxsize=8)
    reader = Process(target=stream, args=(q,)); reader.start()
    slam = None; intr_t = torch.from_numpy(intr).cuda(); n = 0
    while True:
        t, img = q.get()
        if t < 0:
            break
        ten = torch.from_numpy(img).permute(2, 0, 1).cuda()
        if slam is None:
            _, H, W = ten.shape
            print(f"[DPVO/{modality}] {W}x{H} prep={pre.prep} scale={pre.scale}", flush=True)
            slam = DPVO(conf, ckpt, ht=H, wd=W, viz=False)
        slam(t, ten, intr_t); n += 1
        if n % log_every == 0:
            print(f"[DPVO/{modality}] {n} kare", flush=True)
    reader.join()
    poses, tstamps = slam.terminate()
    if out_txt:
        with open(out_txt, "w") as f:
            for i in range(len(tstamps)):
                f.write(f"{int(tstamps[i])} " + " ".join(f"{v:.9f}" for v in poses[i]) + "\n")
        print(f"[DPVO/{modality}] BİTTİ: {len(tstamps)} poz → {out_txt}", flush=True)
    return tstamps.astype(int), poses


if __name__ == "__main__":
    # python dpvo_runner.py <video> <calib> <rgb|termal> <ckpt> <out.txt>
    _, video, calib, modality, ckpt, out = sys.argv[:6]
    run_video(video, calib, modality, ckpt, out)
