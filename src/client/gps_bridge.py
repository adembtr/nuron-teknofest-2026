#!/usr/bin/env python3
"""
GPS köprüsü (Görev 2) — base env tarafı.

DPVO ayrı `dpvo` conda env'inde derlenmiş CUDA uzantısıyla çalışır; orkestratör base
env'dedir. Bu yüzden GPS iki parçalı:
  * scripts/gps_worker.py  (dpvo env)  → DPVOTracker.feed → current_xyz → dosyaya yazar.
  * bu köprü (base env)                → kareyi worker'a verir, dpvo_xyz'i okur,
                                          PositionEstimator ile kalibrasyon/kestirim yapar.

DOSYA-TABANLI IPC (runtime/gps/):
  in/{seq}.jpg   ← orkestratör kareyi yazar (seq artan)
  out/{seq}.txt  → worker "x y z" yazar
Orkestratör out/{seq}.txt belirene kadar bekler (kare başı süre limiti YOK).

RGB (2026-09-03, FINAL-11): PositionEstimator YERİNE src/task2_position/final_kestirici.FinalKestirici.
  Worker artık 8 sütun yazar: x y z d_med nx ny nz rms (yama-derinliği + yer-düzlemi normali).
  health=1 (ilk 450): FinalKestirici kalibrasyon biriktirir, GT aynen döner. İlk health=0: Umeyama + geometri
  (nadir/oblik) + kaynak seviyeleri; sonra XY-EKF + 3 kaynaklı Z füzyonu. Orta-oturum GPS = EKF/Kalman ölçümü.
  Eski RGB çözümü (reanchor, Z v2, XY kendini-doğrulayan düzeltme) KALDIRILDI (arşiv 2026-09-10'da silindi).
  3 oturum ort. hata: eski 1.93/2.09/3.29 → yeni 1.21/0.74/1.33 m (X/Y/Z).

KALİBRASYON (TERMAL — PositionEstimator, DEĞİŞTİRİLMEDİ):
  health=1 (ilk ~450 kare): update_calib(dpvo_xyz, gt_xyz) + GT'yi AYNEN geri gönder (bedava puan).
  kalibrasyon penceresi biter: finalize_calib().
  health=0: estimate(dpvo_xyz) → NED dünya (x=Doğu, y=Kuzey, z=Aşağı).

Worker yoksa/yanıt vermezse (timeout) → GRACEFUL: health=1'de GT echo, health=0'da son
bilinen konumu tut (drift yerine sabit) ve durumu enabled=False yapar.

İNTERNET KESİNTİSİ DAYANIKLILIĞI (KESİNTİ-RESUME):
  * Köprü durumu (seq, kalibrasyon/PositionEstimator, son konum, reanchor) HER kare
    diske pickle'lanır (runtime/gps/bridge_state.pkl).
  * Konsol 2 (arayüz) internet kopunca kapanır; Konsol 1 (worker) AÇIK KALIR ve bekler.
  * Konsol 2 yeniden başlayınca köprü: state TAZE (<10 dk) + worker canlı ise →
    seq'i worker'ın kaldığı yerin BİR SONRASINA ayarlar, kalibrasyonu yükler →
    sanki hiç kesinti olmamış, 0.7 sn beklenmiş gibi devam eder. seq'i 0'dan
    BAŞLATMAZ (bayat out/*.txt okumaz). Yeni oturumda (state bayat) sıfırdan başlar.
  * AYNI kare tekrar gelirse (reconnect'te sunucu son kareyi yollarsa) DPVO'ya TEKRAR
    VERİLMEZ (traje bozulmaz) — hareket yok, son konum döner.
"""
import os
import time
import pickle
from typing import Optional, Tuple
import numpy as np
import cv2

from src.common.config import ROOT, CFG
from src.common.runtime import RUNTIME
from src.common import kosu as KOSU
from src.task2_position.position import PositionEstimator            # TERMAL yolu
from src.task2_position.final_kestirici import FinalKestirici        # RGB yolu (FINAL-11, 2026-09-03)
from src.task2_position.z_termal import ObjectSizeZ, OlcekKilidi     # RGB Z kaynagi A (ORB olcek-orani); TERMAL A kaynagi = OlcekKilidi
from src.task2_position.termal_kestirici import TermalKestirici, SaglamEnsemble, kalibre as termal_kalibre, yukle_prm as termal_prm   # FINAL-TERMAL (2026-09-04)
from src.task2_position.aligner import apply_recipe as _apply_recipe

# 2026-09-17 — KOSU KLASORU: GPS dosya-IPC'si de kosunun kendi klasorunde
#   kosular/folder<N>/runtime/gps/{in,out,STOP,bridge_state.pkl}
# Worker AYRI surec oldugu icin bu yolu SABIT bir isaret dosyasindan ogrenir
# (runtime/AKTIF_KOSU). GPU kuyrugu sinyalleri (GPU_ISTEK/GPU_ONAY) SABIT
# runtime/gps/ altinda kalir — icerikleri yok, arsivlenecek bir sey degil.
GPS_DIR = os.path.join(KOSU.kosu_dir(), "runtime", "gps")
IN_DIR = os.path.join(GPS_DIR, "in")
OUT_DIR = os.path.join(GPS_DIR, "out")
STATE_PKL = os.path.join(GPS_DIR, "bridge_state.pkl")
os.makedirs(IN_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
KOSU.isaret_yaz(GPS_DIR)              # worker bu dosyadan bizi bulur
print(f"[GPS] kosu klasoru: {GPS_DIR}", flush=True)
RESUME_FRESH_SEC = 600.0   # state bu kadar tazeyse KESİNTİ-RESUME; değilse yeni oturum



class TermalKopru:
    """FINAL-TERMAL (2026-09-04) — termal GPS kestirim çekirdeği (worker'dan bağımsız, offline testte de aynen kullanılır).
    Girdi (her kare): vec = worker satırı [tekrar_bayragi, K×(x y z d_med nx ny nz rms)] (veya 8/3 sütun tek örnek), kare (ORB için), gt (health=1 ise), health.
    Akış: kalibrasyon (ilk G, health=1): üye başına ham DPVO + GT birikir → ilk health=0'da üye başına Umeyama (akıllı pencere) →
          sanal traje (üyelerin hizalı medyanı, 1/50 ölçek) ile TermalKestirici kalibre edilir → sonra her karede üye artımlarının sağlam medyanı
          (SaglamEnsemble) → kestirici (XY-EKF + A/B/C Z füzyonu + koruma). A kaynağı: ObjectSizeZ (640 px) + OlcekKilidi (GPS ofsetsiz).
    Tekrar karede (video donması) DPVO/ORB beslenmez; kestirici ölü hesapla çıktı verir. Saf numpy → pickle'lanabilir."""
    def __init__(self, gt_frames=450):
        self.G = gt_frames
        self.prm, self.cfg = termal_prm()
        ens = self.cfg.get("ensemble", {}); self.OLC = float(ens.get("sanal_olcek", 50.0))
        self.zt = ObjectSizeZ(gt_frames=gt_frames); self.zt_w = 640
        self.kilit = OlcekKilidi(self.zt)
        self.est = TermalKestirici(G=gt_frames, prm=self.prm)
        self.ens = None; self.recete = None; self.K = None
        self._ham = None; self._gt = []; self._dm_kal = []; self._ncam_kal = []; self._rms_kal = []
        self._vm = [np.nan]; self._zd = [np.nan]; self.konum = None; self.k = 0; self.n_tekrar = 0
        self.bilgi = {}

    # ---- yardımcılar ----
    def _zt_img(self, img):
        h, w = img.shape[:2]
        return img if w == self.zt_w else cv2.resize(img, (self.zt_w, int(round(h * self.zt_w / w))), interpolation=cv2.INTER_AREA if w > self.zt_w else cv2.INTER_LINEAR)

    @staticmethod
    def _parse(vec):
        """→ (tekrar, bloklar[K][8])"""
        v = np.asarray(vec, np.float64).reshape(-1)
        if v.size >= 9 and (v.size - 1) % 8 == 0:
            K = (v.size - 1) // 8
            return bool(v[0] > 0.5), [v[1 + 8 * k: 9 + 8 * k] for k in range(K)]
        if v.size == 8:
            return False, [v]
        b = np.full(8, np.nan); b[:3] = v[:3]; return False, [b]

    def _zA(self, i):
        """ORB ölçek kilidi z'si (kalibrasyon sonrası; GPS ofseti YOK — olaylar kestiricideki Kalman'la işlenir)."""
        try:
            if len(self.kilit.vm) < 5 or not self._gt:
                return None
            gz_kal = np.stack(self._gt)[:, 2]
            return float(self.kilit.z(gz_kal)[-1])
        except Exception:
            return None

    # ---- ana adım ----
    def adim(self, vec, frame_bgr, gt_xyz, health, frame_idx):
        i = self.k; self.k += 1
        tekrar, bloklar = self._parse(vec) if vec is not None else (False, None)
        gt = None if gt_xyz is None else np.asarray(gt_xyz, float)
        # ── kalibrasyon (ilk G kare, health=1) ──
        if self.recete is None and health == 1 and gt is not None:
            if bloklar is not None:
                if self._ham is None: self.K = len(bloklar); self._ham = [[] for _ in range(self.K)]
                for k in range(self.K): self._ham[k].append(np.nan_to_num(bloklar[k][:3], nan=0.0))
                dm = [self._ham_dm(b) for b in bloklar]; self._dm_kal.append(float(np.nanmedian(dm)) if np.isfinite(dm).any() else np.nan)
                self._ncam_kal.append(bloklar[0][4:7] if np.all(np.isfinite(bloklar[0][4:7])) else None); self._rms_kal.append(float(bloklar[0][7]))
            else:
                if self._ham is not None:
                    for k in range(self.K): self._ham[k].append(self._ham[k][-1] if self._ham[k] else np.zeros(3))
                self._dm_kal.append(np.nan); self._ncam_kal.append(None); self._rms_kal.append(np.nan)
            self._gt.append(gt)
            if frame_bgr is not None and not tekrar:
                self.zt.feed(self._zt_img(frame_bgr))
            else:
                self.zt.ratios.append(1.0); self.zt.pxvel.append(np.nan)
            vmet = float(np.linalg.norm(self._gt[-1][:2] - self._gt[-2][:2])) if len(self._gt) >= 2 else np.nan
            self.kilit.push(vmet, np.nan)
            self.konum = gt.copy()
            return tuple(float(v) for v in gt)
        # ── ilk health=0: kalibrasyonu bitir ──
        if self.recete is None:
            if self._ham is None or len(self._gt) < 50 or self.K is None:
                return None if self.konum is None else tuple(float(v) for v in self.konum)
            self._kalib_bitir()
        # ── tekrar kare: DPVO/ORB yok, kestirici ölü hesap ──
        if tekrar or bloklar is None:
            self.n_tekrar += 1
            for k in range(self.K): self._ham[k].append(self._ham[k][-1])
            self.zt.ratios.append(1.0); self.zt.pxvel.append(np.nan); self.kilit.push(0.0, self._zd[-1])
            out = self.est.adim(None, gt_xyz=gt, health=health, kare_tekrar=True)
            if health == 1 and gt is not None: self.konum = gt.copy()
            if out is not None: self.son_cikti = np.asarray(out, float)
            return None if out is None else tuple(float(v) for v in out)
        # ── normal kare: üye artımları → sağlam ensemble → kestirici ──
        for k in range(self.K): self._ham[k].append(np.nan_to_num(bloklar[k][:3], nan=0.0))
        AL = np.array([_apply_recipe(np.array(self._ham[k][-2:]), *self.recete[k][:3]) for k in range(self.K)])   # [K,2,3]
        art = AL[:, 1] - AL[:, 0]
        dm_met = [self.recete[k][1] * self._ham_dm(bloklar[k]) if np.isfinite(self._ham_dm(bloklar[k])) else np.nan for k in range(self.K)]
        d_ens, dm_ens = self.ens.birlestir(art, dm_met)
        self.konum = self.konum + d_ens
        for k in range(self.K): self._ham[k] = self._ham[k][-2:]                     # bellek: yalnız son iki ham poz
        if frame_bgr is not None:
            self.zt.feed(self._zt_img(frame_bgr))
        else:
            self.zt.ratios.append(1.0); self.zt.pxvel.append(np.nan)
        self._vm.append(float(np.linalg.norm(d_ens[:2]))); self._zd.append(float(self.konum[2])); self.kilit.push(self._vm[-1], self._zd[-1])
        zA = self._zA(i)
        san = self.konum / self.OLC; san = np.array([san[0], san[1], -san[2]])
        r_orb = self.zt.ratios[-1]; px_orb = self.zt.pxvel[-1]
        out = self.est.adim(san, d_med=(dm_ens / self.OLC if dm_ens is not None and np.isfinite(dm_ens) else None), n_cam=None, plan_rms=None,
                            zA_ham=zA, gt_xyz=gt if health == 1 else None, health=health, r_orb=r_orb, px_orb=px_orb)
        if health == 1 and gt is not None: self.konum = gt.copy() * 0 + self.konum   # ensemble konumu kendi yolunda kalır (kestirici GT'yi ölçüm olarak işledi)
        if out is not None: self.son_cikti = np.asarray(out, float)
        return None if out is None else tuple(float(v) for v in out)

    @staticmethod
    def _ham_dm(b):
        return float(b[3]) if np.isfinite(b[3]) else np.nan

    def _kalib_bitir(self):
        G = len(self._gt); gt = np.stack(self._gt)
        self.recete = [termal_kalibre(np.array(self._ham[k]), gt) for k in range(self.K)]        # (R, s, t, pencere, artik, yol)
        AL0 = np.array([_apply_recipe(np.array(self._ham[k]), *self.recete[k][:3]) for k in range(self.K)])   # [K,G,3] dünya (z aşağı)
        san = np.median(AL0, axis=0); san_d = san / self.OLC; san_d[:, 2] *= -1
        s_med = float(np.median([rc[1] for rc in self.recete]))
        ens = self.cfg.get("ensemble", {}); self.ens = SaglamEnsemble(self.K, esik=float(ens.get("esik", 0.5)), ema=float(ens.get("ema", 0.02)))
        self.est = TermalKestirici(G=self.G, prm=self.prm)
        for j in range(G):
            dm = self._dm_kal[j] * s_med / self.OLC if np.isfinite(self._dm_kal[j]) else None
            self.est.adim(san_d[j], d_med=dm, n_cam=self._ncam_kal[j], plan_rms=self._rms_kal[j], zA_ham=None, gt_xyz=gt[j], health=1)
        self.konum = san[-1].copy()
        # ORB kilidi h0 + kalibrasyon hız/z geçmişi (üretim mantığı)
        try:
            self.zt.calibrate_h0(gt[:, :2])
        except Exception:
            pass
        al = san
        m = min(len(self.kilit.zd) - 1, len(al))
        if m > 0: self.kilit.zd[len(self.kilit.zd) - m:] = [float(x) for x in al[-m:, 2]]
        self._zd = list(self.kilit.zd); self._vm = list(self.kilit.vm)
        self.bilgi = dict(K=self.K, s=[round(float(rc[1]), 2) for rc in self.recete], artik=[round(float(rc[4]), 2) for rc in self.recete],
                          yol=round(float(self.recete[0][5]), 1))
        print(f"[GPS] FINAL-TERMAL kalibrasyon: K={self.K} s={self.bilgi['s']} artik={self.bilgi['artik']} yol={self.bilgi['yol']} m", flush=True)

    def kestirici_bilgi(self):
        e = self.est
        if not getattr(e, "kalibre", False): return {}
        return dict(nadir=e.nadir, aci=round(e.aci, 1), rms=round(e.rms, 3), w=np.round(e.w, 2).tolist(), saglik=e.saglik, tutucu=e.tutucu, uyari=e.uyarilar)


class GpsBridge:
    def __init__(self, modality: str, gt_frames: Optional[int] = None,
                 wait_pose_sec: float = 5.0, enable_worker: bool = True):
        self.modality = modality
        self.gt_frames = gt_frames or CFG["session"].get("gt_frames", 450)
        self.wait_pose_sec = wait_pose_sec
        self.enable_worker = enable_worker
        if modality == "rgb":
            # ═══ RGB: FINAL-11 (2026-09-03) ═══ XY-EKF + 3 kaynakli Z; ORB olcek-orani 960 px karede
            # (focal 1389.7*960/1920 = 694.85). Ayarlar config/gps_rgb_final.json.
            self.est = None
            self.fk = FinalKestirici(G=self.gt_frames)
            self.zf = ObjectSizeZ(f=694.85, gt_frames=self.gt_frames)
            self.zf.CLIP_LO, self.zf.CLIP_HI = 0.98, 1.02      # RGB icin olculen en iyi ORB ayarlari
            self.zf.MED_W = 3; self.zf.NEDENSEL = True
            self.zf.OFSET, self.zf.OFSET_N = "son", 10
            self.rgb_z_width = 960
            self._gt_kal = []                                  # kalibrasyon GT'leri (ORB h0 + zA seviyesi)
            self._zA_hazir = False
        else:
            # ═══ TERMAL: FINAL-TERMAL (2026-09-04) — TermalKopru (K DPVO ensemble + TermalKestirici). Eski PositionEstimator yolu kaldırıldı (arşiv 2026-09-10'da silindi)
            self.est = None; self.fk = self.zf = None
            self.tk = TermalKopru(gt_frames=self.gt_frames)
        self.seq = 0
        self._calibrated = False
        self._last_world = np.zeros(3, np.float32)     # son gönderilen konum (fallback)
        self._reanchor = np.zeros(3, np.float32)       # orta-oturum GT gelince drift-sıfırlama ofseti
        # ═══ XY: KENDINI DOGRULAYAN HIZALAMA YENILEME (2026-08-15) ═══
        # Olculdu: DPVO yerel olarak mukemmel (150 karede 0.75 m); butun XY hatasi
        # GLOBAL donme+olcek suruklenmesi (kalibrasyonda olcek 0.99, oturum ortasinda
        # 1.19). Kalibrasyonda donan (R,s) oturumun geri kalanina uymuyor.
        # Duzeltme GPS olayindan cikarilabilir AMA kanit azken ZARARLI:
        #   olay sayisi:      0      1      2      4      8     16   (en kotu veri)
        #   ofset (mevcut) 16.30  15.87  15.48  17.16   6.83   3.81
        #   kapisiz d&o    16.30  15.87  15.48  16.82   4.25   2.08   <- 4 olayda zarar
        #   KENDINI DOGR.  16.30  15.87  15.48  15.55   5.35   2.45   <- hicbir yerde zarar yok
        # Bu yuzden duzeltme once ADAY olarak tutulur, BIR SONRAKI GPS olayinda
        # "hatayi gercekten azaltir miydi?" diye SINANIR, guven ancak ispatlandikca
        # artar. GPS az gelirse guven 0 kalir -> davranis birebir eskisi gibi.
        self._xy_duz_acik = (modality == "termal" and
                             os.environ.get("NURON_XY_DUZELT", "1").strip() != "0")
        self._xyR = np.eye(2, dtype=np.float64)   # uygulanan donme
        self._xys = 1.0                           # uygulanan olcek
        self._xy_gt0 = None                       # son GPS olayindaki GERCEK xy (donme merkezi)
        self._xy_aday = None                      # (aci, olcek) — henuz ispatlanmamis
        self._xy_guven = 0.0                      # [0,1]
        self._xy_olay_kare = -10**9
        self.XY_GUVEN_KAZANC = 0.6
        self.XY_EN_AZ_YOL = 20.0                  # m — kisa yolda aci belirsiz
        self.XY_EN_AZ_KARE = 100
        # KAC OLAYDAN SONRA UYGULANSIN? Olculdu (en kotu veri, ort XY m):
        #   GPS olayi:       2      3      4      5      6      8     12     16
        #   mevcut       15.48  21.17  17.16   9.28   8.94   6.83   4.67   3.81
        #   esik 3       15.48  21.17  15.60   7.02   7.89   5.28   3.18   2.59  <-
        #   esik 4       15.48  21.17  17.17   9.03   8.52   4.76   3.16   2.64
        # esik 3: 3 olaya kadar URETIMLE BIREBIR AYNI, 4'ten itibaren tutarli
        # kazanc. esik 4 bir olay gec kalip 4-6 araligini kaciriyor.
        # NOT: ot4'te 12+ olayda kucuk bozulma var (0.77 -> 1.02 m) — o traje
        # zaten metre alti, duzeltme gurultu ekliyor. Yarismada o kadar GPS yok.
        self.XY_EN_AZ_OLAY = 3
        self._xy_olay_sayisi = 0
        self._served = False                           # worker en az bir kez yanıt verdi mi
        self._last_key = None                          # son işlenen kare kimliği (duplicate tespiti)
        # İLK karede worker DPVO'yu yüklüyor olabilir (uzun sürer) → ilk yanıta kadar
        # geniş tolerans; worker yanıt verdikten sonra kare başı kısa timeout (wait_pose_sec).
        self.first_wait_sec = 180.0
        for d in (IN_DIR, OUT_DIR):
            os.makedirs(d, exist_ok=True)
        self._degraded = not enable_worker             # worker kapalıysa baştan degraded

        # KESİNTİ-RESUME: taze state + canlı worker varsa kaldığı yerden devam, yoksa sıfırdan.
        if enable_worker and self._try_resume():
            pass
        else:
            self._fresh_start()

    # ---- KESİNTİ-RESUME yardımcıları ----
    @staticmethod
    def _out_max_seq() -> Optional[int]:
        """OUT_DIR'deki en yüksek seq (worker'ın en son yanıtladığı kare)."""
        seqs = []
        try:
            for f in os.listdir(OUT_DIR):
                if f.endswith(".txt"):
                    try:
                        seqs.append(int(os.path.splitext(f)[0]))
                    except ValueError:
                        pass
        except FileNotFoundError:
            return None
        return max(seqs) if seqs else None

    def _try_resume(self) -> bool:
        """Taze state + canlı worker (out dosyaları) varsa durumu yükle, seq'i hizala."""
        if not os.path.exists(STATE_PKL):
            return False
        if time.time() - os.path.getmtime(STATE_PKL) > RESUME_FRESH_SEC:
            return False                               # bayat → yeni oturum
        mx = self._out_max_seq()
        if mx is None:
            return False                               # worker hiç yanıt vermemiş
        try:
            with open(STATE_PKL, "rb") as f:
                st = pickle.load(f)
            self.est = st["est"]
            if self.modality == "termal":
                if st.get("tk") is None:
                    return False                           # eski surum state'i -> sifirdan
                self.tk = st["tk"]
            if self.modality == "rgb":
                if st.get("fk") is None:
                    return False                           # eski surum state'i -> sifirdan
                self.fk = st["fk"]; self.zf = st["zf"]
                self._gt_kal = st.get("gt_kal") or []; self._zA_hazir = bool(st.get("zA_hazir", False))
            self._calibrated = bool(st["calibrated"])
            self._last_world = np.asarray(st["last_world"], np.float32)
            self._reanchor = np.asarray(st["reanchor"], np.float32)
            self._last_key = st.get("last_key")
            self._xyR = st.get("xyR", np.eye(2))
            self._xys = st.get("xys", 1.0)
            self._xy_gt0 = st.get("xy_gt0")
            self._xy_aday = st.get("xy_aday")
            self._xy_guven = st.get("xy_guven", 0.0)
            self._xy_olay_kare = st.get("xy_olay_kare", -10**9)
            self._xy_olay_sayisi = st.get("xy_olay_sayisi", 0)
            self.seq = mx + 1                          # worker'ın kaldığı yerin BİR sonrası
            self._served = True                        # worker zaten çalışıyor
            print(f"[GPS] KESINTI-RESUME: seq={self.seq}'den devam "
                  f"(kalibre={self._calibrated}, worker canli) — kesinti yokmus gibi.", flush=True)
            return True
        except Exception as e:
            print(f"[GPS] resume basarisiz ({e}) -> sifirdan.", flush=True)
            return False

    def _fresh_start(self):
        """Yeni oturum: bayat in/out/state temizle (bayat dosya okumayı önle)."""
        for d in (IN_DIR, OUT_DIR):
            try:
                for f in os.listdir(d):
                    os.remove(os.path.join(d, f))
            except FileNotFoundError:
                pass
        try:
            if os.path.exists(STATE_PKL):
                os.remove(STATE_PKL)
        except OSError:
            pass

    def _save_state(self):
        """Köprü durumunu atomik olarak diske yaz (her kare) → kesinti-resume için."""
        try:
            tmp = STATE_PKL + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump({"est": self.est, "fk": self.fk, "zf": self.zf, "tk": getattr(self, "tk", None),
                             "gt_kal": getattr(self, "_gt_kal", None), "zA_hazir": getattr(self, "_zA_hazir", False),
                             "calibrated": self._calibrated,
                             "last_world": np.asarray(self._last_world),
                             "reanchor": np.asarray(self._reanchor),
                             "seq": self.seq, "last_key": self._last_key,
                             "xyR": self._xyR, "xys": self._xys,
                             "xy_gt0": self._xy_gt0, "xy_aday": self._xy_aday,
                             "xy_guven": self._xy_guven,
                             "xy_olay_kare": self._xy_olay_kare,
                             "xy_olay_sayisi": self._xy_olay_sayisi}, f)
            os.replace(tmp, STATE_PKL)
        except Exception:
            pass

    # ---- XY kendini dogrulayan duzeltme ----
    @staticmethod
    def _R2(a):
        c, s_ = np.cos(a), np.sin(a)
        return np.array([[c, -s_], [s_, c]])

    def _xy_duzelt(self, p3):
        """Son GPS demiri etrafinda donme+olcek uygula. Guven 0 iken KIMLIK."""
        if (not self._xy_duz_acik) or self._xy_gt0 is None or self._xy_guven <= 0:
            return p3
        p3 = np.asarray(p3, np.float64).copy()
        d = p3[:2] - self._xy_gt0
        p3[:2] = self._xy_gt0 + self._xys * (self._xyR @ d)
        return p3

    def _xy_olay(self, tah_xy, gt_xy, frame_idx):
        """GPS olayi: once ADAYI SINA, sonra yeni aday hesapla."""
        if not self._xy_duz_acik:
            return
        gt_xy = np.asarray(gt_xy, np.float64)
        if self._xy_gt0 is None or frame_idx - self._xy_olay_kare < self.XY_EN_AZ_KARE:
            self._xy_gt0 = gt_xy.copy()
            self._xy_olay_kare = frame_idx
            return
        d = np.asarray(tah_xy, np.float64) - self._xy_gt0
        hedef = gt_xy - self._xy_gt0
        ny, nh = float(np.linalg.norm(d)), float(np.linalg.norm(hedef))
        if ny > self.XY_EN_AZ_YOL and nh > self.XY_EN_AZ_YOL:
            self._xy_olay_sayisi += 1
            if self._xy_aday is not None:
                aci_a, olc_a = self._xy_aday
                alt = self._xy_gt0 + olc_a * (self._R2(aci_a) @ d)
                h_duz = float(np.linalg.norm(alt - gt_xy))
                h_ham = float(np.linalg.norm(np.asarray(tah_xy) - gt_xy))
                kazanc = (h_ham - h_duz) / max(h_ham, 1e-6)
                self._xy_guven = float(np.clip(
                    self._xy_guven + self.XY_GUVEN_KAZANC * kazanc, 0.0, 1.0))
                uygula = (self._xy_guven > 0 and
                          self._xy_olay_sayisi >= self.XY_EN_AZ_OLAY)
                if uygula:
                    self._xyR = self._R2(self._xy_guven * aci_a) @ self._xyR
                    self._xys = float(np.clip(
                        self._xys * (1 + self._xy_guven * (olc_a - 1)), 0.5, 2.0))
                print(f"[GPS] XY duzeltme sinandi (kare {frame_idx}, olay "
                      f"#{self._xy_olay_sayisi}): kazanc {kazanc:+.2f} -> guven "
                      f"{self._xy_guven:.2f} -> "
                      f"{'UYGULANDI' if uygula else 'bekliyor (esik %d)' % self.XY_EN_AZ_OLAY}",
                      flush=True)
            aci = np.arctan2(hedef[1], hedef[0]) - np.arctan2(d[1], d[0])
            self._xy_aday = (float((aci + np.pi) % (2*np.pi) - np.pi), nh / ny)
        self._xy_gt0 = gt_xy.copy()
        self._xy_olay_kare = frame_idx

    # ---- worker IPC ----
    def _dpvo_xyz(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Kareyi worker'a ver, worker ciktisini oku: 3 sutun (x y z) ya da 8 sutun
        (x y z d_med nx ny nz rms — FINAL-11). Worker yoksa None (degraded)."""
        if self._degraded:
            return None
        s = self.seq
        inp = os.path.join(IN_DIR, f"{s:06d}.jpg")
        outp = os.path.join(OUT_DIR, f"{s:06d}.txt")
        # ATOMIK YAZIM (2026-08-16) — KRITIK HATA DUZELTMESI
        # ESKI: cv2.imwrite(inp, frame_bgr) dogrudan hedef ada yaziyordu.
        # gps_worker.py:82 dosyayi "var olur olmaz" acar; yazim bitmeden okur.
        # YARIM JPEG cogu zaman HATA VERMEZ, "Premature end of JPEG file" uyarisi
        # basip KIRPILMIS bir goruntu dondurur -> worker'daki imread-None yeniden
        # deneme dongusu HIC tetiklenmez, bozuk kare sessizce DPVO'ya gider.
        # OLCULDU (termal uctan uca, 2250 kare): 187-228 kare bozuk (~%9).
        # Sonuc: 3B RMSE 322 m (beklenen ~12 m); kalibrasyon penceresi de bozuldugu
        # icin olcek egilimi t-ist 15.3 yerine 0.4 olculdu.
        # Worker CIKTIYI zaten os.replace ile atomik yaziyordu (gps_worker.py:104);
        # eksik olan GIRDI tarafiydi.
        # imwrite kodlayiciyi UZANTIDAN secer -> ".tmp" ile calismaz. Bu yuzden
        # once bellekte JPEG'e kodla, ham baytlari gecici dosyaya yaz, sonra tasi.
        ok_enc, buf = cv2.imencode(".jpg", frame_bgr)
        tmp_in = inp + ".tmp"
        if ok_enc:
            with open(tmp_in, "wb") as _fh:
                _fh.write(buf.tobytes())
            os.replace(tmp_in, inp)      # ayni dosya sisteminde atomik
        else:
            cv2.imwrite(inp, frame_bgr)  # kodlama basarisiz: eski yola dus
        # worker ilk yanıta kadar (DPVO yükleniyor olabilir) geniş bekle; sonra kısa.
        budget = self.wait_pose_sec if self._served else self.first_wait_sec
        t0 = time.time()
        while time.time() - t0 < budget:
            if os.path.exists(outp):
                try:
                    xyz = np.loadtxt(outp, dtype=np.float64).reshape(-1)
                    if not (xyz.size in (3, 8) or (xyz.size >= 9 and (xyz.size - 1) % 8 == 0)):
                        raise ValueError("beklenmedik sutun sayisi")
                    self._served = True
                    return xyz
                except Exception:
                    time.sleep(0.005)
                    continue
            time.sleep(0.005)
        # worker yanıt vermedi → degraded moda geç
        self._degraded = True
        return None

    # ---- RGB (FINAL-11) ----
    def _orb_kare(self, image_bgr):
        h, w = image_bgr.shape[:2]
        if w == self.rgb_z_width:
            return image_bgr
        return cv2.resize(image_bgr, (self.rgb_z_width, int(h * self.rgb_z_width / w)), interpolation=cv2.INTER_AREA)

    def _process_rgb(self, frame_bgr, frame_idx, gt_xyz, health, frame_key):
        vec = self._dpvo_xyz(frame_bgr)
        dpvo = d_med = n_cam = rms = None
        if vec is not None:
            dpvo = np.asarray(vec[:3], float)
            if vec.size == 8:
                d_med = float(vec[3]) if np.isfinite(vec[3]) else None
                n_cam = np.asarray(vec[4:7], float) if np.all(np.isfinite(vec[4:7])) else None
                rms = float(vec[7]) if np.isfinite(vec[7]) else None
        # ORB olcek-orani: her karede besle (kalibrasyonda da) — 960 px
        try:
            self.zf.feed(self._orb_kare(frame_bgr))
        except Exception:
            pass
        zA = None; zA_kal = None
        if health == 1 and gt_xyz is not None and not self.fk.kalibre:
            self._gt_kal.append(np.asarray(gt_xyz, float))
        elif not self._zA_hazir:
            # kalibrasyon bitiyor: ORB h0 (GSD) + kalibrasyon zA dizisi (seviye icin)
            self._zA_hazir = True
            try:
                gk = np.stack(self._gt_kal)
                if len(gk) < self.zf.G:                    # kalibrasyon kisa kaldiysa (gec baslama/kare kaybi)
                    self.zf.G = max(20, len(gk))           # ObjectSizeZ h0 + seviye penceresi gercek uzunluga
                if self.zf.calibrate_h0(gk[:, :2]) is not None:
                    zA_kal = self.zf.z(gk[:, 2])[:len(gk)]
                else:
                    print("[GPS] ORB h0 kalibre olmadi (kalibrasyonda hareket yok) -> A kaynagi kapali", flush=True)
            except Exception as e:
                print(f"[GPS] ORB kalibrasyon hatasi: {e}", flush=True)
        if self._zA_hazir and self.zf.h0 is not None and self._gt_kal:
            try:
                zA = float(self.zf.z(np.stack(self._gt_kal)[:, 2])[-1])
            except Exception:
                zA = None
        kalibre_onceki = self.fk.kalibre
        out = self.fk.adim(dpvo, d_med=d_med, n_cam=n_cam, plan_rms=rms, zA_ham=zA,
                           gt_xyz=(None if gt_xyz is None else np.asarray(gt_xyz, float)), health=health, zA_kal=zA_kal)
        if self.fk.kalibre and not kalibre_onceki:
            print(f"[GPS] FINAL-11 kalibrasyon bitti: {'NADIR/duz' if self.fk.nadir else 'OBLIK/rolyefli'} "
                  f"(aci {self.fk.aci:.1f} deg, rms {self.fk.rms:.3f}) | saglik {self.fk.saglik}", flush=True)
            for u in self.fk.uyarilar:
                print(f"[GPS] UYARI: {u}", flush=True)
        # 2026-09-19: olay_k YALNIZ kalibrasyon bitince dogar, kalibrasyon da ilk health=0
        # karesinde kapanir. GPS 450'yi asarak gelmeye devam ederse bu LOG satiri
        # AttributeError atiyor ve istisna, translation pakete eklenmeden yukari firliyordu
        # -> o kare sunucuya BOS gidiyordu (18 Eyl oturumu, kare 450). getattr ile log artik
        # hicbir seyi bozmuyor; kestirim hesabina dokunulmadi.
        if health == 1 and gt_xyz is not None and frame_idx >= self.gt_frames:
            print(f"[GPS] orta-oturum GPS (kare {frame_idx}) -> EKF/Kalman olcumu "
                  f"#{getattr(self.fk, 'olay_k', 0)}", flush=True)
        self.seq += 1
        if out is not None:
            self._last_world = np.asarray(out, np.float32)
        self._last_key = frame_key
        self._save_state()
        return tuple(float(v) for v in self._last_world)

    # ---- TERMAL (FINAL-TERMAL, 2026-09-04) ----
    def _process_termal(self, frame_bgr, frame_idx, gt_xyz, health, frame_key):
        vec = self._dpvo_xyz(frame_bgr)                       # [tekrar, K×8] (worker) veya None (degraded)
        kalibre_onceki = self.tk.recete is not None
        out = self.tk.adim(vec, frame_bgr, gt_xyz if health == 1 else None, health, frame_idx)
        if self.tk.recete is not None and not kalibre_onceki:
            print(f"[GPS] FINAL-TERMAL kestirici: {self.tk.kestirici_bilgi()}", flush=True)
        if health == 1 and gt_xyz is not None:
            out = tuple(float(v) for v in gt_xyz)             # GT aynen (bedava puan)
            if frame_idx >= self.gt_frames:
                print(f"[GPS] orta-oturum GPS (kare {frame_idx}) -> EKF/Kalman olcumu", flush=True)
        self.seq += 1
        if out is not None:
            self._last_world = np.asarray(out, np.float32)
        self._last_key = frame_key
        self._save_state()
        return tuple(float(v) for v in self._last_world)

    # ---- ana giriş: her kare ----
    def process(self, frame_bgr: np.ndarray, frame_idx: int,
                gt_xyz: Optional[Tuple[float, float, float]], health: int,
                frame_key: Optional[str] = None
                ) -> Optional[Tuple[float, float, float]]:
        """Bir kare için NED dünya konumu (x,y,z) döndürür. None → gönderme (nadir).
        gt_xyz: health=1 ise sunucudan gelen GT (x,y,z), değilse None.
        frame_key: karenin benzersiz kimliği (örn. frame_001255) — AYNI kare tekrar
        gelirse DPVO'ya tekrar verilmez (hareket yok, son konum döner)."""
        # --- DUPLICATE: reconnect'te sunucu son işlenen kareyi tekrar yollarsa ---
        # DPVO'ya TEKRAR verme (traje bozulur); hareket yok → son konumu aynen döndür.
        if frame_key is not None and frame_key == self._last_key:
            return tuple(float(v) for v in self._last_world)

        if self.modality == "rgb":
            return self._process_rgb(frame_bgr, frame_idx, gt_xyz, health, frame_key)
        return self._process_termal(frame_bgr, frame_idx, gt_xyz, health, frame_key)

        vec = self._dpvo_xyz(frame_bgr)
        dpvo = None if vec is None else np.asarray(vec[:3], np.float32)

        # --- health=1: GT var → AYNEN geri gönder (bedava puan) + kalibrasyon/drift-reset ---
        if health == 1 and gt_xyz is not None:
            gt = np.asarray(gt_xyz, np.float32)
            if not self._calibrated:
                # İLK 450 kalibrasyon dönemi: (dpvo,gt) çiftlerini biriktir
                if dpvo is not None:
                    self.est.update_calib(dpvo, gt, image_bgr=frame_bgr)
                # kalibrasyon bitince son GT xy donme merkezi olur
                self._xy_gt0 = np.asarray(gt[:2], np.float64).copy()
            else:
                # ORTA-OTURUM GT (örn. 1800): kalibre modelin o andaki kestirimi ile GT farkı
                # kadar OFSET uygula → birikmiş drift SIFIRLANIR, buradan devam.
                if dpvo is not None:
                    est_w = self.est.estimate(dpvo, image_bgr=frame_bgr)
                    if est_w is not None:
                        # ONCE: eski reanchor + mevcut duzeltmeyle O ANKI tahmin
                        tah = self._xy_duzelt(
                            np.asarray(est_w, np.float64)[:3] + self._reanchor)
                        self._xy_olay(tah[:2], gt[:2], frame_idx)
                        self._reanchor = gt - np.asarray(est_w, np.float32)[:3]
                    # RGB Z v2: bu GPS karesiyle iki Z kaynaginin agirligini yeniden
                    # hesapla ve ikisini de gercek z'ye demirle. GPS'in 450 sonrasi
                    # kac kez gelecegi bilinmiyor; her geliste iyilesir, hic gelmezse
                    # varsayilan agirlikla bozulmadan devam eder.
                    try:
                        n0 = getattr(self.est, "gps_olay_sayisi", 0)
                        self.est.gps_olayi(float(gt[2]))
                        if getattr(self.est, "gps_olay_sayisi", 0) > n0:
                            # Z ARTIK ICERIDEN demirlendi -> reanchor'in Z'sini SIFIRLA,
                            # yoksa cift duzeltme olur. XY reanchor'i aynen kalir.
                            self._reanchor[2] = 0.0
                            n = self.est.gps_olay_sayisi
                            if getattr(self.est, "modality", "") == "termal":
                                # termal: sonumlu ofset (kalici degil)
                                print(f"[GPS] orta-oturum GPS #{n} (kare {frame_idx}) "
                                      f"-> termal Z olcek kilidi demirlendi", flush=True)
                            else:
                                print(f"[GPS] orta-oturum GPS #{n} (kare {frame_idx}) "
                                      f"-> w_orb={self.est._w_orb:.3f}", flush=True)
                    except AttributeError:
                        pass                      # eski surum: kanca yok
            self.seq += 1
            self._last_world = gt
            result = tuple(float(v) for v in gt)
        else:
            # --- kalibrasyonu sonlandır (ilk defa health=0 geldiğinde) ---
            if not self._calibrated:
                try:
                    self._calibrated = self.est.finalize_calib()
                except Exception:
                    self._calibrated = False

            # --- health=0: sunucudaki GPS YANLIŞ/NaN → KULLANMA. Kendi kestirim + reanchor ---
            self.seq += 1
            if dpvo is not None and self._calibrated:
                w = self.est.estimate(dpvo, image_bgr=frame_bgr)
                if w is not None:
                    p = np.asarray(w, np.float64)[:3] + self._reanchor
                    self._last_world = np.asarray(self._xy_duzelt(p), np.float32)
            # degraded/kalibre değil → son bilinen konumu tut (drift göndermekten iyi)
            result = tuple(float(v) for v in self._last_world)

        # kesinti-resume için durumu diske yaz + kare kimliğini güncelle
        self._last_key = frame_key
        self._save_state()
        return result

    def stop(self):
        """Worker'a dur sinyali."""
        try:
            open(os.path.join(GPS_DIR, "STOP"), "w").close()
        except Exception:
            pass
