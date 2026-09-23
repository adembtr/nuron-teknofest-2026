#!/usr/bin/env python3
"""
FINAL-11 RGB GPS KESTİRİCİSİ (Görev 2) — kare kare, nedensel, durum taşır. 2026-09-03.

Kaynak çalışma: /home/adem/Desktop/gps/rgb_gps/cozum_xyz_final (OKU.md, SONUC_calisma_ozeti.md).
3 oturum (2026, 2025_ot2, 2025_ot3) ortalaması, GPS-kapalı karelerde ort |hata|:
    üretim (eski) X 1.93 / Y 2.09 / Z 3.29 m   →   FINAL-11 X 1.21 / Y 0.74 / Z 1.33 m

Girdi (her kare, gps_bridge'den):
  dpvo_xyz : DPVO ölçeksiz konum (3,)                                  [worker sütun 0-2]
  d_med    : DPVO yeni kare yama-derinliği medyanı (nan olabilir)      [worker sütun 3]
  n_cam, plan_rms : yer-düzlemi normali (kamera çerçevesi) ve göreli rms [worker sütun 4-7]; kalibrasyonda kullanılır
  zA_ham   : ORB ölçek-oranı Z (ObjectSizeZ.z(...)[-1]); kalibrasyon bitiminde zA_kal dizisi verilir
  gt_xyz / health : sunucudan GT (health=1) ya da None

Çıktı: (x, y, z) dünya (GT) çerçevesinde. health=1 karelerde GT aynen döner.

Mimari
  XY : DPVO artımları + EKF [px, py, θ, λ]; GPS olayı = ölçüm. λ-drift önseli (DPVO ölçeği kalibrasyon sonrası
       her karede -2e-5 log kayar), ilk olaylarda büyük ölçüm gürültüsü (kanıta bağlı güven).
  Z  : 3 kaynak — A: ORB ölçek-oranı, B: hizalanmış DPVO z, C: −s·d_med (mutlak). Kaynak başına Kalman:
       A → [c, k] (h0-ölçek), B/C → [c, a, b] (eğim: DPVO z hatası yatay yer değiştirmeyle doğrusal).
       Ön-ağırlık kalibrasyon GEOMETRİSİNDEN: nadir+düz (açı<15°, rms<0.08) → A ağır; oblik/rölyefli → B/C ağır
       ve olayda yeniden-ağırlık yok. GPS olayında hata-ağırlıklı EMA; ardışık GPS karelerinde patlama koruması.
"""
import os
import json
import numpy as np

from src.common.config import ROOT
from src.task2_position.aligner import fit_recipe_smart, movement_start_idx, apply_recipe

_CFG_YOL = os.path.join(ROOT, "config", "gps_rgb_final.json")
_VARSAYILAN = {
    "xy": {"q_p": 0.1, "q_th": 1e-3, "q_lam": 5e-4, "p0_th_deg": 0.5, "p0_lam": 0.02, "lam_drift": -2e-5,
           "r_takvim": [2.0, 1.0, 0.3], "adim_tavan": 3.0},
    "z": {"aci_esik": 15.0, "rms_esik": 0.08, "w_nadir": [0.8, 0.1, 0.1], "w_oblik": [0.1, 0.6, 0.3], "guc": 2.0,
          "taban": 0.15, "r": 0.7, "p_ab": 0.02, "q_c": 0.01, "q_ab": 1e-5, "p_k": 0.1, "ema_takvim": [0.15, 0.3],
          "oblik_olay": {"r": 1.5, "p_ab": 0.01, "ema_takvim": [0.0]}, "w_min_ara": 30, "patlama_maks": 2},
    # KORUMA (2026-09-03, no_gps testleri): kalibrasyon guvenilmezse (deniz/dokusuz ilk 450 -> DPVO olcegi bozuk;
    # ornek1: artik 30 m, s 0.35, Z 980 m!) tutucu moda gec; her modda fiziksel kelepceler.
    "koruma": {"artik_esik_m": 4.0, "s_min": 5.0, "s_max": 400.0, "yol_min_m": 30.0,
               "adim_maks_xy": 3.0, "adim_maks_z": 1.5, "z_sapma_maks": 60.0,
               "tutucu_hiz_pencere": 30, "tutucu_hiz_sure": 60, "tutucu_hiz_maks": 2.0},
}


def yukle_cfg():
    cfg = json.loads(json.dumps(_VARSAYILAN))
    try:
        d = json.load(open(_CFG_YOL))
        for k in ("xy", "z", "koruma"):
            cfg[k].update(d.get(k, {}))
    except Exception:
        pass
    return cfg


def R2(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def dR2(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[-s, -c], [c, -s]])


class KalmanKaynak:
    """Z kaynağı düzeltmesi. tip 'egim': durum [c,a,b], ölçüm gz−z = c + a·dx + b·dy ;
    tip 'olcek': [c,k], ölçüm gz−z = c + k·(z − z_kal) ; tip 'sapma': [c]."""
    def __init__(self, tip, r, p_ab=0.02, q_c=0.01, q_ab=1e-5, p_k=0.1, z_kal=0.0, tau=None):
        # ═══ Z-v2 (2026-09-12): (a) p_ab/p_k == 0 -> egim/olcek katsayisi TAMAMEN donuk (P ve Q sifir): tek GPS olayindan
        #     egim/olcek OGRENILMEZ (2026 1200+1600: tek olaydan ogrenilen k, irtifa ters yone gecince Z hatasini 1.4 -> 4.0 m yapiyordu);
        #     (b) tau: ofset c her karede exp(-1/tau) ile 0'a soner (kaynak hatasi rastgele yuruyus degil, ortalamaya donen drift).
        #     tau=None ve eski JSON anahtarlari -> eski davranis BIREBIR. Kaynak: gps/thermal_gps/work_fable_5.1/adim11_rgb_z/RAPOR.md
        self.tip, self.r, self.z_kal = tip, float(r), float(z_kal); self.tau = float(tau) if tau else None
        if tip == "egim":
            self.x = np.zeros(3); self.P = np.diag([0.3 ** 2, p_ab ** 2, p_ab ** 2]); self.Q = np.diag([q_c ** 2, q_ab ** 2, q_ab ** 2])
            if p_ab == 0: self.Q[1, 1] = self.Q[2, 2] = 0.0                  # Z-v2: donuk egim
        elif tip == "olcek":
            self.x = np.zeros(2); self.P = np.diag([0.3 ** 2, p_k ** 2]); self.Q = np.diag([q_c ** 2, 1e-12])
            if p_k == 0: self.Q[1, 1] = 0.0                                   # Z-v2: donuk olcek
        else:
            self.x = np.zeros(1); self.P = np.diag([0.3 ** 2]); self.Q = np.diag([q_c ** 2])

    def h(self, z_ham, dxy):
        if self.tip == "egim":
            return np.array([1.0, dxy[0], dxy[1]])
        if self.tip == "olcek":
            return np.array([1.0, z_ham - self.z_kal])
        return np.array([1.0])

    def tahmin(self):
        self.P = self.P + self.Q
        tau = getattr(self, "tau", None)                                      # Z-v2 (eski bridge_state.pkl ile uyumlu)
        if tau: self.x[0] *= np.exp(-1.0 / tau)

    def duzelt(self, z_ham, dxy):
        return float(z_ham + self.h(z_ham, dxy) @ self.x)

    def olcum(self, z_ham, dxy, gz):
        hh = self.h(z_ham, dxy)
        y = (gz - z_ham) - hh @ self.x
        S = hh @ self.P @ hh + self.r ** 2
        K = self.P @ hh / S
        self.x = self.x + K * y
        self.P = (np.eye(len(self.x)) - np.outer(K, hh)) @ self.P


class FinalKestirici:
    """Saf numpy; pickle'lanabilir (kesinti-resume). Her kare adim(...) çağrılır."""

    def __init__(self, G=450, cfg=None):
        self.cfg = cfg or yukle_cfg()
        self.G = int(G)
        self.kalibre = False
        self.k = 0                       # işlenen kare sayısı
        self._dp, self._gt, self._dmed, self._ncam, self._rms, self._zA = [], [], [], [], [], []
        self.son = None
        self.uyarilar = []
        self.saglik = {}                 # kalibrasyon teşhisi (log için)

    # ---------- kalibrasyon ----------
    def _kalib_ekle(self, dpvo, gt, d_med, n_cam, rms, zA):
        self._dp.append(np.asarray(dpvo, float)); self._gt.append(np.asarray(gt, float))
        self._dmed.append(np.nan if d_med is None else float(d_med))
        self._ncam.append(None if n_cam is None else np.asarray(n_cam, float))
        self._rms.append(np.nan if rms is None else float(rms))
        self._zA.append(np.nan if zA is None else float(zA))

    def kalib_bitir(self, zA_kal=None):
        """Kalibrasyonu kapat. zA_kal: kalibrasyon karelerinin ORB Z dizisi (seviye için son 10 kare)."""
        if self.kalibre:
            return
        dp = np.array(self._dp); gt = np.array(self._gt); n = len(dp)
        Zc = self.cfg["z"]; Xc = self.cfg["xy"]
        s0 = movement_start_idx(gt); s0 = s0 if n - s0 >= 100 else 0
        self.R, self.s, self.t, _ = fit_recipe_smart(dp[s0:], gt[s0:], G=n - s0)
        al = np.nan_to_num(apply_recipe(dp, self.R, self.s, self.t), nan=0.0)
        gz = gt[:, 2]; self.gz_kal = float(np.median(gz[-10:]))
        # teşhis: kalibrasyon yolu, hizalama artığı
        art = np.linalg.norm(al[s0:, :2] - gt[s0:, :2], axis=1)
        self.saglik = dict(kare=n, hareket_bas=int(s0), yol_m=float(np.linalg.norm(np.diff(gt[:, :2], axis=0), axis=1).sum()),
                           olcek_s=float(self.s), artik_ort=float(np.mean(art)) if len(art) else np.nan,
                           dpvo_gecerli=float(np.mean(np.isfinite(dp).all(1))))
        # geometri: nadir + düz arazi mi?
        nc = [x for x in self._ncam if x is not None and np.all(np.isfinite(x))]
        if len(nc) >= 20:
            nrm = np.median(np.array(nc), 0); nrm /= max(np.linalg.norm(nrm), 1e-9)
            self.aci = float(np.degrees(np.arccos(min(1.0, abs(nrm[2]))))); self.rms = float(np.nanmedian(self._rms))
        else:
            self.aci, self.rms = 90.0, 1.0
            self.uyarilar.append("kalibrasyonda yer-duzlemi normali yok -> oblik rejim varsayiliyor (nadir oturumda Z bozulur)")
        self.nadir = self.aci < Zc["aci_esik"] and self.rms < Zc["rms_esik"]
        P = dict(Zc)
        if not self.nadir:
            P.update(Zc["oblik_olay"])
        self.ema_takvim = list(P["ema_takvim"]); self.guc = P["guc"]; self.taban = P["taban"]
        # kaynak seviyeleri (son 10 kalibrasyon karesi)
        zB = al[:, 2] - al[0, 2]; self.levB = float(np.median(zB[-10:]) - self.gz_kal)
        dm = np.array(self._dmed); dmv = dm[np.isfinite(dm)]
        self._dm_pencere = [float(x) for x in dmv[-5:]]; self.dm_son = float(np.median(self._dm_pencere)) if len(self._dm_pencere) else np.nan
        zC = -self.s * dm; zCv = zC[np.isfinite(zC)]
        self.levC = float(np.median(zCv[-10:]) - self.gz_kal) if len(zCv) >= 10 else None
        if zA_kal is not None:
            zA_kal = np.asarray(zA_kal, float); zAv = zA_kal[np.isfinite(zA_kal)]
        else:
            zA_kal = np.array(self._zA, float); zAv = zA_kal[np.isfinite(zA_kal)]
        self.levA = float(np.median(zAv[-10:]) - self.gz_kal) if len(zAv) >= 10 else None
        if self.levA is None:
            self.uyarilar.append("ORB olcek-orani (zA) yok -> A kaynagi devre disi")
        if self.levC is None:
            self.uyarilar.append("DPVO derinlik (d_med) yok -> C kaynagi devre disi")
        r_ol = P.get("r_olay", P["r"]); tau_c = P.get("tau_c")               # Z-v2 (2026-09-12): anahtar yoksa eski davranis
        self.kA = KalmanKaynak("olcek", r_ol, p_k=P["p_k"], q_c=P["q_c"], z_kal=self.gz_kal, tau=tau_c)
        self.kB = KalmanKaynak("egim", r_ol, p_ab=P["p_ab"], q_c=P["q_c"], q_ab=P["q_ab"], tau=tau_c)
        self.kC = KalmanKaynak("egim", r_ol, p_ab=P["p_ab"], q_c=P["q_c"], q_ab=P["q_ab"], tau=tau_c)
        w = np.array(Zc["w_nadir"] if self.nadir else Zc["w_oblik"], float); self.w = w / w.sum()
        # XY-EKF
        self.x = np.array([gt[-1, 0], gt[-1, 1], 0.0, 0.0])
        self.Pxy = np.diag([0.05 ** 2, 0.05 ** 2, np.radians(Xc["p0_th_deg"]) ** 2, Xc["p0_lam"] ** 2])
        self.Q = np.diag([Xc["q_p"] ** 2, Xc["q_p"] ** 2, Xc["q_th"] ** 2, Xc["q_lam"] ** 2])
        self.H = np.zeros((2, 4)); self.H[0, 0] = self.H[1, 1] = 1.0
        self.al_onceki = al[-1].copy(); self.d_onceki = np.zeros(2); self.zB_onceki = float(zB[-1]); self.p0 = gt[-1, :2].copy()
        self.olay_k = 0; self.est_onceki = None; self._son_olay_kare = -10 ** 9; self._patlama_sayac = 0
        # ─── GÜVEN KAPISI: hizalama artığı büyük / ölçek anlamsız / yol kısa → kalibrasyon GÜVENİLMEZ ───
        Kc = self.cfg["koruma"]
        neden = []
        if not np.isfinite(self.saglik["artik_ort"]) or self.saglik["artik_ort"] > Kc["artik_esik_m"]:
            neden.append(f"hizalama artigi {self.saglik['artik_ort']:.1f} m > {Kc['artik_esik_m']}")
        if not (Kc["s_min"] <= self.s <= Kc["s_max"]):
            neden.append(f"olcek s {self.s:.2f} makul degil")
        if self.saglik["yol_m"] < Kc["yol_min_m"]:
            neden.append(f"kalibrasyon yolu {self.saglik['yol_m']:.0f} m kisa")
        self.guven = len(neden) == 0
        if not self.guven:
            self.uyarilar.append("KALIBRASYON GUVENILMEZ (" + "; ".join(neden) + ") -> TUTUCU mod: XY son GT + sonumlu hiz, Z son GT; GPS olayinda yeniden demir")
        # tutucu mod: son GT konum/hiz (kalibrasyonun son karelerinden)
        W = int(Kc["tutucu_hiz_pencere"])
        vv = np.diff(gt[-W:, :2], axis=0) if len(gt) > W else np.diff(gt[:, :2], axis=0)
        self.t_hiz = np.median(vv, axis=0) if len(vv) else np.zeros(2)
        hn = float(np.linalg.norm(self.t_hiz)); hm = Kc["tutucu_hiz_maks"]
        if hn > hm: self.t_hiz = self.t_hiz * (hm / hn)
        self.t_demir = gt[-1].copy(); self.t_demir_k = self.k
        self.z_demir = float(gt[-1, 2]); self.son_cikti = gt[-1].copy()
        self.kalibre = True
        # kalibrasyon listeleri artık gerekmez (bellek/pickle)
        self._dp, self._gt, self._ncam = [], [], []

    # ---------- kare ----------
    def adim(self, dpvo_xyz, d_med=None, n_cam=None, plan_rms=None, zA_ham=None, gt_xyz=None, health=0, zA_kal=None):
        """Bir kare işle; (x, y, z) döndür (kalibrasyon bitmeden health=0 gelirse son bilinen)."""
        Xc = self.cfg["xy"]; Zc = self.cfg["z"]
        if health == 1 and gt_xyz is not None and not self.kalibre:
            if dpvo_xyz is not None:
                self._kalib_ekle(dpvo_xyz, gt_xyz, d_med, n_cam, plan_rms, zA_ham)
            self.son = tuple(float(v) for v in gt_xyz); self.k += 1
            return self.son
        if not self.kalibre:
            if len(self._dp) < 50:
                self.k += 1
                return self.son
            self.kalib_bitir(zA_kal)
        # --- DPVO artımı (hizalanmış) ---
        if dpvo_xyz is None or not np.all(np.isfinite(dpvo_xyz)):
            al = self.al_onceki.copy()
        else:
            al = apply_recipe(np.asarray(dpvo_xyz, float)[None], self.R, self.s, self.t)[0]
        d = al[:2] - self.al_onceki[:2]; nd = float(np.linalg.norm(d))
        if not np.isfinite(nd) or nd > Xc["adim_tavan"]:
            d = self.d_onceki
        dz = float(al[2] - self.al_onceki[2]); dz = dz if np.isfinite(dz) and abs(dz) < Xc["adim_tavan"] else 0.0
        self.d_onceki = d; self.al_onceki = al
        # --- XY-EKF tahmin ---
        self.x[3] += Xc["lam_drift"]
        th, lam = self.x[2], self.x[3]; Rt = R2(th); es = np.exp(lam)
        self.x[:2] = self.x[:2] + es * (Rt @ d)
        F = np.eye(4); F[:2, 2] = es * (dR2(th) @ d); F[:2, 3] = es * (Rt @ d)
        self.Pxy = F @ self.Pxy @ F.T + self.Q
        # --- Z kaynakları (ham, seviyeli) ---
        self.zB_onceki += dz; zB = self.zB_onceki - self.levB
        if d_med is not None and np.isfinite(d_med):
            self._dm_pencere = (self._dm_pencere + [float(d_med)])[-5:]; self.dm_son = float(np.median(self._dm_pencere))
        zC = (-self.s * self.dm_son - self.levC) if (self.levC is not None and np.isfinite(self.dm_son)) else None
        zA = (float(zA_ham) - self.levA) if (self.levA is not None and zA_ham is not None and np.isfinite(zA_ham)) else None
        for kk in (self.kA, self.kB, self.kC):
            kk.tahmin()
        dxy = self.x[:2] - self.p0
        est = [self.kA.duzelt(zA, dxy) if zA is not None else None, self.kB.duzelt(zB, dxy), self.kC.duzelt(zC, dxy) if zC is not None else None]
        if health == 1 and gt_xyz is not None:
            gt = np.asarray(gt_xyz, float)
            # patlama koruması (Z): ardışık GPS kareleri
            if self.k - self._son_olay_kare >= Zc["w_min_ara"]:
                self._patlama_sayac = 0
            z_guncelle = self._patlama_sayac < Zc["patlama_maks"]; self._patlama_sayac += 1; self._son_olay_kare = self.k
            # XY ölçüm (kanıta bağlı R)
            rt = Xc["r_takvim"]; rr = rt[min(self.olay_k, len(rt) - 1)]; Rm = np.eye(2) * rr ** 2
            y = gt[:2] - self.x[:2]; S = self.H @ self.Pxy @ self.H.T + Rm; K = self.Pxy @ self.H.T @ np.linalg.inv(S)
            self.x = self.x + K @ y; self.Pxy = (np.eye(4) - K @ self.H) @ self.Pxy; self.x[:2] = gt[:2]
            # Z: ağırlık (önceki kare kestirimleriyle) + Kalman ölçümleri
            if z_guncelle and self.est_onceki is not None:
                e = np.array([abs(v - gt[2]) if v is not None else 50.0 for v in self.est_onceki])
                ww = (1 / (e + self.taban)) ** self.guc; ww /= ww.sum()
                ema = self.ema_takvim[min(self.olay_k, len(self.ema_takvim) - 1)]
                self.w = self.w + ema * (ww - self.w)
            dxy_gt = gt[:2] - self.p0
            if z_guncelle:
                if zA is not None:
                    self.kA.olcum(zA, dxy_gt, gt[2])
                self.kB.olcum(zB, dxy_gt, gt[2])
                if zC is not None:
                    self.kC.olcum(zC, dxy_gt, gt[2])
            self.olay_k += 1; self.est_onceki = est
            # tutucu mod demirleri (guvenilmez kalibrasyonda) + Z sapma demiri (her modda)
            self.t_demir = gt.copy(); self.t_demir_k = self.k; self.z_demir = float(gt[2]); self.son_cikti = gt.copy()
            self.son = tuple(float(v) for v in gt); self.k += 1
            return self.son
        # --- füzyon ---
        w = self.w.copy(); vals = np.array([v if v is not None else 0.0 for v in est]); w[[v is None for v in est]] = 0.0
        z = float(w @ vals / w.sum()) if w.sum() > 0 else (self.son[2] if self.son else 0.0)
        self.est_onceki = est
        cik = np.array([float(self.x[0]), float(self.x[1]), z])
        Kc = self.cfg["koruma"]
        if not self.guven:
            # TUTUCU: son GT demirinden sonumlu sabit hizla ilerle (en fazla tutucu_hiz_sure kare), Z = son GT z.
            dt = min(self.k - self.t_demir_k, int(Kc["tutucu_hiz_sure"]))
            cik = np.array([self.t_demir[0] + self.t_hiz[0] * dt, self.t_demir[1] + self.t_hiz[1] * dt, self.z_demir])
        else:
            # her modda: Z son GPS demirinden fazla sapamaz (mutlak kelepce; iyi oturumlarda devreye girmez)
            cik[2] = float(np.clip(cik[2], self.z_demir - Kc["z_sapma_maks"], self.z_demir + Kc["z_sapma_maks"]))
        # kare-arasi fiziksel kelepce (iyi oturumlarda maks gercek adim ~1.5 m; devreye girmez)
        d = cik - self.son_cikti
        d[:2] = np.clip(d[:2], -Kc["adim_maks_xy"], Kc["adim_maks_xy"]); d[2] = np.clip(d[2], -Kc["adim_maks_z"], Kc["adim_maks_z"])
        cik = self.son_cikti + d; self.son_cikti = cik.copy()
        self.son = tuple(float(v) for v in cik); self.k += 1
        return self.son
