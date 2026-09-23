#!/usr/bin/env python3
"""
FINAL-TERMAL v4 (G″) akış kestiricisi (2026-09-11) — kare kare, nedensel, durum taşır.
Kaynak: bu dosyanın 2026-09-04 üretim sürümü + work_fable_5.1/adim2(kestirici4.py)/adim2b(kestirici4b.py) v4 blokları.
YENİ (hepsi JSON ile açılır; JSON'da yoksa varsayılanlar KAPALI = eski üretim davranışı bit-bit):
  1) xy.kuyruk_donme  : kalibrasyon DÖNMESİ pencerenin son xy.kuyruk_n karesinden; ölçek/öteleme akıllı pencerede kalır.
  2) xy.rampa         : doyan ölçek rampası lam_ramp(t)=log(1+A*(1-exp(-(t-G+1)/tau))); A kalibrasyondan kestirilir (GPS'siz).
  3) xy.rampa_kapi    : A kapısı — "isaret_buyukluk" (ÖNERİLEN) | "t" (KIRILGAN, kullanmayın) | "isaret" | "t_ve_isaret".
  4) xy.duzey_sizinti : DPVO'nun düşeye sızan yatay hareketini yatay adıma geri kazandırır.
  5) z.nadir_kural    : nadir kararı açı-birincil ("aci"), rms yalnız gevşek tavan (z.rms_tavan).
  6) z.olay_min_kare  : kalibrasyondan bu kadar kare geçmeden gelen GPS olayı Z kaynaklarını/ağırlıkları GÜNCELLEMEZ.
Geri dönüş: NURON_TERMAL_V4=0 → config/gps_termal_final.json (v1/eski üretim) yüklenir.

Girdi (her kare): dpvo_xyz (K DPVO örneğinin SAĞLAM ENSEMBLE ile birleştirilmiş sanal konumu, 1/50 ölçekli), d_med (ensemble metrik irtifa / 50),
  n_cam, plan_rms (kalibrasyonda; nadir/oblik kararı), zA_ham (ORB ölçek kilidi z'si — z_termal.OlcekKilidi), gt_xyz|None, health, r_orb/px_orb, kare_tekrar.
Çıktı: (x, y, z) dünya (NED).
XY: kalibrasyon Umeyama (akıllı pencere, aligner.fit_recipe_smart) [+ ölçek eğilimi] → EKF [px,py,θ,λ]; GPS olayı = ölçüm.
Z : A = ORB ölçek kilidi, B = hizalanmış DPVO z (birikimli), C = −s·d_med (yama derinliği, birikimsiz); geometri kuralı (nadir→C, oblik→B);
    GPS olayında kaynak başına Kalman + kanıt ağırlığı. Koruma: göreli artık kapısı → yarı tutucu (XY DPVO, Z son GT); kelepçe; Z sapma sınırı.
Ölçülen (GPS-kapalı ort |hata| X/Y/Z, tekrar-kare atlamalı DPVO, K=2): ot4 yok 1.18/2.15/2.20, 10 olay 0.99/0.63/1.65; 2026 yok 2.3/15.8/2.3, 10 olay 1.4/5.0/2.4.
Canlı-benzeri doğrulama: ot4 1.34/1.27/1.66; 2026 (1400+1800) 1.74/13.60/1.86.
"""
import os, json, copy
import numpy as np
from src.task2_position.aligner import umeyama, apply_recipe, movement_start_idx, fit_recipe_smart, olcek_carpani, OLCEK_PENCERELERI

_CONF_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "config")
_CFG = os.path.join(_CONF_DIR, "gps_termal_final.json")            # v1 / eski üretim
_CFG_V4 = os.path.join(_CONF_DIR, "gps_termal_final_v4.json")      # v4 (G″)
_CFG_V5 = os.path.join(_CONF_DIR, "gps_termal_final_v5.json")      # v5 (2026-09-19, folder4 saglamlastirmasi)


def kalibre(dp, gt):
    """Üretim kalibrasyonu: hareket kapısı + akıllı pencere. Döner R, s, t, pencere, artık(m), yol(m)."""
    dp = np.asarray(dp, float); gt = np.asarray(gt, float); n = len(dp)
    s0 = movement_start_idx(gt); s0 = s0 if n - s0 >= 100 else 0
    R, s, t, w = fit_recipe_smart(dp[s0:], gt[s0:], G=n - s0)
    al = apply_recipe(dp[s0:], R, s, t)
    res = float(np.linalg.norm(al[w[0]:w[1]] - gt[s0:][w[0]:w[1]], axis=1).mean())   # artık: SEÇİLEN akıllı pencere üzerinde (koruma eşikleri buna göre ayarlandı: 2026 .29, ot4 .96, ornek1 4.7, ornek2 3.8 m)
    yol = float(np.sum(np.linalg.norm(np.diff(gt[s0:, :2], axis=0), axis=1)))
    return R, s, t, (w[0] + s0, w[1] + s0), res, yol


def secili_cfg():
    """GERİ DÖNÜŞ ANAHTARI: NURON_TERMAL_V4=0 -> eski gps_termal_final.json (v1 davranışı birebir).
    Aksi halde v4 dosyası varsa o, yoksa eski dosya."""
    if os.environ.get("NURON_TERMAL_V4", "1").strip() == "0":
        return _CFG                                   # v1 / eski uretim
    if os.environ.get("NURON_TERMAL_V5", "1").strip() != "0" and os.path.exists(_CFG_V5):
        return _CFG_V5                                # v5 (varsayilan)
    return _CFG_V4 if os.path.exists(_CFG_V4) else _CFG


def yukle_prm():
    try:
        with open(secili_cfg()) as f:
            c = json.load(f)
        return {k: v for k, v in c.items() if k in ("xy", "z", "koruma", "uy2", "E")}, c
    except Exception:
        return {}, {}


def olcek_egilimi_gen(pred, gt, min_alt=40):
    """aligner.olcek_egilimi'nin genellenmişi: KISA pencerelere de izin verir (n>=90, alt-pencere>=min_alt)
    ve alt-pencere s/s_ref dizisini de döndürür. Döner: (egim /kare, t_ist, merkez, s_orani_dizisi).
    Kaynak: work_fable_5.1/adim2_kestirici_rampa/kestirici4.py [k4:21-47]."""
    pred = np.asarray(pred, float); gt = np.asarray(gt, float); n = len(pred)
    if n < 90:
        return 0.0, 0.0, 0.0, []
    ss, xs = [], []
    for f0, f1 in OLCEK_PENCERELERI:
        a, b = int(n * f0), int(n * f1)
        if b - a < min_alt:
            continue
        gr = gt[a:b].copy(); gr[:, 2] *= -1
        _, s_, _ = umeyama(pred[a:b], gr); ss.append(s_); xs.append((a + b) / 2.0)
    if len(ss) < 4:
        return 0.0, 0.0, 0.0, []
    gr = gt.copy(); gr[:, 2] *= -1
    _, s_ref, _ = umeyama(pred, gr)
    if not np.isfinite(s_ref) or s_ref <= 0:
        return 0.0, 0.0, 0.0, []
    y = np.asarray(ss) / s_ref; x = np.asarray(xs); xm, ym = x.mean(), y.mean()
    Sxx = ((x - xm) ** 2).sum()
    if Sxx <= 0:
        return 0.0, 0.0, 0.0, list(map(float, y))
    egim = ((x - xm) * (y - ym)).sum() / Sxx
    art = y - (egim * (x - xm) + ym); sig2 = (art ** 2).sum() / max(len(x) - 2, 1)
    se = np.sqrt(sig2 / Sxx) if sig2 > 0 else 0.0
    return float(egim), float(abs(egim) / se if se > 0 else 0.0), float(xm), list(map(float, y))


class SaglamEnsemble:
    """K DPVO örneğinin kalibrasyon sonrası artımlarını nedensel birleştirir (kaynak: ens_poz.py --saglam).
    Üye başına uyumsuzluk EMA'sı (|d_k − medyan| / |medyan|) + diğer üyelere GÖRE d_med çökme tespiti (×2 / ÷2 30 karede) → kalıcı dışlama; en az 2 üye kalır."""
    def __init__(self, K, esik=0.5, ema=0.02):
        self.K = K; self.esik = esik; self.ema = ema; self.uyum = np.zeros(K); self.kalici = np.zeros(K, bool); self.DM = [[] for _ in range(K)]; self.i = 0; self.dislanan_kare = {}

    def birlestir(self, artimlar, dm_metrik):
        d = np.asarray(artimlar, float); ref = np.median(d, axis=0)
        for k in range(self.K):
            sapma = np.linalg.norm(d[k, :2] - ref[:2]) / (np.linalg.norm(ref[:2]) + 0.05); self.uyum[k] = (1 - self.ema) * self.uyum[k] + self.ema * min(sapma, 5.0)
            self.DM[k].append(float(dm_metrik[k]) if dm_metrik[k] is not None else np.nan)
            if len(self.DM[k]) > 70: self.DM[k] = self.DM[k][-70:]
        if self.i >= 60 and self.K >= 3:
            DMm = np.array([x[-61:] for x in self.DM]); onc = np.nanmedian(DMm[:, :30], axis=1); sim = np.nanmedian(DMm[:, -6:], axis=1)
            oran = sim / np.maximum(onc, 1e-6); ref_o = np.nanmedian(oran)
            for k in range(self.K):
                r_k = oran[k] / max(ref_o, 1e-6)
                if not self.kalici[k] and np.isfinite(r_k) and (r_k > 2.0 or r_k < 0.5):
                    self.kalici[k] = True; self.dislanan_kare[k] = self.i
        dahil = (self.uyum < self.esik) & ~self.kalici
        if dahil.sum() < 2:
            aday = np.where(~self.kalici)[0] if (~self.kalici).sum() >= 2 else np.arange(self.K); sira = aday[np.argsort(self.uyum[aday])[:2]]; dahil = np.zeros(self.K, bool); dahil[sira] = True
        self.i += 1
        dmv = np.array([self.DM[k][-1] for k in range(self.K) if not self.kalici[k]], float)
        return np.median(d[dahil], axis=0), (float(np.nanmedian(dmv)) if np.isfinite(dmv).any() else None)


PRM = dict(
    xy=dict(q_p=0.1, q_th=3e-4, q_lam=2e-4, p0_th_deg=0.5, p0_lam=0.1, lam_drift=0.0, r_takvim=[2.0, 1.0, 0.3], adim_tavan=10.0,   # tavan 10: tekrar-kare boslugu sonrasi 4-6 m adimlar mesru (3 m tavan d1_w960 Y 3.2->9.8 bozuyordu); termal taramasi (taban + tekrar_d1, iki kez ayni en iyi)
            olcek_egilimi=True, egilim_T0=4.0, egilim_T1=12.0, egilim_doyum=0.25, demir="al",
            # durma/telafi filtresi: DPVO bazen 3-6 kare durup sonra 2-8 kat buyuk adimlarla telafi ediyor (2026: 509-514/966-971/...)
            derinlik_alfa=0.0, derinlik_kaynak="A", adim_filtre=False, durma_oran=0.15, telafi_kat=2.0, hiz_pencere=20, telafi_borc=True, telafi_kirp=False,
            # ── v4 EK 1: kalibrasyon DONMESI kuyruktan (olcek/oteleme akilli pencereden) ──
            kuyruk_donme=False, kuyruk_n=100,
            kuyruk_mod="xy",             # "xy": XY zinciri kuyruk R, Z kaynagi B akilli-pencere R (ONERILEN) | "yaw": yalniz yaw farki | "tam": her sey kuyruk R
            kuyruk_rampa_bagli=True,     # True: kuyruk donmesi YALNIZ rampa etkinken (A!=0) uygulanir
            # ── v4 EK 2/3: doyan olcek rampasi + A_hat kapisi ──
            rampa="yok",                 # "yok" | "drift" (lam durumuna artim) | "onsel" (toplamsal)
            rampa_A=None,                # None -> A_hat kalibrasyondan kestirilir; sayi -> sabit
            rampa_tau=100.0,
            rampa_T0=2.0, rampa_T1=4.0,          # A_hat t-istatistigi kapisi (KIRILGAN: canli kosuda t=2.461 ile kapandi)
            rampa_A_maks=0.15, rampa_A_min=0.0,  # doyum ust siniri / negatif rampaya izin
            rampa_kapi="t",              # "t" (v1/G) | "isaret" | "isaret_buyukluk" (ONERILEN) | "t_ve_isaret"
            rampa_k=1.0, rampa_A_esik=0.0,
            rampa_span=(200,),           # egim tahmini icin kalibrasyon kuyruk uzunluklari (medyan)
            rampa_ufuk=0.5,              # A_ham = egim x (G + f*(toplam-G) - merkez)
            rampa_nadir=False, rampa_mono=0.0, rampa_olay="devam",
            # ── v4 EK 4: DUSEY SIZINTI duzeltmesi ──
            duzey_sizinti="yok",         # "yok" | "sifir" | "ac" | "fuzyon"
            sizinti_kirp=(0.85, 1.25), sizinti_pencere=20, sizinti_yumusat="medyan", sizinti_nadir=False),
    z=dict(aci_esik=15.0, rms_esik=0.08, w_nadir=(0.1, 0.1, 0.8), w_oblik=(0.2, 0.6, 0.2), guc=4.0, taban=0.15, r=0.7, p_ab=0.02,
           q_c=0.01, q_ab=1e-5, p_k=0.3, ema_takvim=(0.15, 0.3), oblik_olay=dict(r=1.5, p_ab=0.01, ema_takvim=(0.0,)),
           w_min_ara=30, patlama_maks=2, dm_med=5, c_lam=True, lev_n=10, kaynaklar=("A", "B", "C"), kB_tip="egim", kC_tip="egim", kA_tip="olcek", kB_tip_oblik="olcek",
           # ── v4: nadir kapisi + erken GPS olayi korumasi ──
           nadir_kural="ve",     # "ve" (v1: aci<aci_esik VE rms<rms_esik) | "aci" (ONERILEN: aci birincil, rms yalniz gevsek tavan)
           rms_tavan=0.12,       # "aci" kuralinda rms'in gevsek emniyet tavani
           olay_min_kare=0),     # kalibrasyondan bu kadar kare gecmeden gelen GPS olayi Z kaynaklarini/agirliklari GUNCELLEMEZ (0 = v1)
    koruma=dict(acik=True, artik_esik_m=4.0, artik_oran_esik=0.02, tutucu_xy=False, s_min=5.0, s_max=400.0, yol_min_m=30.0, adim_maks_xy=3.0, adim_maks_z=1.5,
                z_sapma_maks=60.0, tutucu_hiz_pencere=30, tutucu_hiz_sure=60, tutucu_hiz_maks=2.0,
                # DPVO olcek COKMESI tespiti (tek ornek guvenligi): C-irtifasi (s*d_med) 30 karede x1.8'den fazla degisirken ORB kilidi (zA) irtifasi %25'ten az degisirse
                cokme_acik=False, cokme_pencere=30, cokme_oran=1.8, cokme_zA_tol=0.3, cokme_q_oran=3.0,
                # F_z: tutucu modda Z kaynagi. "son_gt"=TABAN (donuk), "A"=ORB olcek kilidi, "A_delta"=son GT + A artimi,
                #      "fuz"=normal fuzyon, "A_fuz"=A varsa A yoksa fuzyon
                tutucu_z="son_gt", tutucu_w_A=False, tutucu_z_k=1.0, tutucu_z_beta_maks=2e-3),
    toplam_kare=2250,
)


def prm_birlestir(ust, alt):
    out = copy.deepcopy(ust)
    for k, v in (alt or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = prm_birlestir(out[k], v)
        else:
            out[k] = v
    return out


def R2(a):
    c, s = np.cos(a), np.sin(a); return np.array([[c, -s], [s, c]])


def dR2(a):
    c, s = np.cos(a), np.sin(a); return np.array([[-s, -c], [c, -s]])


class KalmanKaynak:
    """'egim': [c,a,b] (gz-z = c + a dx + b dy) | 'olcek': [c,k] (gz-z = c + k (z - z_kal)) | 'sapma': [c]"""
    def __init__(self, tip, r, p_ab=0.02, q_c=0.01, q_ab=1e-5, p_k=0.1, z_kal=0.0):
        self.tip = tip; self.r = r; self.z_kal = z_kal
        if tip == "egim":
            self.x = np.zeros(3); self.P = np.diag([0.3 ** 2, p_ab ** 2, p_ab ** 2]); self.Q = np.diag([q_c ** 2, q_ab ** 2, q_ab ** 2])
        elif tip == "olcek":
            self.x = np.zeros(2); self.P = np.diag([0.3 ** 2, p_k ** 2]); self.Q = np.diag([q_c ** 2, 1e-12])
        else:
            self.x = np.zeros(1); self.P = np.diag([0.3 ** 2]); self.Q = np.diag([q_c ** 2])

    def h(self, z_ham, dxy):
        if self.tip == "egim": return np.array([1.0, dxy[0], dxy[1]])
        if self.tip == "olcek": return np.array([1.0, z_ham - self.z_kal])
        return np.array([1.0])

    def tahmin(self): self.P = self.P + self.Q
    def duzelt(self, z_ham, dxy): return float(z_ham + self.h(z_ham, dxy) @ self.x)

    def olcum(self, z_ham, dxy, gz):
        h = self.h(z_ham, dxy); y = (gz - z_ham) - h @ self.x; S = h @ self.P @ h + self.r ** 2; K = self.P @ h / S
        self.x = self.x + K * y; self.P = (np.eye(len(self.x)) - np.outer(K, h)) @ self.P


class _TK_TABAN:
    def __init__(self, G=450, prm=None):
        self.G = G; self.P = prm_birlestir(PRM, prm)
        self.kalibre = False; self.k = 0; self.son = None
        self._dp, self._gt, self._dmed, self._ncam, self._rms = [], [], [], [], []
        self.uyarilar = []; self.saglik = True; self.tutucu = False
        self.bilgi = {}

    # ---------- kalibrasyon ----------
    def _kalib_ekle(self, dpvo, gt, d_med, n_cam, rms, px_orb=None):
        self._px_kal = getattr(self, '_px_kal', []); self._px_kal.append(np.nan if px_orb is None else float(px_orb))   # F_z
        self._dp.append(np.asarray(dpvo, float)); self._gt.append(np.asarray(gt, float))
        self._dmed.append(np.nan if d_med is None else float(d_med))
        self._ncam.append(None if n_cam is None else np.asarray(n_cam, float)); self._rms.append(np.nan if rms is None else float(rms))

    def _kalib_bitir(self):
        P = self.P; K = P["koruma"]; Z = P["z"]; XY = P["xy"]
        dp = np.array(self._dp); gt = np.array(self._gt); n = len(dp)
        self.R, self.s, self.t, self.pencere, self.artik, self.yol = kalibre(dp, gt)
        # ── v4 (b): geometri (nadir/oblik) R,s,t'den BAGIMSIZ; A_hat kapisi ve sizinti_nadir icin kalibrasyon BASINDA gerekir ──
        nc = [x for x in self._ncam if x is not None and np.all(np.isfinite(x))]
        if len(nc) >= 20:
            nrm = np.median(np.array(nc), 0); nrm /= np.linalg.norm(nrm); self.aci = float(np.degrees(np.arccos(abs(nrm[2])))); self.rms = float(np.nanmedian(self._rms))
        else:
            self.aci, self.rms = 90.0, 1.0; self.uyarilar.append("yer-duzlemi normali yok -> oblik varsayildi")
        if Z.get("nadir_kural", "ve") in ("aci", "yumusak"):
            # aci birincil ayirici (2026 1.2-5.9 derece, ot4 19-69 derece); rms yalniz duzlem uydurmasi
            # tamamen bozuldugunda veto eder. v1'in rms_esik=0.08'i bicak sirtiydi (clip5: rms 0.082 -> Z 19.6 m).
            self.nadir = self.aci < Z["aci_esik"] and self.rms < float(Z.get("rms_tavan", 1e9))
        else:
            self.nadir = self.aci < Z["aci_esik"] and self.rms < Z["rms_esik"]
        # ── v4 (c): doyan olcek rampasi A_hat (GPS'siz) — kuyruk donmesinden ONCE (kuyruk_rampa_bagli icin) ──
        self._rampa_kur(dp, gt)
        # ── v4 (d): DONME kuyruktan (son kuyruk_n kare), OLCEK+OTELEME akilli pencereden ──
        self.kuyruk = False; self.artik_kuyruk = np.nan; self.kuyruk_dpsi = np.nan
        R0s, s0s, t0s = self.R, self.s, self.t          # akilli-pencere recetesi (Z zinciri icin yedek)
        if XY.get("kuyruk_donme") and not (XY.get("kuyruk_rampa_bagli") and not self._rampa_aktif):
            try:
                a = max(0, n - int(XY.get("kuyruk_n", 100)))
                if n - a >= 40:
                    gk = gt[a:n].copy(); gk[:, 2] *= -1
                    Rt, _, _ = umeyama(dp[a:n], gk)
                    if XY.get("kuyruk_mod", "xy") == "yaw":
                        # yalniz yaw farki: R_yeni = Rz(dpsi) @ R_akilli -> hizalanmis z (Z kaynagi B) HIC DEGISMEZ
                        Rrel = Rt @ self.R.T; dpsi = float(np.arctan2(Rrel[1, 0], Rrel[0, 0]))
                        c_, s_ = np.cos(dpsi), np.sin(dpsi)
                        Rt = np.array([[c_, -s_, 0.0], [s_, c_, 0.0], [0.0, 0.0, 1.0]]) @ self.R
                        self.kuyruk_dpsi = float(np.degrees(dpsi))
                    idx = np.arange(self.pencere[0], self.pencere[1])
                    Sm = dp[idx] - dp[idx].mean(0); gw = gt[idx].copy(); gw[:, 2] *= -1; Dm = gw - gw.mean(0)
                    s2 = float((Dm * (Rt @ Sm.T).T).sum() / max((Sm ** 2).sum(), 1e-12))
                    t2 = gw.mean(0) - s2 * (Rt @ dp[idx].mean(0))
                    if np.isfinite(s2) and s2 > 0:
                        self.R, self.s, self.t = Rt, s2, t2
                        # NOT: koruma kapisi (self.artik) BILEREK akilli-pencere artiginda birakilir; kuyruk donmesi
                        # kalibrasyon penceresine kasten daha kotu oturur (amac post-450 yaw'i) — video-GPS uyumsuzlugu DEGIL.
                        # artik_kuyruk YALNIZ loglanir (adim2 §6.3'un "2x artik ise iptal" kurali UYGULANMAZ: en kotu hucre 2.71 -> 5.62).
                        self.artik_kuyruk = float(np.linalg.norm((s2 * (Rt @ dp[idx].T)).T + t2 - gw, axis=1).mean())
                        self.kuyruk = True
            except Exception as e:
                self.uyarilar.append(f"kuyruk donmesi hesaplanamadi: {e}")
        # Z zinciri (B kaynagi) recetesi: kuyruk_mod="xy" ise akilli-pencere recetesi ("tam" ot4 Z'sini 2.09 -> 8.23 bozuyor)
        self.Rz, self.sz, self.tz = (R0s, s0s, t0s) if (self.kuyruk and XY.get("kuyruk_mod", "xy") == "xy") else (self.R, self.s, self.t)
        self.olcek_k, self.olcek_t = 1.0, 0.0
        if XY.get("olcek_egilimi"):
            try:
                ufuk = n + (P["toplam_kare"] - n) / 2.0
                kk, t_ist, g = olcek_carpani(dp[self.pencere[0]:], gt[self.pencere[0]:], ufuk - self.pencere[0])   # aligner.olcek_carpani (T0 4, T1 12, doyum .25 modül sabitleri)
                self.olcek_k, self.olcek_t = kk, t_ist
                if abs(kk - 1.0) > 1e-6:
                    p_ref = dp[-1]; self.t = self.t + (self.s - self.s * kk) * (self.R @ p_ref); self.s = self.s * kk
                    if self.Rz is self.R: self.Rz, self.sz, self.tz = self.R, self.s, self.t
            except Exception as e:
                self.uyarilar.append(f"olcek egilimi hesaplanamadi: {e}")
        # koruma: kalibrasyon guven kapisi
        self.saglik = True
        if K["acik"]:
            if not (np.isfinite(self.artik) and self.artik <= K["artik_esik_m"]): self.saglik = False; self.uyarilar.append(f"hizalama artigi {self.artik:.1f} m > {K['artik_esik_m']}")
            if np.isfinite(self.artik) and self.yol > 0 and self.artik / self.yol > K.get("artik_oran_esik", 1.0): self.saglik = False; self.uyarilar.append(f"hizalama artigi/yol {self.artik/self.yol*100:.1f}% > {100*K['artik_oran_esik']:.0f}% (video-GPS uyumsuz?)")
            if not (K["s_min"] <= self.s <= K["s_max"]): self.saglik = False; self.uyarilar.append(f"olcek s={self.s:.1f} sinir disi")
            if self.yol < K["yol_min_m"]: self.saglik = False; self.uyarilar.append(f"kalibrasyon yolu {self.yol:.0f} m < {K['yol_min_m']}")
        self.tutucu = not self.saglik
        al = np.nan_to_num(apply_recipe(dp, self.R, self.s, self.t), nan=0.0)
        # v4 (e): Z zinciri (B kaynagi) kendi recetesinden
        alz = al if (self.Rz is self.R and self.sz == self.s) else np.nan_to_num(apply_recipe(dp, self.Rz, self.sz, self.tz), nan=0.0)
        LN = int(Z.get("lev_n", 10)); gz = gt[:, 2]; self.gz_kal = float(np.median(gz[-LN:]))
        PZ = dict(Z)
        if not self.nadir: PZ.update(Z["oblik_olay"])
        self.ema_takvim = PZ["ema_takvim"]; self.guc = PZ["guc"]; self.taban = PZ["taban"]
        # kaynak seviyeleri
        zB = alz[:, 2] - alz[0, 2]; self.levB = float(np.median(zB[-LN:]) - self.gz_kal)   # v4 (e): alz
        dm = np.array(self._dmed); dmv = dm[np.isfinite(dm)]; self.dm_son = float(dmv[-1]) if len(dmv) else np.nan
        zC = -self.s * dm; zCv = zC[np.isfinite(zC)]; self.levC = float(np.median(zCv[-LN:]) - self.gz_kal) if len(zCv) >= 10 else None
        if self.levC is None: self.uyarilar.append("d_med yok -> C kapali")
        mk = lambda tip: KalmanKaynak(tip, PZ["r"], p_ab=PZ["p_ab"], q_c=PZ["q_c"], q_ab=PZ["q_ab"], p_k=PZ["p_k"], z_kal=self.gz_kal)
        kb_tip = Z.get("kB_tip", "egim") if self.nadir else Z.get("kB_tip_oblik", Z.get("kB_tip", "egim"))
        self.kA = mk(Z.get("kA_tip", "olcek")); self.kB = mk(kb_tip); self.kC = mk(Z.get("kC_tip", "egim"))
        w = np.array(Z["w_nadir"] if self.nadir else Z["w_oblik"], float)
        for j, ad in enumerate("ABC"):
            if ad not in Z["kaynaklar"]: w[j] = 0.0
        self.w = w / max(w.sum(), 1e-9)
        # XY-EKF
        # demir: 'gt' = kalibrasyon sonu GT'ye demirle (RGB FINAL-11); 'al' = hizalanmis DPVO konumundan devam (uretim termal, ofset yok)
        x0 = gt[-1, :2] if XY.get("demir", "al") == "gt" else al[-1, :2]
        self.x = np.array([x0[0], x0[1], 0.0, 0.0])
        self.Pxy = np.diag([0.05 ** 2, 0.05 ** 2, np.radians(XY["p0_th_deg"]) ** 2, XY["p0_lam"] ** 2])
        self.Q = np.diag([XY["q_p"] ** 2, XY["q_p"] ** 2, XY["q_th"] ** 2, XY["q_lam"] ** 2]); self.H = np.zeros((2, 4)); self.H[0, 0] = self.H[1, 1] = 1.0
        self.al_onceki = al[-1].copy(); self.alz_onceki = alz[-1].copy(); self.d_onceki = np.zeros(2); self.zB_onceki = float(zB[-1]); self.p0 = gt[-1, :2].copy()
        self._adimlar = list(np.linalg.norm(np.diff(al[-XY['hiz_pencere']-1:, :2], axis=0), axis=1)); self._borc = np.zeros(2); self._durma = 0
        self.olay_k = 0; self.kalibre = True; self.est_onceki = None; self._son_olay_kare = -10 ** 9; self._patlama_sayac = 0
        self._dm_pencere = [x for x in list(dm[-Z["dm_med"]:]) if np.isfinite(x)]
        # koruma durumu
        self._son_gt = gt[-1].copy(); self._son_gt_kare = self.k
        tail = gt[-K["tutucu_hiz_pencere"]:, :2]
        self._tutucu_hiz = (tail[-1] - tail[0]) / max(len(tail) - 1, 1) if len(tail) >= 2 else np.zeros(2)
        hn = np.linalg.norm(self._tutucu_hiz)
        if hn > K["tutucu_hiz_maks"]: self._tutucu_hiz *= K["tutucu_hiz_maks"] / hn
        self._son_cikti = gt[-1].copy(); self._zA_demir = None                # F_z
        # --- F_z: OLCEK-SERBEST zemin seviyesi (taban). a_i = f*|dGT_xy_i|/px_i  -> irtifa; taban = medyan(z_gt + a) ---
        self._taban = None; self._L_demir = None; self._FOC = float(P["z"].get("focal", 731.7965))
        try:
            pk = np.asarray(getattr(self, "_px_kal", []), float)
            if len(pk) >= len(gt):
                mh = np.r_[np.nan, np.linalg.norm(np.diff(gt[:, :2], axis=0), axis=1)]
                aa = np.full(len(gt), np.nan); g_ = np.isfinite(pk[:len(gt)]) & np.isfinite(mh) & (pk[:len(gt)] > 1.0) & (mh > 0.08)
                aa[g_] = self._FOC * mh[g_] / pk[:len(gt)][g_]
                if np.isfinite(aa).sum() >= 30:
                    self._taban = float(np.nanmedian(gt[:, 2] + aa))
                    self.uyarilar.append(f"F_z: taban(zemin z)={self._taban:.2f}, GSD irtifa medyani={np.nanmedian(aa):.1f} m")
        except Exception as e_:
            self.uyarilar.append(f"F_z taban hesaplanamadi: {e_}")
        # --- F_z: L'nin KARE BASINA SISTEMATIK YANLILIGI (beta) kalibrasyonda olculur (OLCEK-SERBEST) ---
        #   fizik: nadire bakan kamerada goruntu olcegi ~ 1/irtifa  ->  log h(i) = log h(0) - L(i)
        #   gerceklikte ORB ikili-mesafe orani kucuk bir carpimsal yanlilik tasiyor (2026'da medyan 0.99977)
        #   -> 2000 karede exp(+0.46) yapay "tirmanis". Yanliligi GT'li pencerede olcup cikariyoruz.
        self._beta = 0.0; self._k_demir = self.k
        try:
            nG = len(gt); Lk = np.asarray(self._L[:nG + 1], float)[1:]          # Lk[j] <-> gt[j]
            if self._taban is not None and len(Lk) == nG:
                hg = self._taban - gt[:, 2]
                g_ = hg > 2.0
                if g_.sum() >= 100:
                    y = np.log(hg[g_]) - np.log(hg[g_][0]) + (Lk[g_] - Lk[g_][0])   # k=1 kalintisi
                    ii = np.arange(nG)[g_].astype(float)
                    A_ = np.c_[np.ones(g_.sum()), ii - ii[0]]
                    co, *_ = np.linalg.lstsq(A_, y, rcond=None)
                    bm = float(K.get("tutucu_z_beta_maks", 2e-3))
                    self._beta = float(np.clip(co[1], -bm, bm))
                    self.uyarilar.append(f"F_z: ORB olcek yanliligi beta={self._beta:+.2e}/kare (kalib artigi std {np.std(y - A_ @ co):.3f})")
        except Exception as e_:
            self.uyarilar.append(f"F_z beta hesaplanamadi: {e_}")
        self._L_demir = float(self._L[len(gt)]) if getattr(self, "_L", None) and len(self._L) > len(gt) else 0.0
        if K.get("tutucu_w_A") and not self.saglik and "A" in Z["kaynaklar"]:
            self.w = np.array([1.0, 0.0, 0.0]); self.uyarilar.append("F_z: kalibrasyon guvensiz -> Z agirliklari A kaynagina (1,0,0)")
        self._hC, self._hA = [], []; self.cokme_kare = None; self.h0_kal = float(self.s * np.nanmedian(dm[-30:])) if np.isfinite(dm[-30:]).any() else np.nan
        self.bilgi = dict(s=self.s, artik=self.artik, yol=self.yol, pencere=self.pencere, aci=self.aci, rms=self.rms, nadir=self.nadir,
                          olcek_k=self.olcek_k, olcek_t=self.olcek_t, saglik=self.saglik,
                          # v4 ZORUNLU TELEMETRI (recete §5)
                          A=self.rampa_A, A_ham=self.rampa_A_ham, rampa_t=self.rampa_t, rampa_kapi_g=self.rampa_kapi_g,
                          rampa_eg=self.rampa_eg, rampa_mono=self.rampa_mono, kuyruk=self.kuyruk, artik_kuyruk=self.artik_kuyruk,
                          kuyruk_dpsi=self.kuyruk_dpsi, w=tuple(float(x) for x in self.w))


    # ---------- v4: doyan olcek rampasi ----------
    def _rampa_kur(self, dp, gt):
        """A_hat'i GPS'siz kestir (kalibrasyon penceresi yerel-olcek egimi x ufuk) ve rampa durumunu kur."""
        XY = self.P["xy"]; n = len(dp)
        self.rampa_A = 0.0; self.rampa_A_ham = 0.0; self.rampa_t = 0.0; self.rampa_eg = 0.0; self.rampa_mono = 1.0
        self.rampa_kapi_g = 0.0
        self._lr_son = 0.0; self._lr_onceki = 0.0; self._rampa_dondu = False
        self._sz_g = []; self._sz_zref = None; self._sz_dz = 0.0; self._z_onceki = None; self._sz_son = 1.0
        self._rampa_aktif = False
        if XY.get("rampa", "yok") == "yok":
            return
        A_sabit = XY.get("rampa_A")
        if A_sabit is None:
            a0 = int(self.pencere[0]); ufuk = n + (self.P["toplam_kare"] - n) * float(XY.get("rampa_ufuk", 0.5))
            spans = XY.get("rampa_span") or (200,)
            if not isinstance(spans, (list, tuple)): spans = (spans,)
            AL, TL, ML = [], [], []
            for L in spans:
                b0 = max(a0, n - int(L))
                if n - b0 < 90: continue
                try:
                    eg, ti, mr, sd = olcek_egilimi_gen(dp[b0:n], gt[b0:n])
                except Exception:
                    continue
                if ti == 0.0 and eg == 0.0: continue
                AL.append(eg * (ufuk - b0 - mr)); TL.append(ti); ML.append(eg)
                if len(sd) >= 3:
                    d = np.diff(sd); self.rampa_mono = min(self.rampa_mono, float((d > 0).mean()))
            if not AL:
                self.uyarilar.append("rampa: olcek egimi hesaplanamadi -> A=0"); return
            A_ham = float(np.median(AL)); t_ist = float(np.median(TL)); self.rampa_eg = float(np.median(ML))
            self.rampa_A_ham = A_ham; self.rampa_t = t_ist
            # --- A kapisi ---
            kapi = XY.get("rampa_kapi", "t")
            g_t = float(np.clip((t_ist - XY["rampa_T0"]) / max(XY["rampa_T1"] - XY["rampa_T0"], 1e-9), 0.0, 1.0))
            A = float(XY.get("rampa_k", 1.0)) * A_ham
            if kapi == "t":                                      # v1/G — KIRILGAN, KULLANMAYIN (canli: t=2.461 -> Y 10.26 m)
                A = g_t * A
            elif kapi == "isaret":
                pass                                             # isaret: A<=0 zaten rampa_A_min=0 kirpmasiyla elenir
            elif kapi == "isaret_buyukluk":                      # ONERILEN (G'')
                if abs(A_ham) <= float(XY.get("rampa_A_esik", 0.0)): A = 0.0
            elif kapi == "t_ve_isaret":
                if abs(A_ham) <= float(XY.get("rampa_A_esik", 0.0)): A = 0.0
                A = g_t * A
            self.rampa_kapi_g = g_t
            if XY.get("rampa_nadir") and not self.nadir: A = 0.0
            if XY.get("rampa_mono", 0.0) > 0 and self.rampa_mono < XY["rampa_mono"]: A = 0.0
            A = float(np.clip(A, XY.get("rampa_A_min", 0.0), XY.get("rampa_A_maks", 0.15)))
        else:
            A = float(A_sabit); self.rampa_A_ham = A; self.rampa_t = np.nan
        self.rampa_A = A
        self._rampa_aktif = abs(A) > 1e-6

    # ---------- v4: dusey sizinti duzeltmesi ----------
    def _sizinti_duzelt(self, d, dz_xy, zA_ham, zC_ham=None):
        """DPVO'nun hizalanmis 3B yer degistirmesinin buyuklugunu yatay adima geri kazandir:
        |d'| = sqrt(max(|d3B|^2 - dz_ref^2, 0)). dz_ref = gercek dusey hiz kestirimi (mod'a gore)."""
        XY = self.P["xy"]; mod = XY["duzey_sizinti"]
        if XY.get("sizinti_nadir") and not self.nadir:
            return d
        nd = float(np.linalg.norm(d))
        if not np.isfinite(nd) or nd < 1e-6 or not np.isfinite(dz_xy):
            return d
        if mod == "sifir":
            zref = 0.0
        elif mod == "ac":
            z_yeni = None
            if zA_ham is not None and np.isfinite(zA_ham): z_yeni = float(zA_ham)
            elif self.levC is not None and np.isfinite(self.dm_son): z_yeni = float(-self.s * self.dm_son - self.levC)
            if z_yeni is None: return d
            if self._sz_zref is None: self._sz_zref = z_yeni; return d
            a = 2.0 / (float(XY["sizinti_pencere"]) + 1.0)
            self._sz_dz = (1 - a) * self._sz_dz + a * (z_yeni - self._sz_zref); self._sz_zref = z_yeni
            zref = self._sz_dz
        else:                                    # "fuzyon": onceki karenin fuzyon z artimi (yumusatilmis)
            zref = self._sz_dz
        m = np.sqrt(max(nd * nd + dz_xy * dz_xy - zref * zref, 0.0))
        g = m / nd
        lo, hi = XY["sizinti_kirp"]
        g = float(np.clip(g, lo, hi))
        W = int(XY["sizinti_pencere"])
        self._sz_g.append(g); self._sz_g = self._sz_g[-W:]
        gs = float(np.median(self._sz_g)) if XY.get("sizinti_yumusat", "medyan") == "medyan" else float(np.mean(self._sz_g))
        self._sz_son = gs
        return d * gs

    def _sz_fuzyon_guncelle(self, z):
        """'fuzyon' modu icin fuzyon z'sinin yumusatilmis artimi (1 kare gecikmeli)."""
        if self.P["xy"].get("duzey_sizinti") != "fuzyon" or z is None or not np.isfinite(z):
            return
        if self._z_onceki is None: self._z_onceki = float(z); return
        a = 2.0 / (float(self.P["xy"]["sizinti_pencere"]) + 1.0)
        self._sz_dz = (1 - a) * self._sz_dz + a * (float(z) - self._z_onceki); self._z_onceki = float(z)

    def _lam_ramp(self, k):
        XY = self.P["xy"]; u = max(k - self.G + 1, 0)
        return float(np.log1p(self.rampa_A * (1.0 - np.exp(-u / max(float(XY["rampa_tau"]), 1e-6)))))

    # ---------- durma / telafi filtresi ----------
    def _adim_filtre(self, d):
        """DPVO 'durma' (adim ~0 iken son hiz >0) -> son ortalama hizla olu hesap (borc birikir);
        'telafi' (adim > telafi_kat x medyan hiz) -> once biriken borcu dus, kalani medyan hiza kirp (yon korunur)."""
        XY = self.P["xy"]; nd = float(np.linalg.norm(d))
        hizlar = np.array(self._adimlar[-XY["hiz_pencere"]:]) if self._adimlar else np.array([nd])
        med = float(np.median(hizlar)) if len(hizlar) else nd
        yon = np.array([1.0, 0.0])
        if len(self._adimlar) >= 3:
            # son gecerli yon: son 5 adimin vektor ortalamasi (durmada yon bilgisi yok)
            yon = self._yon if hasattr(self, "_yon") else yon
        if med > 0.15 and nd < XY["durma_oran"] * med:
            # DURMA: olu hesap
            d_yeni = yon * med
            if XY.get("telafi_borc"): self._borc = self._borc + d_yeni
            self._durma += 1
            return d_yeni
        if med > 0.1 and nd > XY["telafi_kat"] * med:
            # TELAFI: borcu dus
            d_kalan = d - self._borc if XY.get("telafi_borc") else d
            self._borc = np.zeros(2)
            nk = float(np.linalg.norm(d_kalan))
            if XY.get("telafi_kirp") and nk > XY["telafi_kat"] * med:
                d_kalan = d_kalan / nk * (XY["telafi_kat"] * med)
            self._adimlar.append(float(np.linalg.norm(d_kalan)))
            if np.linalg.norm(d_kalan) > 1e-6: self._yon = 0.7 * yon + 0.3 * d_kalan / np.linalg.norm(d_kalan); self._yon /= np.linalg.norm(self._yon)
            return d_kalan
        # normal adim: borc zamanla unutulur (telafi gelmediyse)
        self._borc *= 0.8
        self._adimlar.append(nd)
        if nd > 1e-6:
            yy = d / nd; self._yon = (0.7 * yon + 0.3 * yy) if hasattr(self, "_yon") else yy; self._yon /= np.linalg.norm(self._yon)
        return d

    # ---------- koruma ----------
    def _kelepce(self, out):
        K = self.P["koruma"]
        if not K["acik"]:
            self._son_cikti = out.copy(); return out
        o = out.copy(); p = self._son_cikti
        for i in range(2):
            o[i] = float(np.clip(o[i], p[i] - K["adim_maks_xy"], p[i] + K["adim_maks_xy"]))
        o[2] = float(np.clip(o[2], p[2] - K["adim_maks_z"], p[2] + K["adim_maks_z"]))
        o[2] = float(np.clip(o[2], self._son_gt[2] - K["z_sapma_maks"], self._son_gt[2] + K["z_sapma_maks"]))
        self._son_cikti = o.copy(); return o

    # ---------- F_z: OLCEK-SERBEST, SON GT'YE DEMIRLI irtifa kaynagi (A2) ----------
    def _z_A2(self, beta=False):
        """h(i) = h(demir) * exp(-(L(i)-L(demir))),  z = taban - h.
        L: ORB ikili-mesafe oraninin kumulatif logu -> goruntu zoom'u; nadire bakan kamerada olcek ~ 1/irtifa,
        yani k=1 FIZIKSEL ONCUL (fit YOK). Kalibrasyon benzerlik donusumu bozuksa bile gecerli: L ne DPVO
        olceginden ne de hizalamadan etkilenir. taban ve h(demir) yalniz health=1 GT'den gelir -> nedensel."""
        if self._taban is None or self._L_demir is None: return None
        h0 = self._taban - float(self._son_gt[2])
        if not np.isfinite(h0) or h0 <= 2.0: return None
        kk = float(self.P["koruma"].get("tutucu_z_k", 1.0))
        dL = (self._L[-1] - self._L_demir)
        if beta or self.P["koruma"].get("tutucu_z", "") in ("A5", "A5_A"):
            dL = dL - self._beta * (self.k - self._k_demir)          # F_z: olculen yanliligi cikar
        h = h0 * float(np.exp(-kk * dL))
        h = float(np.clip(h, 2.0, 300.0))
        return float(self._taban - h)

    # ---------- F_z: tutucu modda Z ----------
    def _tutucu_z(self, z_fuz, est):
        """Tutucu (kalibrasyon guvensiz) modda Z cikisini sec. TABAN: son GT'ye DON (donuk)."""
        mod = self.P["koruma"].get("tutucu_z", "son_gt")
        zA = est[0]
        if mod == "son_gt" or (zA is None and mod in ("A", "A_delta")):
            return float(self._son_gt[2])
        if mod == "A":
            return float(zA)
        if mod in ("med3", "med3b"):
            # SAGLAM FUZYON: uc bagimsiz kestirimin MEDYANI. Tek bir kaynak patlasa bile sonucu suruklyemez.
            #   A  = uretim ORB olcek kilidi (k uyarlamali, GSD hakemli)
            #   A2 = son GT'ye demirli saf geometri (k=1, yanlilik duzeltmesi YOK)
            #   A5 = ayni + kalibrasyonda olculen kare-basina ORB yanliligi cikarilmis   [med3]
            #   A_d= A'nin yalniz artimi, son GT'ye demirli                              [med3b]
            c = []
            if zA is not None: c.append(float(zA))
            z2 = self._z_A2(beta=False)
            if z2 is not None: c.append(z2)
            if mod == "med3":
                z5 = self._z_A2(beta=True)
                if z5 is not None: c.append(z5)
            else:
                if zA is not None:
                    if getattr(self, "_zA_demir", None) is None: self._zA_demir = float(zA)
                    c.append(float(self._son_gt[2] + (zA - self._zA_demir)))
            if len(c) >= 2: return float(np.median(c))
            return float(c[0]) if c else float(self._son_gt[2])
        if mod in ("A2", "A2_A", "A5", "A5_A"):
            z2 = self._z_A2()
            if mod in ("A2", "A5"):
                return z2 if z2 is not None else (float(zA) if zA is not None else float(self._son_gt[2]))
            if z2 is not None and zA is not None: return 0.5 * (z2 + float(zA))
            return z2 if z2 is not None else (float(zA) if zA is not None else float(self._son_gt[2]))
        if mod == "A_delta":
            # son GT anindaki A degerine gore SADECE ARTIM: mutlak ofset hatasi tasinmaz, gecis sicramasi olmaz
            if getattr(self, "_zA_demir", None) is None:
                self._zA_demir = float(zA)
            return float(self._son_gt[2] + (zA - self._zA_demir))
        if mod == "fuz":
            return float(z_fuz)
        if mod == "A_fuz":
            return float(zA) if zA is not None else float(z_fuz)
        return float(self._son_gt[2])

    # ---------- kare ----------
    def adim(self, dpvo_xyz, d_med=None, n_cam=None, plan_rms=None, zA_ham=None, gt_xyz=None, health=0, r_orb=None, px_orb=None, kare_tekrar=False):
        P = self.P; Z = P["z"]; XY = P["xy"]; K = P["koruma"]
        if kare_tekrar and self.kalibre and not (health == 1 and gt_xyz is not None):
            # VIDEO TEKRAR KARESI: DPVO'ya verilmedi -> durum ilerletme; cikti = son cikti + son artim (olu hesap). Sonraki gercek karede
            # DPVO artimi boslugu kapsar (tstamp ile), EKF o artimi tek seferde isler (ekstrapolasyon yalniz cikti icindir).
            self._tekrar_n = getattr(self, "_tekrar_n", 0) + 1; self._bosluk = getattr(self, "_bosluk", 0) + 1
            hiz = np.exp(self.x[3] + getattr(self, "_lr_son", 0.0)) * (R2(self.x[2]) @ self.d_onceki)   # v4: rampa="onsel" icin
            out = np.array(self.son, float) + np.array([hiz[0], hiz[1], 0.0]) if self.son is not None else np.zeros(3)
            self.k += 1; return tuple(float(v) for v in out)
        if not hasattr(self, "_L"): self._L = [0.0]
        rr = float(r_orb) if (r_orb is not None and np.isfinite(r_orb)) else 1.0
        self._L.append(self._L[-1] + float(np.log(np.clip(rr, 0.9, 1.1))))     # ORB goruntu zoom'u (kumulatif log-oran; k=1 saf geometri)
        if health == 1 and gt_xyz is not None and not self.kalibre:
            if dpvo_xyz is not None: self._kalib_ekle(dpvo_xyz, gt_xyz, d_med, n_cam, plan_rms, px_orb)   # F_z
            self.son = tuple(float(v) for v in gt_xyz); self.k += 1; return self.son
        if not self.kalibre:
            if len(self._dp) < 50: self.k += 1; return self.son
            self._kalib_bitir()
        # --- DPVO artimi ---
        if dpvo_xyz is None or not np.all(np.isfinite(dpvo_xyz)): al = self.al_onceki.copy(); alz = self.alz_onceki.copy()
        else:
            al = apply_recipe(np.asarray(dpvo_xyz, float)[None], self.R, self.s, self.t)[0]
            alz = al if self.Rz is self.R else apply_recipe(np.asarray(dpvo_xyz, float)[None], self.Rz, self.sz, self.tz)[0]   # v4 (f.1)
        d = al[:2] - self.al_onceki[:2]; nd = float(np.linalg.norm(d))
        bosluk = getattr(self, "_bosluk", 0); tavan = XY["adim_tavan"] * (1 + bosluk)     # tekrar kareler atlandiysa devam adimi (bosluk+1) kare kapsar
        if not np.isfinite(nd) or nd > tavan: d = self.d_onceki * (1 + bosluk)
        dz_xy = float(al[2] - self.al_onceki[2]); dz_xy = dz_xy if (np.isfinite(dz_xy) and abs(dz_xy) < tavan) else 0.0   # v4 (f.2): sizinti icin AL'dan
        dz = float(alz[2] - self.alz_onceki[2]); dz = dz if np.isfinite(dz) and abs(dz) < tavan else 0.0                  # v4 (f.2): Z kaynagi ALZ'den
        self._bosluk = 0
        self.al_onceki = al; self.alz_onceki = alz
        if XY.get("adim_filtre", False):
            d = self._adim_filtre(d)
        if XY.get("duzey_sizinti", "yok") != "yok":
            d = self._sizinti_duzelt(d, dz_xy, zA_ham, zC_ham=None)      # v4 (f.3)
        self.d_onceki = d
        # --- derinlik gecikmesi duzeltmesi (ANA testi: k=(h/(s*d_med))^alfa; nedensel h = ORB kilidi (A) irtifasi) ---
        k_der = 1.0
        if XY.get("derinlik_alfa", 0.0) > 0 and np.isfinite(self.dm_son) and np.isfinite(getattr(self, "h0_kal", np.nan)):
            hC = self.s * self.dm_son
            hA = (self.h0_kal - (float(zA_ham) - self.gz_kal)) if (zA_ham is not None and np.isfinite(zA_ham)) else np.nan
            if np.isfinite(hA) and hA > 5 and hC > 5:
                k_der = float(np.clip((hA / hC) ** XY["derinlik_alfa"], 0.7, 1.4))
        d = d * k_der
        # --- XY-EKF tahmin ---
        self.x[3] += XY["lam_drift"]
        # v4 (f.4): doyan olcek rampasi
        if self._rampa_aktif and not self._rampa_dondu:
            lrt = self._lam_ramp(self.k)
            if XY["rampa"] == "drift":
                self.x[3] += (lrt - self._lr_onceki); self._lr_onceki = lrt
            else:                                     # "onsel": toplamsal (durum bozulmaz)
                self._lr_son = lrt
        th, lam = self.x[2], self.x[3] + self._lr_son; Rt = R2(th); es = np.exp(lam)
        self.x[:2] = self.x[:2] + es * (Rt @ d)
        F = np.eye(4); F[:2, 2] = es * (dR2(th) @ d); F[:2, 3] = es * (Rt @ d); self.Pxy = F @ self.Pxy @ F.T + self.Q
        # --- Z kaynaklari ---
        self.zB_onceki += dz; zB = self.zB_onceki - self.levB
        if d_med is not None and np.isfinite(d_med):
            self._dm_pencere = (self._dm_pencere + [float(d_med)])[-Z["dm_med"]:]; self.dm_son = float(np.median(self._dm_pencere))
        sC = self.s * (es if Z["c_lam"] else 1.0)
        zC = (-sC * self.dm_son - self.levC) if (self.levC is not None and np.isfinite(self.dm_son)) else None
        zA = float(zA_ham) if (zA_ham is not None and np.isfinite(zA_ham)) else None
        for kk in (self.kA, self.kB, self.kC): kk.tahmin()
        dxy = self.x[:2] - self.p0
        est = [self.kA.duzelt(zA, dxy) if zA is not None else None, self.kB.duzelt(zB, dxy), self.kC.duzelt(zC, dxy) if zC is not None else None]
        # --- DPVO olcek cokmesi tespiti ---
        if K.get("cokme_acik") and self.cokme_kare is None and np.isfinite(self.dm_son) and np.isfinite(self.h0_kal):
            hC = self.s * self.dm_son                                   # DPVO'ya gore irtifa (metrik)
            hA = (self.h0_kal - (float(zA_ham) - self.gz_kal)) if (zA_ham is not None and np.isfinite(zA_ham)) else np.nan   # ORB kilidine gore irtifa
            self._hC.append(hC); self._hA.append(hA)
            if not hasattr(self, "_q"): self._q = []
            self._q.append((float(np.linalg.norm(d)) / float(px_orb)) if (px_orb is not None and np.isfinite(px_orb) and px_orb > 2.0 and np.isfinite(np.linalg.norm(d))) else np.nan)
            W_ = K["cokme_pencere"]
            if len(self._hC) > 2 * W_:
                c0 = np.nanmedian(self._hC[-2 * W_:-W_]); c1 = np.nanmedian(self._hC[-5:])
                a0 = np.nanmedian(self._hA[-2 * W_:-W_]); a1 = np.nanmedian(self._hA[-5:])
                # ASIL kriter: metrik DPVO adimi / ORB piksel hizi (= irtifa/(f)) 30 karede x3'ten fazla degisirse -> olcek cokmesi
                # (gercek irtifa degisimi ~x2'yi asmaz; oblik d_med dalgalanmasi (ot4 873) XY'yi bozmaz -> q sabit kalir)
                q = np.array(self._q); q0 = np.nanmedian(q[-2 * W_:-W_]) if np.isfinite(q[-2 * W_:-W_]).sum() >= 5 else np.nan
                q1 = np.nanmedian(q[-8:]) if np.isfinite(q[-8:]).sum() >= 4 else np.nan
                q_cokme = np.isfinite(q0) and np.isfinite(q1) and q0 > 0 and (q1 / q0 < 1 / K["cokme_q_oran"] or q1 / q0 > K["cokme_q_oran"])
                if q_cokme:
                    if True:
                        self.cokme_kare = self.k; self.tutucu = True; self.uyarilar.append(f"DPVO olcek cokmesi kare {self.k}: adim/px {q0:.4f}->{q1:.4f}, hC {c0:.1f}->{c1:.1f} -> tutucu mod")
                        self._son_gt = np.array(self.son, float) if self.son is not None else self._son_gt; self._son_gt_kare = self.k
                        self.w = np.array([0.5, 0.5, 0.0]) if self.w[0] + self.w[1] > 0 else self.w
        if health == 1 and gt_xyz is not None:
            gt = np.asarray(gt_xyz, float)
            if self.k - self._son_olay_kare >= Z["w_min_ara"]: self._patlama_sayac = 0
            # v4 (g): ERKEN GPS OLAYI KAPISI — kalibrasyondan olay_min_kare gecmeden Z kaynaklari/agirliklari guncellenmez
            z_guncelle = self._patlama_sayac < Z["patlama_maks"] and (self.k - self.G) >= Z.get("olay_min_kare", 0)
            self._patlama_sayac += 1; self._son_olay_kare = self.k
            rt = XY.get("r_takvim"); rr = rt[min(self.olay_k, len(rt) - 1)] if rt else 0.3; Rm = np.eye(2) * rr ** 2
            y = gt[:2] - self.x[:2]; S = self.H @ self.Pxy @ self.H.T + Rm; Kg = self.Pxy @ self.H.T @ np.linalg.inv(S)
            self.x = self.x + Kg @ y; self.Pxy = (np.eye(4) - Kg @ self.H) @ self.Pxy; self.x[:2] = gt[:2]
            if z_guncelle and self.est_onceki is not None:
                e = np.array([abs(v - gt[2]) if v is not None else 50.0 for v in self.est_onceki]); ww = (1 / (e + self.taban)) ** self.guc
                ww[self.w == 0] = 0.0
                if ww.sum() > 0:
                    ww /= ww.sum(); ema = self.ema_takvim[min(self.olay_k, len(self.ema_takvim) - 1)]; self.w = self.w + ema * (ww - self.w)
            dxy_gt = gt[:2] - self.p0
            if z_guncelle:
                if zA is not None: self.kA.olcum(zA, dxy_gt, gt[2])
                self.kB.olcum(zB, dxy_gt, gt[2])
                if zC is not None: self.kC.olcum(zC, dxy_gt, gt[2])
            self._sz_fuzyon_guncelle(gt[2])                                              # v4 (f.5)
            if XY.get("rampa_olay") == "dondur": self._rampa_dondu = True                # v4: olay lam'i olctu -> rampa artimi durur
            self.olay_k += 1; self.est_onceki = est; self.son = tuple(float(v) for v in gt); self.k += 1
            # koruma: yeniden demir
            self._son_gt = gt.copy(); self._son_gt_kare = self.k; self._son_cikti = gt.copy(); self._tutucu_kare = 0
            self._zA_demir = float(est[0]) if est[0] is not None else None     # F_z: A_delta demiri GPS olayinda tazelenir
            self._L_demir = float(self._L[-1])                                  # F_z: A2 demiri GPS olayinda tazelenir
            self._k_demir = self.k                                              # F_z
            return self.son
        # --- fuzyon ---
        w = self.w.copy(); vals = np.array([v if v is not None else 0.0 for v in est]); w[[v is None for v in est]] = 0.0
        z = float(w @ vals / w.sum()) if w.sum() > 0 else (self.son[2] if self.son else 0.0)
        self.est_onceki = est
        out = np.array([self.x[0], self.x[1], z], float)
        if self.tutucu:
            if K.get("tutucu_xy", True):
                n_since = self.k - self._son_gt_kare
                hiz = self._tutucu_hiz if n_since <= K["tutucu_hiz_sure"] else np.zeros(2)
                out = np.array([self._son_gt[0] + hiz[0] * min(n_since, K["tutucu_hiz_sure"]), self._son_gt[1] + hiz[1] * min(n_since, K["tutucu_hiz_sure"]), self._tutucu_z(z, est)])
                self.x[:2] = out[:2]
            else:
                out = np.array([self.x[0], self.x[1], self._tutucu_z(z, est)])      # YARI TUTUCU: XY DPVO (kelepceli), Z = koruma.tutucu_z
        out = self._kelepce(out)
        self._sz_fuzyon_guncelle(out[2])                                                 # v4 (f.5)
        self.son = tuple(float(v) for v in out); self.k += 1
        return self.son


# ══════════════════════════════════════════════════════════════════════════════════════
#  v5 (2026-09-19) — folder4 SAGLAMLASTIRMASI.  Hepsi VARSAYILAN KAPALI:
#  v4 JSON ile yuklenince cikti v4 ile BIT-BIT AYNI (esdegerlik testi: NIHAI/kos.py).
#
#  C   (koruma.c_*)   : kalibrasyon kapisi olcek-serbest (yerel olcek dagilimi) + fiziksel
#                       irtifa kapisi (s*d_med) + kapi dustugunde XY'yi son GT'ye demirle.
#  E   (xy.adim_filtre + E.*) : NEDENSEL "DPVO sagliksiz" dedektoru
#                       (plan_rms > e_rms) VEYA (|log(med d_med[t-4:t] / med d_med[t-25:t-5])| > e_dm)
#                       -> bayrakli karede DPVO artimi ATILIR, son saglikli yon x medyan hizla olu hesap.
#                       OLCULEN ATESLEME: 2026 %0.1, ot4 %0.1 (ISPATLI NO-OP), folder4 %26.8.
#  E2  (E.olay_sus)   : GPS olayindan sonra N kare dedektoru sustur (taze demirde ham DPVO daha iyi).
#  UY2 (uy2.*)        : kalibrasyon kapisindan BAGIMSIZ olcek suruklenmesi duzeltmesi;
#                       r = (h_A/d_med)/s, olu bant SAGLIKLI VERI ISTATISTIGINDEN (0.35).
#  F   (koruma.tutucu_z): tutucu modda Z, olcek-serbest ORB irtifa kaynaklarinin medyani.
#
#  DIKKAT: koruma.c_uy ile uy2 AYNI kanala (_lr_son) yazar -> c_uy=False SART.
#  Olculen (GPS geldigindeki sicrama |pred[olay-1]-gt[olay]|, folder4 canli traje):
#      @1050 176.1 -> 43.4 m ,  @1780 417.1 -> 193.4 m
#  Regresyon (2026+ot4, 22 takvim x 3 eksen = 66 hucre): en kotu +0.043 m.
# ══════════════════════════════════════════════════════════════════════════════════════

def yaw_duzeltmesi(al_xy, gt_xy, min_adim=0.05):
    """OLCEK-SERBEST yaw: hizalanmis DPVO ile GT kare-arasi artimlarinin ACI farkinin,
    GT adim uzunluguyla agirlikli DAIRESEL ORTALAMASI. Buyuklukler KULLANILMAZ -> olcek suruklenmesinden
    etkilenmez (Umeyama donmesi ise buyukluklerle agirlikli, bu yuzden olcek suruklenmesinde sapiyor)."""
    da = np.diff(np.asarray(al_xy, float), axis=0); dg = np.diff(np.asarray(gt_xy, float), axis=0)
    a = np.arctan2(da[:, 1], da[:, 0]); b = np.arctan2(dg[:, 1], dg[:, 0])
    na = np.linalg.norm(da, axis=1); ng = np.linalg.norm(dg, axis=1)
    m = (na > min_adim) & (ng > min_adim) & np.isfinite(a) & np.isfinite(b)
    if m.sum() < 20: return 0.0, 0
    w = ng[m]; d = b[m] - a[m]
    return float(np.arctan2(float((w * np.sin(d)).sum()), float((w * np.cos(d)).sum()))), int(m.sum())

C_VARSAYILAN = dict(c_acik=False, c_L=50, c_adim=25, c_yol_min=5.0, c_dagilim_esik=3.0,
                    c_irtifa_ara=(5.0, 500.0), c_artik_oran_esik=0.02, c_mutlak_kapilar=False,
                    c_olcek="yok", c_olcek_kirp=(0.25, 4.0), c_demir_gt=False, c_z="v4",
                    c_surekli=False, c_guven_q=10.0, c_dagilim_p=(10, 90),
                    # --- uyarlanir (irtifa/derinlik) XY olcegi: lam_t = beta*log(clip(r_t/r_G)) ---
                    c_uy=False, c_uy_W=15, c_uy_beta=1.0, c_uy_kirp=(0.3, 3.0), c_uy_norm=True,
                    # --- fiziksel hiz siniri: adim tavani = p99(kalibrasyon GT adimi) * kat (0 = kapali) ---
                    c_hiz_kat=0.0, c_hiz_mod="tavan", c_hiz_kosullu=False,
                    # --- kapi dustugunde kuyruk (son-100-kare) donmesini iptal et ---
                    c_kuyruk_kapat=False,
                    # --- OLCEK-SERBEST yaw duzeltmesi (yalniz YONLERDEN) ---
                    c_yaw=False, c_yaw_min_adim=0.05, c_yaw_maks_deg=45.0)


def yerel_olcek_istatistigi(dp, gt, L=50, adim=25, yol_min=5.0, pct=(10, 90)):
    """Kalibrasyon penceresi ICINDEKI yerel Umeyama olceklerinin dagilimi (NEDENSEL: yalniz 0..G)."""
    dp = np.asarray(dp, float); gt = np.asarray(gt, float); n = len(dp)
    ss = []
    for a in range(0, n - L + 1, max(adim, 1)):
        b = a + L
        if np.linalg.norm(np.diff(gt[a:b, :2], axis=0), axis=1).sum() < yol_min: continue
        g = gt[a:b].copy(); g[:, 2] *= -1
        try: _, s_, _ = umeyama(dp[a:b], g)
        except Exception: continue
        if np.isfinite(s_) and s_ > 0: ss.append(float(s_))
    if len(ss) < 4:
        return dict(n=len(ss), med=np.nan, geo=np.nan, dagilim=np.nan, ss=np.array(ss))
    ss = np.array(ss)
    lo, hi = np.percentile(ss, pct[0]), np.percentile(ss, pct[1])
    return dict(n=len(ss), med=float(np.median(ss)), geo=float(np.exp(np.mean(np.log(ss)))),
                dagilim=float(hi / max(lo, 1e-12)), ss=ss)


class KestiriciC(_TK_TABAN):
    def _kalib_bitir(self):
        # 2026-09-20 DUZELTME: eskiden "from kestirici_v4 import kalibre as _kalibre" yaziyordu.
        # kestirici_v4 gelistirme klasorunde kalmis (~/Desktop/gps/thermal_gps/work_0919_f4/),
        # repoya hic girmemis ve sys.path'e de eklenmiyor -> bu satir ImportError atiyordu.
        # Istisna ilk health=0 karesinde (kare ~450) firliyor, ustteki genis except yutuyordu:
        # v5'in TUM korumalari (c_demir_gt, c_irtifa_ara, c_hiz_tavan, c_dagilim) HIC calismadi.
        # 18-19 Eylul termal oturumundaki 425 m drift'in kok nedeni budur.
        # kalibre() ZATEN BU DOSYADA (satir 33) ve kaynaktakiyle BIREBIR AYNI kod (diff ile
        # dogrulandi) -> dis modul gereksiz, modul-ici fonksiyon kullaniliyor. Hesap DEGISMEDI.
        _kalibre = kalibre
        dp = np.array(self._dp); gt = np.array(self._gt)
        dmed = np.array(self._dmed, float)
        K = self.P["koruma"]
        C = {k: K.get(k, v) for k, v in C_VARSAYILAN.items()}
        self.C = C
        # --- ON HESAP: kapi gostergeleri super()'den ONCE (kuyruk_donme karari icin gerekli) ---
        st = yerel_olcek_istatistigi(dp, gt, C["c_L"], C["c_adim"], C["c_yol_min"], tuple(C["c_dagilim_p"]))
        self.c_ist = st
        _, _, _, _, _artik, _yol = _kalibre(dp, gt)
        self.c_bozuk_on = bool(C["c_acik"] and K["acik"] and (
            (np.isfinite(st["dagilim"]) and st["dagilim"] > C["c_dagilim_esik"]) or
            (np.isfinite(_artik) and _yol > 0 and _artik / _yol > C["c_artik_oran_esik"])))
        if C["c_acik"] and C["c_kuyruk_kapat"] and self.c_bozuk_on:
            self.P["xy"] = dict(self.P["xy"]); self.P["xy"]["kuyruk_donme"] = False
        # --- FIZIKSEL HIZ SINIRI: kalibrasyon GT hizindan (metre/kare), olcekten BAGIMSIZ ---
        self.c_hiz_tavan = np.nan
        if C["c_acik"] and C["c_hiz_kat"] > 0 and (self.c_bozuk_on or not C["c_hiz_kosullu"]):
            gd = np.linalg.norm(np.diff(gt[:, :2], axis=0), axis=1)
            gd = gd[np.isfinite(gd)]
            if len(gd) >= 50:
                self.c_hiz_tavan = float(np.percentile(gd, 99) * C["c_hiz_kat"])
        super()._kalib_bitir()
        if np.isfinite(self.c_hiz_tavan):
            self.P["xy"] = dict(self.P["xy"]); self.P["koruma"] = dict(self.P["koruma"])
            self.P["xy"]["adim_tavan"] = self.c_hiz_tavan
            self.P["koruma"]["adim_maks_xy"] = self.c_hiz_tavan
            if C["c_hiz_mod"] == "kirp": self.P["xy"]["adim_filtre"] = True
            K = self.P["koruma"]
        dmv = dmed[np.isfinite(dmed)]
        self.c_irtifa = float(self.s * np.median(dmv[-60:])) if len(dmv) >= 10 else np.nan
        self.bilgi.update(c_dagilim=st["dagilim"], c_s_med=st["med"], c_irtifa=self.c_irtifa, c_n=st["n"],
                          c_hiz_tavan=self.c_hiz_tavan)
        if not C["c_acik"]:
            return
        # ---- OLCEK-SERBEST / FIZIKSEL KAPILAR ----
        self.saglik = True; c_uyari = []
        if K["acik"]:
            if np.isfinite(self.artik) and self.yol > 0 and self.artik / self.yol > C["c_artik_oran_esik"]:
                self.saglik = False; c_uyari.append(f"C: artik/yol {100*self.artik/self.yol:.2f}% > {100*C['c_artik_oran_esik']:.1f}%")
            if np.isfinite(st["dagilim"]) and st["dagilim"] > C["c_dagilim_esik"]:
                self.saglik = False; c_uyari.append(f"C: yerel olcek dagilimi p90/p10={st['dagilim']:.1f} > {C['c_dagilim_esik']}")
            a, b = C["c_irtifa_ara"]
            if not (np.isfinite(self.c_irtifa) and a <= self.c_irtifa <= b):
                self.saglik = False; c_uyari.append(f"C: kestirilen irtifa {self.c_irtifa:.1f} m [{a},{b}] disi")
            if self.yol < K["yol_min_m"]:
                self.saglik = False; c_uyari.append(f"C: kalibrasyon yolu {self.yol:.0f} m < {K['yol_min_m']}")
            if C["c_mutlak_kapilar"]:
                if not (np.isfinite(self.artik) and self.artik <= K["artik_esik_m"]): self.saglik = False; c_uyari.append("C: mutlak artik")
                if not (K["s_min"] <= self.s <= K["s_max"]): self.saglik = False; c_uyari.append("C: mutlak s")
        self.uyarilar = [u for u in self.uyarilar if not u.startswith(("hizalama artigi", "olcek s=", "kalibrasyon yolu"))] + c_uyari
        self.tutucu = not self.saglik
        self.bilgi["saglik"] = self.saglik
        # ---- SUREKLI GUVEN SKORU (0..1) ----
        g_d = 1.0 / (1.0 + max(st["dagilim"] - 1.0, 0.0) / 1.0) if np.isfinite(st["dagilim"]) else 0.0
        _ao = max(float(C["c_artik_oran_esik"]), 1e-9)
        g_a = float(np.exp(-max(self.artik / max(self.yol, 1e-9), 0.0) / _ao)) if self.yol > 0 else 0.0
        self.guven = float(np.clip(min(g_d, g_a), 0.0, 1.0))
        self.bilgi["c_guven"] = self.guven
        # ---- GERI DUSUS: XY olcegi ----
        self.c_lam0 = 0.0
        if self.tutucu and C["c_olcek"] in ("med", "geo") and np.isfinite(st[{"med": "med", "geo": "geo"}[C["c_olcek"]]]):
            s_hedef = st["med"] if C["c_olcek"] == "med" else st["geo"]
            kir = float(np.clip(s_hedef / self.s, C["c_olcek_kirp"][0], C["c_olcek_kirp"][1]))
            self.c_lam0 = float(np.log(kir))
            self.x[3] += self.c_lam0
            self.uyarilar.append(f"C: XY olcegi s={self.s:.1f} -> s_med={s_hedef:.1f} (lam0={self.c_lam0:+.3f})")
        # ---- GERI DUSUS: OLCEK-SERBEST yaw duzeltmesi (EKF baslangic yonu) ----
        self.c_dyaw = 0.0
        if self.tutucu and C["c_yaw"]:
            try:
                al_k = np.nan_to_num(apply_recipe(dp, self.R, self.s, self.t), nan=0.0)
                dth, nk = yaw_duzeltmesi(al_k[:, :2], gt[:, :2], C["c_yaw_min_adim"])
                if nk >= 20 and abs(np.degrees(dth)) <= C["c_yaw_maks_deg"]:
                    self.c_dyaw = dth; self.x[2] += dth
                    self.uyarilar.append(f"C: olcek-serbest yaw duzeltmesi {np.degrees(dth):+.2f} deg (n={nk})")
            except Exception as e:
                self.uyarilar.append(f"C: yaw duzeltmesi hesaplanamadi: {e}")
        self.bilgi["c_dyaw_deg"] = float(np.degrees(self.c_dyaw))
        if self.tutucu and C["c_demir_gt"]:
            self.x[:2] = gt[-1, :2].copy()
            self._son_cikti = np.array([gt[-1, 0], gt[-1, 1], self._son_cikti[2]])
        if C["c_surekli"]:
            f = 1.0 + (C["c_guven_q"] - 1.0) * (1.0 - self.guven)
            self.Q = self.Q * (f ** 2)
            self.Pxy[3, 3] *= (f ** 2)
        self.bilgi.update(c_lam0=self.c_lam0)

    # ---------- fiziksel hiz siniri: buyukluk kirpmasi (yonu korur) ----------
    def _adim_filtre(self, d):
        C = getattr(self, "C", None)
        if C is None or C["c_hiz_mod"] != "kirp" or not np.isfinite(getattr(self, "c_hiz_tavan", np.nan)):
            return super()._adim_filtre(d)
        nd = float(np.linalg.norm(d))
        tav = self.c_hiz_tavan * (1 + getattr(self, "_bosluk", 0))
        return d * (tav / nd) if (np.isfinite(nd) and nd > tav > 0) else d

    # ---------- uyarlanir olcek: r_t = (h_A(t)/d_med(t))/s ----------
    def _uy_lam(self, zA_ham, d_med):
        """NEDENSEL: h_A = ORB olcek kilidi irtifasi (DPVO'dan BAGIMSIZ), d_med = DPVO yama derinligi.
        Fizik: metre/DPVO-birimi orani irtifa/derinlik ile olceklenir. Kalibrasyon penceresinde yerel olcek
        kararliysa (dagilim ~1) bu duzeltme YALNIZ gurultu katar -> sadece kapi dustugunde uygulanir."""
        C = self.C
        if d_med is not None and np.isfinite(d_med):
            self._uy_dm = (getattr(self, "_uy_dm", []) + [float(d_med)])[-int(C["c_uy_W"]):]
        dm = float(np.median(self._uy_dm)) if getattr(self, "_uy_dm", None) else np.nan
        if not (np.isfinite(dm) and dm > 1e-9 and zA_ham is not None and np.isfinite(zA_ham)
                and np.isfinite(getattr(self, "h0_kal", np.nan))):
            return getattr(self, "_uy_lam_son", 0.0)
        hA = self.h0_kal - (float(zA_ham) - self.gz_kal)
        r = (hA / dm) / self.s
        if not np.isfinite(r) or r <= 0:
            return getattr(self, "_uy_lam_son", 0.0)
        if C["c_uy_norm"]:
            if getattr(self, "_uy_rG", None) is None: self._uy_rG = r
            r = r / self._uy_rG
        lam = float(C["c_uy_beta"] * np.log(np.clip(r, C["c_uy_kirp"][0], C["c_uy_kirp"][1])))
        self._uy_lam_son = lam
        return lam

    def adim(self, dpvo_xyz, **kw):
        zA_ham = kw.get("zA_ham")
        C = getattr(self, "C", None)
        if C is not None and C["c_acik"] and C["c_uy"] and self.tutucu and self.kalibre:
            self._lr_son = self._uy_lam(zA_ham, kw.get("d_med"))
        out = super().adim(dpvo_xyz, **kw)
        C = getattr(self, "C", None)
        if out is None or C is None or not C["c_acik"] or not self.tutucu or C["c_z"] == "v4":
            return out
        if kw.get("health") == 1 and kw.get("gt_xyz") is not None:
            self._c_zA_ofset = (float(kw["gt_xyz"][2]) - float(zA_ham)) if (zA_ham is not None and np.isfinite(zA_ham)) else getattr(self, "_c_zA_ofset", 0.0)
            self._c_z_onceki = float(kw["gt_xyz"][2])            # GPS olayinda yeniden demir
            return out
        if zA_ham is None or not np.isfinite(zA_ham):
            return out
        z_yeni = float(zA_ham) + (getattr(self, "_c_zA_ofset", 0.0) if C["c_z"] == "A_ofset" else 0.0)
        o = np.array(out, float)
        K = self.P["koruma"]
        # DIKKAT: hiz siniri KENDI onceki Z'mize gore; ana sinifin _kelepce'si _son_cikti[2]'yi bu kare
        # icinde zaten DONUK degere dogru cekti -> ona gore kirpmak cekismeye ve Z'nin donuk kalmasina yol acar.
        z_onceki = getattr(self, "_c_z_onceki", None)
        if z_onceki is None: z_onceki = float(self._son_gt[2])
        if K["acik"]:
            z_yeni = float(np.clip(z_yeni, z_onceki - K["adim_maks_z"], z_onceki + K["adim_maks_z"]))
            z_yeni = float(np.clip(z_yeni, self._son_gt[2] - K["z_sapma_maks"], self._son_gt[2] + K["z_sapma_maks"]))
        self._c_z_onceki = z_yeni; o[2] = z_yeni
        self._son_cikti = o.copy(); self.son = tuple(float(v) for v in o)
        return self.son


class KestiriciE(_TK_TABAN):
    VARSAYILAN = dict(mod="yok", e_rms=1.0, e_dm=0.5, hiz_pen=30, sonme=0.6, kirp_kat=1.0, isit=10, kor_periyot=0, harman_w=0.5, e_rms_kat=0.0)

    def __init__(self, G=450, prm=None):
        super().__init__(G=G, prm=prm)
        self.E = dict(self.VARSAYILAN); self.E.update((prm or {}).get("E", {}))
        self._dm_buf = []; self._hiz_buf = []; self._son_yon = None; self._pr_kal = []; self._pr_med = None; self._e_rms_etkin = None
        self.bayrak = []          # kare basina sagliksiz mi (teshis)
        self._ns = 0

    # --- NEDENSEL dedektor ---
    def _sagliksiz(self, plan_rms, d_med):
        self._dm_buf.append(float(d_med) if (d_med is not None and np.isfinite(d_med)) else np.nan)
        if len(self._dm_buf) > 40: self._dm_buf = self._dm_buf[-40:]
        lr = np.nan
        if len(self._dm_buf) >= 26:
            a = np.nanmedian(self._dm_buf[-26:-5]); b = np.nanmedian(self._dm_buf[-5:])
            if np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0: lr = float(np.log(b / a))
        pr = float(plan_rms) if (plan_rms is not None and np.isfinite(plan_rms)) else 0.0
        self._pr_kal.append(pr) if not self.kalibre else None
        kor = self.E.get("kor_periyot", 0)
        if kor:                                 # KONTROL: dedektoru yok say, her kor_periyot karede bir atesle
            return bool((self.k % int(kor)) == 0), lr, pr
        e_rms = self.E["e_rms"]
        kat = self.E.get("e_rms_kat", 0.0)
        if kat > 0:                             # OZ-NORMALLESEN esik: kalibrasyon penceresinin kendi medyaninin kat katı
            if self._pr_med is None and self.kalibre and self._pr_kal:
                self._pr_med = float(np.median([x for x in self._pr_kal if np.isfinite(x)]))
            if self._pr_med is not None: e_rms = max(e_rms, kat * self._pr_med)
        self._e_rms_etkin = e_rms
        kotu = (pr > e_rms) or (np.isfinite(lr) and abs(lr) > self.E["e_dm"])
        return bool(kotu), lr, pr

    def adim(self, dpvo_xyz, d_med=None, n_cam=None, plan_rms=None, zA_ham=None, gt_xyz=None,
             health=0, r_orb=None, px_orb=None, kare_tekrar=False):
        kotu, lr, pr = self._sagliksiz(plan_rms, d_med)
        self.bayrak.append(kotu)
        self._E_kotu = kotu
        return super().adim(dpvo_xyz, d_med=d_med, n_cam=n_cam, plan_rms=plan_rms, zA_ham=zA_ham,
                            gt_xyz=gt_xyz, health=health, r_orb=r_orb, px_orb=px_orb, kare_tekrar=kare_tekrar)

    # kestirici_v4 icindeki adim_filtre kancasini kullanmak yerine kendi kancamiz:
    def _adim_filtre(self, d):
        """v4 bunu yalniz xy.adim_filtre=True iken cagirir; biz her zaman cagrilmasini saglamak icin
        asagidaki _E_uygula'yi adim() sonrasi degil, v4'un cagri noktasindan kullanmiyoruz.
        (bkz. kos_E: prm['xy']['adim_filtre']=True verilir, boylece bu metot her karede cagrilir.)"""
        mod = self.E["mod"]
        nd = float(np.linalg.norm(d))
        if np.isfinite(nd) and nd > 1e-9:
            yy = d / nd
            self._son_yon = yy if self._son_yon is None else (0.8 * self._son_yon + 0.2 * yy)
            if np.linalg.norm(self._son_yon) > 1e-9: self._son_yon /= np.linalg.norm(self._son_yon)
        if not getattr(self, "_E_kotu", False) or mod == "yok":
            if np.isfinite(nd): self._hiz_buf.append(nd); self._hiz_buf = self._hiz_buf[-self.E["hiz_pen"]:]
            return d
        self._ns += 1
        if len(self._hiz_buf) < self.E["isit"]:
            return d
        med = float(np.median(self._hiz_buf))
        if mod == "olu":
            return (self._son_yon if self._son_yon is not None else np.array([1.0, 0.0])) * med
        if mod == "kirp":
            tav = self.E["kirp_kat"] * med
            return d if nd <= tav else d / nd * tav
        if mod == "sonme":
            return d * self.E["sonme"]
        if mod == "dur":                       # KONTROL: artimi tamamen sifirla (olu hesap YOK)
            return np.zeros(2)
        if mod == "buyukluk":                  # YALNIZ buyuklugu degistir, DPVO yonunu koru
            return d / nd * med if nd > 1e-9 else d
        if mod == "harman":                    # yumusak gecis: (1-w)*DPVO + w*olu hesap
            w = float(self.E.get("harman_w", 0.5))
            olu = (self._son_yon if self._son_yon is not None else np.array([1.0, 0.0])) * med
            return (1.0 - w) * d + w * olu
        return d


class KestiriciE2(KestiriciE):
    def __init__(self, G=450, prm=None):
        super().__init__(G=G, prm=prm)
        self.E.setdefault("bayat_L", 0)          # kac ardisik bayraktan sonra sonum baslasin (0=kapali)
        self.E.setdefault("bayat_sonum", 0.85)   # kare basina buyukluk carpani
        self.E.setdefault("bayat_taban", 0.0)    # sonumun inebilecegi en dusuk carpan
        self.E.setdefault("olay_sus", 0)         # GPS olayindan sonra dedektoru kac kare SUSTUR (0=kapali)
        self._ardisik = 0; self._olay_k = -10**9

    def _adim_filtre(self, d):
        # OLAY SUSTURMA: bir GPS olayindan hemen sonra konum TAZE demirli, birikmis sapma SIFIR.
        # O anda ham DPVO adimi (gurultulu ama yansiz) BAYAT bir yondeki olu hesaptan daha iyidir.
        # Olculdu: folder4 GPS@1780 -> taban 5.3 m, olu hesap 44.4 m.
        sus = int(self.E.get("olay_sus", 0))
        if sus > 0 and (self.k - self._olay_k) <= sus:
            self.bayrak.append(False)
            return d
        d2 = super()._adim_filtre(d)
        L = int(self.E.get("bayat_L", 0))
        if L <= 0:
            return d2
        # super() bayragi bu karede kurdu: E._adim_filtre icinde self._son_bayrak set ediliyorsa onu kullan,
        # yoksa bayrak listesinin son elemani.
        bay = bool(self.bayrak[-1]) if len(self.bayrak) else False
        if bay:
            self._ardisik += 1
        else:
            self._ardisik = 0
            return d2
        if self._ardisik <= L:
            return d2
        k = max(float(self.E["bayat_taban"]),
                float(self.E["bayat_sonum"]) ** (self._ardisik - L))
        return d2 * k

    # GPS olayinda olu-hesap durumunu TAZELE (demirlenmis konumdan bayat yonle ucma)
    def adim(self, dpvo_xyz, **kw):
        if kw.get("health") == 1 and kw.get("gt_xyz") is not None and self.kalibre:
            self._ardisik = 0; self._olay_k = self.k
        return super().adim(dpvo_xyz, **kw)


class KestiriciUY2(KestiriciE2):
    VAR = dict(uy2_ac=False, uy2_beta=1.0, uy2_olu=0.35, uy2_W=31, uy2_M=200, uy2_kirp=1.5)

    def __init__(self, G=450, prm=None):
        super().__init__(G=G, prm=prm)
        self.U = dict(self.VAR); self.U.update((prm or {}).get("uy2", {}))
        self._u_dm = []; self._u_hist = []; self._rG = None; self._uy2_lam = 0.0

    def _uy2(self, zA_ham, d_med):
        U = self.U
        if d_med is not None and np.isfinite(d_med):
            self._u_dm = (self._u_dm + [float(d_med)])[-int(U["uy2_W"]):]
        if not self._u_dm: return self._uy2_lam
        dm = float(np.median(self._u_dm))
        if not (dm > 1e-9 and zA_ham is not None and np.isfinite(zA_ham)
                and np.isfinite(getattr(self, "h0_kal", np.nan)) and self.s > 0):
            return self._uy2_lam
        hA = self.h0_kal - (float(zA_ham) - self.gz_kal)
        r = (hA / dm) / self.s
        if not np.isfinite(r) or r <= 0: return self._uy2_lam
        if self._rG is None: self._rG = r; return self._uy2_lam
        self._u_hist = (self._u_hist + [float(np.log(r / self._rG))])[-int(U["uy2_M"]):]
        u = float(np.median(self._u_hist))
        olu = float(U["uy2_olu"])
        d = np.sign(u) * max(abs(u) - olu, 0.0)
        self._uy2_lam = float(np.clip(U["uy2_beta"] * d, -U["uy2_kirp"], U["uy2_kirp"]))
        return self._uy2_lam

    def adim(self, dpvo_xyz, **kw):
        if self.U.get("uy2_ac") and self.kalibre:
            # uretimdeki TOPLAMSAL lam kanali (_lr_son) -> EKF durumu (x[3]) BOZULMAZ.
            # rampa "drift" modunda x[3]'e yaziyor, biz _lr_son'a yaziyoruz; catisma yok.
            self._lr_son = self._uy2(kw.get("zA_ham"), kw.get("d_med"))
        return super().adim(dpvo_xyz, **kw)


class TermalKestirici(KestiriciC, KestiriciUY2):
    """v5 URETIM KESTIRICISI — C + E/E2/UY2 + F, taban v4 (_TK_TABAN).
    MRO: KestiriciC -> KestiriciUY2 -> KestiriciE2 -> KestiriciE -> _TK_TABAN.
    Butun yeni anahtarlar kapaliyken _TK_TABAN (v4) ile bit-bit ayni."""
    pass
