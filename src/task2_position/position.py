#!/usr/bin/env python3
"""
Görev 2 üst-seviye cephe (facade) — YALNIZ TERMAL (2026-09-03'ten itibaren).

  RGB   : ARTIK BURADA DEĞİL → src/task2_position/final_kestirici.py (FINAL-11: XY-EKF + 3 kaynaklı Z füzyonu).
          Eski RGB çözümü (Z v2 iki kaynak, decorrelation füzyonu) kaldırıldı (arşiv 2026-09-10'da silindi).
  Termal: XY = DPVO sharpen + Umeyama reçetesi ; Z = nesne-boyutu (+ ölçek kilidi).

Bu modül DPVO worker (dpvo env) içinde çalışır — Aligner/Z saf numpy/opencv/scipy.
İki mod:
  * STREAMING (yarışma): update_calib()/finalize_calib()/estimate() — nedensel (causal).
  * solve_offline(): tam diziyle kanıtlanmış çözümü üretir (test).
"""
import os

import numpy as np
import cv2
from src.task2_position import paths as P
from src.task2_position.aligner import Aligner, fit_recipe, apply_recipe
from src.task2_position.z_termal import ObjectSizeZ, OlcekKilidi


class PositionEstimator:
    def __init__(self, modality, gt_frames=None, rgb_focal=None, rgb_z_width=960):
        assert modality == "termal", "RGB icin src/task2_position/final_kestirici.FinalKestirici kullanilir (FINAL-11)"
        self.modality = modality
        self.G = gt_frames or P.GT_FRAMES
        self.aligner = Aligner(self.G, modality=modality)
        self.zt = ObjectSizeZ(gt_frames=self.G) if modality == "termal" else None
        self.termal_z_width = 640   # ObjectSizeZ'nin f=731.80'i bu genislik icin
        # ═══ TERMAL Z v2 — OLCEK KILIDI (2026-08-15) ═══
        # ObjectSizeZ'nin sinyali dogru ama OLCEGI kayabiliyor: h0 kalibrasyondan
        # GSD ile bulunuyor, oysa kalibrasyon penceresinde irtifa degismiyorsa
        # (2026: GT z std 0.41 m) olcek oradan COZULEMEZ -> 63.71 m hata.
        # Kilit, log(GSD irtifasi) = log h0 − k·L regresyonuyla olcegi ONLINE
        # cozer; ikinci kaynak olarak DPVO z'yi de kilitler, hakem GSD'dir.
        # Olculen (Z RMSE, GPS kapali kareler):
        #   2026     63.71 -> 7.33   |  Oturum 4  7.89 -> 4.82   (yalniz ilk450)
        #   2026     49.27 -> 7.27   |  Oturum 4  5.71 -> 4.74   (450+1600+2100)
        # Kapatmak icin: export NURON_TERMAL_Z_V2=0
        self.TERMAL_Z_V2 = (modality == "termal" and
                            os.environ.get("NURON_TERMAL_Z_V2", "1").strip() != "0")
        self.zk = OlcekKilidi(self.zt) if (self.zt is not None and self.TERMAL_Z_V2) else None
        self.zf = None                 # RGB kaynaklari kaldirildi (FINAL-11 -> final_kestirici.py)
        self.Z_FUSE_W = 0.6        # füzyon ağırlığı: 0.6·feature + 0.4·decorrelation
        self._gt_calib = []        # kalibrasyon GT (offset için)
        self._aligned = []         # dünya çerçevesi geçmiş (rgb Z decorrelation)
        self._calibrated = False
        # --- BAŞLANGIÇ ORIGIN'i: XYZ ilk kareden (kalkış) sıfırdan başlasın (GT gibi) ---
        self._origin = np.zeros(3)  # finalize_calib'te ilk GT karesi
        # --- RGB Z geçiş güvenliği (GPS kapanınca ANİ sıçramayı kes, GERÇEK değişimi bırak) ---
        # DİKKAT: z tam KİLİTLENMEZ — Oturum gibi z sonradan gerçekten düşebilir/çıkabilir.
        # Sadece fiziksel-olmayan ANİ sıçrama (ör. 20m/kare) kesilir; gerçek ~0.7 m/kare geçer.
        self._z_anchor = None      # GPS kapanışındaki son GT z (rate-limit başlangıcı)
        self._z_prev = None        # bir önceki kare düzeltilmiş z
        self.Z_RATE = 1.2          # kare-arası max |Δz| (m). gerçek ~0.7 → geçer; 20m ani → kesilir
        # MUTLAK Z-CLAMP (2026-07-16, güvenlik): rate-limit yavaş-birikmeli-yanlış-trendi durdurmaz
        # (40 kare × 1.2 = -48m birikebilir). Geçişten sonra ilk Z_CLAMP_FRAMES karede z, anchor'dan
        # ±Z_MAX_DEV'den fazla SAPAMAZ → zayıf-kalibrasyon videosunda -50/-500m felaketi kesilir.
        # Gerçek hareketi bozmaz (drone ~150 karede=20s 30m'den fazla irtifa değiştirmez).
        self.Z_MAX_DEV = 30.0      # anchor'dan izin verilen mutlak |z| sapması (m)
        self.Z_CLAMP_FRAMES = 150  # geçişten sonra clamp'ın aktif olduğu kare sayısı
        self._z_clamp_n = 0        # geçişten beri kare sayacı (finalize_calib'te 0'lanır)
        self.MAX_STEP = 2.0        # EVRENSEL kare-arası clamp (tum eksen). 7 videoda gercek maks
                                   # sicrama 1.5m (Z<=0.7) -> 1.8 ustu KESIN sahte (loop-closure teleport)
        self._last_emit = None     # son gonderilen dunya koordinati (clamp referansi)
        # --- XY geçiş güvenliği (GPS kapanınca loop-closure/kalibrasyon sıçraması) ---
        self._xy_prev = None       # bir önceki kare düzeltilmiş [x,y]
        # XY_RATE ADAPTİF: finalize_calib'te GPS açıkkenki (kalibrasyon sonu) gerçek
        # kare-arası XY hızından türetilir. Drone o an ne hızla gidiyorsa, kapandıktan
        # sonra o hızın biraz fazlasına (XY_RATE_FACTOR) izin ver → ışınlanmayı kes.
        self.XY_RATE = None        # finalize_calib'te hesaplanır
        self.XY_RATE_FACTOR = 2.0  # kalibrasyon-sonu hızının kaç katına izin (biraz fazla)
        self.XY_RATE_MIN = 3.0     # taban (drone kalibrasyonda çok yavaşsa bile bu kadar serbest)
        self.XY_GUARD_FRAMES = 4   # SADECE ilk bu kadar health=0 karede sınırla, sonra SERBEST
        self._xy_frames = 0        # health=0 kare sayacı (XY guard için)
        self.gps_olay_sayisi = 0

    def _zt_img(self, image_bgr):
        """TERMAL Z (ObjectSizeZ) icin kareyi SABIT 640 genislige normalize et.

        ObjectSizeZ sabit f = TERMAL_F = 731.80 kullanir; bu deger 640 px genis
        termal kare icin kalibre edildi. GSD irtifasi = f * v_metrik / v_piksel
        oldugundan, kareler 1280 gelirse v_piksel 2 KATINA cikar ve irtifa
        YARIYA duser -> h0 yanlis -> Z olcegi bozulur.
        RGB tarafinda _zf_img zaten 960'a kuculturuyordu; termalde boyle bir
        normalizasyon YOKTU. (2026-08-16)
        Bugun gelen kareler 640 -> bu fonksiyon NO-OP; saf sigorta."""
        h, w = image_bgr.shape[:2]
        if w == self.termal_z_width:
            return image_bgr
        return cv2.resize(image_bgr,
                          (self.termal_z_width,
                           int(round(h * self.termal_z_width / w))),
                          interpolation=(cv2.INTER_AREA
                                         if w > self.termal_z_width
                                         else cv2.INTER_LINEAR))

    # ---- kalibrasyon (ilk G kare, health=1) ----
    def update_calib(self, dpvo_xyz, gt_xyz, image_bgr=None):
        self.aligner.push_gt(dpvo_xyz, gt_xyz)
        self._gt_calib.append(np.asarray(gt_xyz, float))
        if self.zt is not None and image_bgr is not None:
            self.zt.feed(self._zt_img(image_bgr))
            if self.zk is not None:
                # kalibrasyonda metrik hiz GT'den gelir (hizalama henuz yok);
                # zd sonra finalize_calib'de geri doldurulur.
                gc = self._gt_calib
                vmet = (float(np.linalg.norm(gc[-1][:2] - gc[-2][:2]))
                        if len(gc) >= 2 else np.nan)
                self.zk.push(vmet, np.nan)

    def finalize_calib(self):
        ok = self.aligner.calibrate()
        if self.zt is not None:
            self.zt.calibrate_h0(np.stack(self._gt_calib)[:, :2])
        if self.zk is not None and ok:
            # Kalibrasyon boyunca hizalama yoktu -> DPVO z'si NaN birakilmisti.
            # Recete artik hazir; saklanan olceksiz pozlari donusturup geri doldur.
            try:
                pred = np.asarray(self.aligner._pred, float)
                al = self.aligner.transform(pred)
                m = min(len(self.zk.zd) - 1, len(al))
                if m > 0:
                    self.zk.zd[len(self.zk.zd) - m:] = [float(x) for x in al[-m:, 2]]
            except Exception:
                pass
        # GPS kapanış anındaki z'yi anchor'la: decorrelation kararlı veri toplayana
        # kadar buradan başlanır, ani -500m sıçramalar engellenir.
        if self._gt_calib:
            gc = np.stack(self._gt_calib)
            self._origin = gc[0].astype(float).copy()         # kalkış = origin (XYZ sıfırdan)
            self._z_anchor = float(gc[-1, 2])
            self._z_prev = self._z_anchor
            self._z_clamp_n = 0                                # mutlak-clamp sayacını sıfırla (geçiş başı)
            self._xy_prev = gc[-1, :2].astype(float).copy()   # son GT xy (geçiş başlangıcı)
            # ADAPTİF XY_RATE: kalibrasyon SONUNDAKİ gerçek kare-arası XY hızı (son 30 kare).
            # GPS açıkken drone bu hızla gidiyordu → kapandıktan sonra biraz fazlasına izin.
            tail = gc[-30:, :2] if len(gc) >= 31 else gc[:, :2]
            if len(tail) >= 2:
                spd = np.linalg.norm(np.diff(tail, axis=0), axis=1)
                v = float(np.percentile(spd, 90))             # son dönem tipik-üst hız
            else:
                v = self.XY_RATE_MIN
            self.XY_RATE = max(self.XY_RATE_MIN, v * self.XY_RATE_FACTOR)
        self._calibrated = ok
        return ok

    # ---- tahmin (kalibrasyon sonrası, nedensel) ----
    def estimate(self, dpvo_xyz, image_bgr=None):
        """Tek kare → dünya (x=Doğu, y=Kuzey, z=Aşağı). Kalibre değilse None."""
        if not self._calibrated:
            return None
        w = self.aligner.transform(dpvo_xyz)[0]          # [x,y,z_dpvo]
        # --- XY RATE-LIMIT: SADECE geçişin ilk XY_GUARD_FRAMES karesinde (ışınlanmayı
        #     yumuşat), sonra SERBEST → DPVO gerçek XY'yi tuttursun. ---
        self._xy_frames += 1
        if self._xy_prev is not None and self._xy_frames <= self.XY_GUARD_FRAMES and self.XY_RATE:
            dxy = w[:2] - self._xy_prev
            n = float(np.linalg.norm(dxy))
            if n > self.XY_RATE:
                w[:2] = self._xy_prev + dxy * (self.XY_RATE / n)   # yönü koru, büyüklüğü kes
        self._xy_prev = w[:2].astype(float).copy()
        self._aligned.append(w.copy())
        if self.modality == "termal":
            if image_bgr is not None:
                self.zt.feed(self._zt_img(image_bgr))
                if self.zk is not None:
                    onc = self._aligned[-2][:2] if len(self._aligned) >= 2 else None
                    vmet = (float(np.linalg.norm(w[:2] - onc))
                            if onc is not None else np.nan)
                    self.zk.push(vmet, float(w[2]))
            gz_kal = np.stack(self._gt_calib)[:, 2]
            if self.zk is not None and len(self.zk.vm) >= 5:
                z = float(self.zk.z(gz_kal)[-1])       # OLCEK KILIDI
            else:
                z = self.zt.z(gz_kal)[-1]              # eski yol (kilit kapali)
        else:
            raise RuntimeError("RGB modalitesi PositionEstimator'da desteklenmiyor (FinalKestirici)")
        if self.modality != "termal":
            # --- RATE-LIMIT: SADECE ani sıçramayı kes, gerçek değişimi bırak ---
            # z_prev anchor'dan başlar (GPS kapanış anı). z_raw ne derse desin kare-arası
            # en fazla Z_RATE kadar hareket eder → 20m'lik ani spike yumuşatılır, ama
            # gerçek ~0.7 m/kare değişim (Oturum'daki iniş/çıkış dahil) tam geçer.
            z = z_raw
            if self._z_prev is not None:
                dz = z - self._z_prev
                if dz > self.Z_RATE:  z = self._z_prev + self.Z_RATE
                elif dz < -self.Z_RATE: z = self._z_prev - self.Z_RATE
            # MUTLAK CLAMP: geçiş sonrası ilk Z_CLAMP_FRAMES karede z anchor'dan ±Z_MAX_DEV içinde.
            # Yavaş-birikmeli felaketi (rate-limit'in kaçırdığı) keser; normal aralıkta devreye girmez.
            if self._z_anchor is not None and self._z_clamp_n < self.Z_CLAMP_FRAMES:
                lo, hi = self._z_anchor - self.Z_MAX_DEV, self._z_anchor + self.Z_MAX_DEV
                if z < lo:   z = lo
                elif z > hi: z = hi
                self._z_clamp_n += 1
            self._z_prev = z
        w[2] = z
        out = (w - self._origin).astype(float)     # XYZ kalkışa göre (sıfırdan)
        # --- EVRENSEL KARE-ARASI CLAMP: gerçek maks ~1.5m/kare → >1.8m sahte (teleport) kes ---
        if self._last_emit is not None:
            d = out - self._last_emit
            for i in range(3):
                if d[i] >  self.MAX_STEP:  out[i] = self._last_emit[i] + self.MAX_STEP
                elif d[i] < -self.MAX_STEP: out[i] = self._last_emit[i] - self.MAX_STEP
        self._last_emit = out.copy()
        return out

    # ---- GPS OLAYI: kalibrasyon SONRASI health=1 karesi geldiginde ----
    def gps_olayi(self, gt_z):
        # --- TERMAL: sonumlu ofset (kalici yama oncülü bozuyor, olculdu) ---
        if self.modality == "termal":
            if self.zk is not None:
                self.zk.gps_olayi(float(gt_z), len(self._aligned))
                self.gps_olay_sayisi = getattr(self, "gps_olay_sayisi", 0) + 1
            return
        return   # RGB yolu kaldirildi (FinalKestirici)

    # ---- OFFLINE (tam dizi, kanıtlanmış) ----
    def solve_offline(self, kare, pred, gt, video=None):
        R, s, t = fit_recipe(pred, gt, self.G)
        al = apply_recipe(pred, R, s, t)
        if self.modality != "rgb":
            if video is None:
                raise ValueError("termal Z için video gerek")
            from src.task2_position.z_termal import from_video
            z, _, _ = from_video(video, None, gt_frames=self.G) if False else (None, None, None)
            # not: from_video GT csv ister; offline testte z_termal.from_video ayrı çağrılır
        return kare, al, gt
