#!/usr/bin/env python3
"""
GPS worker — SADECE `dpvo` conda env'inde çalışır (Python 3.11, derlenmiş CUDA uzantısı).

Base env'deki orkestratör (gps_bridge.py) ile DOSYA-IPC:
  runtime/gps/in/{seq}.jpg   ← orkestratör kareyi yazar
  runtime/gps/out/{seq}.txt  → bu worker RGB: "x y z d_med nx ny nz rms" (8 sütun) yazar;
                               TERMAL (FINAL-TERMAL 2026-09-04): "tekrar_bayragi + K×(x y z d_med nx ny nz rms)" (1+8K sütun; K=2 → 17)
                               (ölçeksiz DPVO konumu + yama-derinliği medyanı + yer-düzlemi normali/rms;
                                FINAL-11 Z füzyonu ve nadir/oblik kararı için — 2026-09-03)
  runtime/gps/STOP           → durdur sinyali

Bu worker DPVO'yu besler ve o ana kadarki ölçeksiz kamera konumunu üretir. Ölçekleme/
hizalama/kalibrasyon base env'de PositionEstimator ile yapılır (bu worker sadece VO).

ÇALIŞTIRMA (dpvo env):
    conda run -n dpvo python scripts/gps_worker.py --modality termal
    # veya:  "$(conda info --base)/envs/dpvo/bin/python" scripts/gps_worker.py --modality rgb
"""
import os
import sys
import time
import argparse
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.task2_position import paths as P     # noqa: E402
from src.common import gpu_kuyruk as KUY      # noqa: E402  (GPU kuyrugu — yalniz stdlib)
import torch                                  # noqa: E402  (empty_cache; dpvo env'de zaten var)
sys.path.insert(0, P.DPVO_REPO)               # dpvo deposunu path'e ekle
import cv2                                    # noqa: E402
from src.task2_position.dpvo_runner import DPVOTracker  # noqa: E402

# 2026-09-17 — KOSU KLASORU: istemci her kosuda kendi klasorunu acar
# (kosular/folder<N>/runtime/gps). Bu worker AYRI surec oldugu icin o yolu
# SABIT isaret dosyasindan ogrenir: runtime/AKTIF_KOSU (istemci acilirken yazar).
# Ilk kare islenene kadar isaret her turda yeniden okunur -> istemci bizden SONRA
# acilsa da (RGB'de worker once baslar) dogru klasore BAGLANIRIZ.
from src.common import kosu as KOSU                # noqa: E402

GPS_DIR = IN_DIR = OUT_DIR = STOP = None           # _kosu_baglan() doldurur


def _bayat_kosu(d):
    """d ESKI (bitmis ya da yarim kalmis) bir kosunun GPS klasoru mu?
    Olcut: out/ icinde bir worker'in yazdigi yanit (.txt) var. Yeni istemci kendi
    klasorunu BOS acar (ve _fresh_start ile in/out'u bosaltir); dolu out/ demek
    o kosuya zaten baska bir worker bakmis demek."""
    if not os.path.isdir(d):
        return True          # 2026-09-18: isaretin gosterdigi klasor SILINMIS -> eski; yeniden YARATMA
    try:
        return any(f.endswith(".txt") for f in os.listdir(os.path.join(d, "out")))
    except OSError:
        return False


def _kosu_baglan(bekle=True, log=True):
    """Isaret dosyasini oku, degistiyse klasorlere baglan. Doner: degisti mi.
    2026-09-18: isaret ESKI bir kosuya (out/ dolu) isaret ediyorsa ona BAGLANMAYIZ.
    Aksi halde eski kareleri yeniden islemeye baslar, sayac 0'dan ayrilir ve yeni
    istemcinin klasorune bir daha gecemezdik (istemci 180 sn sonra GPS'siz kalirdi).
    bekle=True : temiz bir isaret yazilana kadar bekle.
    bekle=False: isaret yok ya da eskiyse 'degisiklik yok' say (False)."""
    global GPS_DIR, IN_DIR, OUT_DIR, STOP
    yeni_dir = KOSU.isaret_oku()
    uyari = None
    while yeni_dir is None or (yeni_dir != GPS_DIR and _bayat_kosu(yeni_dir)):
        if not bekle:
            return False
        msg = ("[gps_worker] istemci bekleniyor (runtime/AKTIF_KOSU yok)..." if yeni_dir is None
               else f"[gps_worker] AKTIF_KOSU eski/bitmis kosuya isaret ediyor: {yeni_dir}\n"
                    f"             -> yeni istemci (yeni kosu klasoru) bekleniyor...")
        if msg != uyari:
            print(msg, flush=True)
            uyari = msg
        time.sleep(0.2)
        yeni_dir = KOSU.isaret_oku()
    if yeni_dir == GPS_DIR:
        return False
    GPS_DIR = yeni_dir
    IN_DIR = os.path.join(GPS_DIR, "in")
    OUT_DIR = os.path.join(GPS_DIR, "out")
    STOP = os.path.join(GPS_DIR, "STOP")
    os.makedirs(IN_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    if log:
        print(f"[gps_worker] kosu klasoru: {GPS_DIR}", flush=True)
    return True


def _auto_detect_modality():
    """İlk kareyi (runtime/gps/in/000000.jpg) bekle, görüntüden modaliteyi algıla.
    Böylece kullanıcı --modality yazmak ZORUNDA değil; iki konsolun sırası da önemsiz."""
    from src.common.modality import detect_from_image
    first = os.path.join(IN_DIR, "000000.jpg")
    print("[gps_worker] modalite için ilk kare bekleniyor (ana arayüzü başlat)...", flush=True)
    while not os.path.exists(STOP):
        if os.path.exists(first):
            for _ in range(50):
                img = cv2.imread(first)
                if img is not None:
                    m = detect_from_image(img)
                    print(f"[gps_worker] modalite algılandı: {m.upper()}", flush=True)
                    return m
                time.sleep(0.02)
        time.sleep(0.05)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", default="rgb", choices=["rgb", "termal"],
                    help="YARISMA 2026: RGB sabit (termal yok). Varsayilan rgb.")
    ap.add_argument("--idle_timeout", type=float, default=0.0,
                    help="kaç sn kare gelmezse çık (0=sonsuz bekle)")
    args = ap.parse_args()

    KUY.temizle()                                  # bayat GPU kuyrugu dosyalarini sil

    m = args.modality  # YARISMA 2026: RGB sabit (otomatik tespit kaldirildi)

    print(f"[gps_worker/{m}] DPVO yükleniyor...", flush=True)
    # ═══ FINAL-TERMAL (2026-09-04): termalde K DPVO örneği (sağlam ensemble) + tekrar-kare atlama; RGB tek örnek, değişmedi ═══
    prof = P.GPS_PROFILE[m]; K = int(prof.get("K", 1)) if m == "termal" else 1; tekrar_esik = float(prof.get("tekrar_esik", 0.0)) if m == "termal" else 0.0
    trackers = []
    for k in range(K):
        try:
            trackers.append(DPVOTracker(m, ckpt=P.DPVO_CKPT(m), calib=P.CALIB(m), cfg_file=P.DPVO_CFG))
        except Exception as e:                      # GPU belleği yetmezse daha az örnekle devam
            print(f"[gps_worker/{m}] örnek {k} açılamadı ({e}) -> K={len(trackers)}", flush=True); break
    K = max(1, len(trackers)); tracker = trackers[0]
    print(f"[gps_worker/{m}] DPVO hazır. K={K} örnek, tekrar_esik={tekrar_esik}.", flush=True)
    # MODEL YUKLENDIKTEN SONRA kosu klasorune baglan: istemci bizden sonra acilsa
    # bile DPVO yuklemesi bosa beklemez (istemcinin ilk yanit butcesi 180 sn).
    _kosu_baglan()
    if os.path.exists(STOP):
        os.remove(STOP)
    print(f"[gps_worker/{m}] kare bekleniyor: {IN_DIR}", flush=True)
    son_satir = None

    seq = 0
    last_seen = time.time()
    kuyruk_n = kuyruk_sn = 0
    while True:
        if seq == 0 and _kosu_baglan(bekle=False):
            # istemci bizden SONRA acildi (ya da yeniden basladi) -> yeni klasore gec
            print("[gps_worker] istemci yeni kosu acti -> klasor degisti", flush=True)
        if os.path.exists(STOP):
            print("[gps_worker] STOP sinyali, çıkılıyor.", flush=True)
            break
        # ═══ GPU KUYRUGU (2026-09-10): istemci OOM aldiysa bellegi BIRAK ve o is
        #     bitene kadar DUR. Kart 8 GB, istemci G3 tepesi ~5.5 GB; bu worker
        #     oturum boyunca 0.6 -> 2.1 GB'a cikiyor ve istemciyi OOM'a sokuyordu.
        #     Burada beklemek AKISI YAVASLATMAZ: istemci G3'u bitirmeden bu karenin
        #     girdisini (in/{seq}.jpg) yazmaz, yani bu worker zaten bos bekliyordu.
        _dur = KUY.worker_kontrol(bosalt=torch.cuda.empty_cache)
        if _dur > 0:
            kuyruk_n += 1; kuyruk_sn += _dur
            print(f"\n[gps_worker/{m}] GPU kuyrugu: bellek birakildi, {_dur*1000:.0f} ms "
                  f"durdum (istemci OOM) — toplam {kuyruk_n} kez / {kuyruk_sn:.1f} sn", flush=True)
        inp = os.path.join(IN_DIR, f"{seq:06d}.jpg")
        if not os.path.exists(inp):
            if args.idle_timeout and (time.time() - last_seen > args.idle_timeout):
                print("[gps_worker] idle timeout, çıkılıyor.", flush=True)
                break
            time.sleep(0.003)
            continue
        # kare hazır olana kadar minik bekleme (yazım tamamlansın)
        img = None
        for _ in range(50):
            img = cv2.imread(inp)
            if img is not None:
                break
            time.sleep(0.002)
        if img is None:
            seq += 1
            continue
        if m == "termal":
            # ─── FINAL-TERMAL: tekrar kare → DPVO'ya verme, son satırı bayrakla tekrarla; değilse K örneği GERÇEK indeks (seq) ile besle ───
            tekrar = tekrar_esik > 0 and tracker.tekrar_mi(img, tekrar_esik)
            if tekrar and son_satir is not None:
                satir = son_satir.copy(); satir[0] = 1.0
            else:
                bloklar = []
                for k, tr in enumerate(trackers):
                    xyz = tr.feed(img, tstamp=seq, seed=k * 100003 + seq)
                    if xyz is None:
                        xyz = np.zeros(3, np.float64)
                    d_med, n_cam, rms = tr.ek_cikti()
                    bloklar.append(np.concatenate([np.asarray(xyz, np.float64), [d_med], np.asarray(n_cam, np.float64), [rms]]))
                satir = np.concatenate([[0.0]] + bloklar)           # [tekrar_bayragi, K x (x y z d_med nx ny nz rms)]
                son_satir = satir.copy()
            xyz = satir[1:4]
        else:
            xyz = tracker.feed(img)                       # ölçeksiz DPVO konumu (veya None)
            if xyz is None:
                xyz = np.zeros(3, np.float64)
            d_med, n_cam, rms = tracker.ek_cikti()        # FINAL-11: derinlik + yer-düzlemi (nan olabilir)
            satir = np.concatenate([np.asarray(xyz, np.float64), [d_med], np.asarray(n_cam, np.float64), [rms]])
        outp = os.path.join(OUT_DIR, f"{seq:06d}.txt")
        tmp = outp + ".tmp"
        np.savetxt(tmp, satir.reshape(1, -1), fmt="%.6f")
        os.replace(tmp, outp)                          # atomik yazım
        # ekranda CANLI ilerleme: her 5 karede bir tek satir (üzerine yazar).
        if seq % 5 == 0:
            print(f"\r[gps_worker/{m}] islenen kare: {seq}  son xyz={np.round(xyz,2)}   ",
                  end="", flush=True)
        seq += 1
        last_seen = time.time()


if __name__ == "__main__":
    main()
