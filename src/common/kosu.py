#!/usr/bin/env python3
"""
KOSU KLASORU — her calistirmaya kendi klasoru (2026-09-17).

Amac: ust uste 4-5 kosu yapilabilsin, aralarinda elle temizlik GEREKMESIN ve
hicbir kosunun ciktisi digerinin uzerine yazmasin.

Duzen (NURON_DRONE kokunde):
  kosular/folder0/  kareler/    <- sunucudan inen kareler
                    frames/     <- islenen kare kopyalari (new/ + old/)
                    log/        <- <takim>_<tarih>.log
                    tahminler/  <- sunucuya GIDEN JSON'lar (<oturum>/predictions_*.jsonl)
  kosular/folder1/  ... (bir sonraki kosu)

Klasor adi istemci acilirken BIR KEZ secilir: bos olan ilk folder<N>.
Kesintiden sonra AYNI klasore devam etmek icin:  export NURON_KOSU=folder3

DIKKAT: runtime/gps/ (worker ile dosya-IPC) BURAYA GIRMEZ — ayri surec oldugu
icin sabit yolda kalmak ZORUNDA; koprü zaten her oturum basinda temizler.
"""
import os

from src.common.config import ROOT

KOSULAR = os.path.join(ROOT, "kosular")
_SECILEN = None


def kosu_dir() -> str:
    """Bu surecin kosu klasoru (ilk cagrida secilir ve olusturulur, sonra sabit)."""
    global _SECILEN
    if _SECILEN is not None:
        return _SECILEN
    os.makedirs(KOSULAR, exist_ok=True)
    ad = os.environ.get("NURON_KOSU", "").strip()
    if ad:                                       # kesinti-devam: elle verilen klasor
        yol = os.path.join(KOSULAR, ad)
        os.makedirs(yol, exist_ok=True)
    else:
        n = 0
        while True:                              # bos olan ilk folder<N>
            yol = os.path.join(KOSULAR, f"folder{n}")
            try:
                os.makedirs(yol)                 # exist_ok=False: yaris durumunda da tek sahip
                break
            except FileExistsError:
                n += 1
    _SECILEN = yol
    return yol


def alt(ad: str) -> str:
    """Kosu klasoru altinda bir alt klasor (yoksa olusturur). Donen: mutlak yol."""
    p = os.path.join(kosu_dir(), ad)
    os.makedirs(p, exist_ok=True)
    return p


# ─────────────────── WORKER ILE BAGLANTI (isaret dosyasi) ────────────────────
# Worker AYRI bir surec (dpvo env) ve istemcinin hangi folder<N>'i sectigini
# BILEMEZ. Istemci acilirken bu SABIT dosyaya kendi GPS klasorunu yazar; worker
# kare gelmeye baslayana kadar bu dosyayi okumaya devam eder ve ona baglanir.
ISARET = os.path.join(ROOT, "runtime", "AKTIF_KOSU")


def isaret_yaz(gps_dir: str) -> None:
    """Istemci: aktif kosunun GPS klasorunu duyur (atomik)."""
    os.makedirs(os.path.dirname(ISARET), exist_ok=True)
    tmp = ISARET + ".tmp"
    with open(tmp, "w") as f:
        f.write(gps_dir)
    os.replace(tmp, ISARET)


def isaret_oku():
    """Worker: aktif kosunun GPS klasoru (yoksa None)."""
    try:
        with open(ISARET) as f:
            y = f.read().strip()
        return y or None
    except OSError:
        return None
