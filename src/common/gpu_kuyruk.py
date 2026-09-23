#!/usr/bin/env python3
"""
GPU KUYRUGU — istemci (G1 + G3) ile DPVO worker'i arasinda BELLEK SIRASI.

SORUN (olculdu 2026-09-10, 8 GB RTX 4060):
  Iki AYRI surec ayni karti paylasiyor. Istemci G3 tepe noktasinda ~5.5 GB
  ister; worker (DPVO) oturum boyunca 0.6 GB'dan 2.1 GB'a CIKAR. Toplam kartin
  7.62 GB'ini asinca istemcide "CUDA out of memory" olur ve o karede referans
  kutusu gitmez (2249 karede 181 kare). `torch.cuda.empty_cache()` YALNIZ kendi
  surecinin onbellegini birakir; eksik bellek OTEKI surecte durdugu icin tek
  basina hicbir sey kurtarmaz (olculdu: 130 denemenin 130'u yine patladi).

COZUM — SIRA, EKSILTME DEGIL:
  Hicbir model bosaltilmaz, hicbir kare atlanmaz, cozunurluk/esik degismez.
  Yalniz OOM ANINDA gorevler sıraya alinir:

    1. Istemci OOM alir            -> kendi onbellegini bosaltir + ISTEK yazar
    2. Worker ISTEK'i gorur        -> empty_cache() yapar, ONAY yazar ve DURUR
    3. Istemci ONAY'i gorur        -> isi TEKRAR dener (artik yer var)
    4. Istemci ISTEK'i siler       -> worker devam eder -> G2 calisir -> sonuc gonderilir

  Worker'in durdurulmasi BEDAVA: kare akisinda G3 calisirken worker o karenin
  girdisini henuz almamistir (istemci girdiyi G2 adiminda yazar), yani bos bekler.

  Dosya-tabanli IPC — projenin geri kalaniyla ayni idiom (runtime/gps/).
  YALNIZCA stdlib: worker `dpvo` env'inde bu modulu bedavaya import eder.

KAPATMA:  export NURON_GPU_KUYRUK=0   -> davranis eski surumun birebir aynisi.
"""
import os
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
KUYRUK_DIR = os.path.join(_ROOT, "runtime", "gps")
ISTEK = os.path.join(KUYRUK_DIR, "GPU_ISTEK")     # istemci -> worker: "bellegi birak ve dur"
ONAY  = os.path.join(KUYRUK_DIR, "GPU_ONAY")      # worker -> istemci: "biraktim, duruyorum"

ACIK = os.environ.get("NURON_GPU_KUYRUK", "1").strip().lower() not in ("0", "false", "kapali")
ONAY_BEKLE   = float(os.environ.get("NURON_KUYRUK_ONAY_SN", "6.0"))    # istemci ONAY'i en fazla bu kadar bekler
WORKER_BEKLE = float(os.environ.get("NURON_KUYRUK_DUR_SN", "45.0"))    # worker en fazla bu kadar durur (kilitlenme sigortasi)


def _yaz(p):
    try:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write(str(time.time()))
        return True
    except OSError:
        return False


def _sil(p):
    try:
        os.remove(p)
    except OSError:
        pass


# ───────────────────────── ISTEMCI TARAFI (base env) ─────────────────────────
class gps_durdur:
    """OOM aninda GPS worker'ini durdurup bellegini aldiran baglam yoneticisi.

        with gps_durdur("G3") as k:
            if k.onay: ...        # worker bellegi birakti
            sonuc = tekrar_dene()

    Cikista ISTEK silinir -> worker kaldigi yerden devam eder. Worker hic
    yoksa/yanit vermezse onay=False doner ve is yine denenir (zarar yok)."""

    def __init__(self, sebep="G3", bekle=None):
        self.sebep = sebep
        self.bekle = ONAY_BEKLE if bekle is None else bekle
        self.onay = False
        self.gecen = 0.0

    def __enter__(self):
        if not ACIK:
            return self
        _sil(ONAY)                       # bayat onay okumayalim
        if not _yaz(ISTEK):
            return self
        t0 = time.time()
        while time.time() - t0 < self.bekle:
            if os.path.exists(ONAY):
                self.onay = True
                break
            time.sleep(0.01)
        self.gecen = time.time() - t0
        return self

    def __exit__(self, *_):
        if ACIK:
            _sil(ISTEK)                  # worker devam etsin
        return False                     # istisnayi YUTMA


# ───────────────────────── WORKER TARAFI (dpvo env) ──────────────────────────
def worker_kontrol(bosalt=None, log=None):
    """Worker dongusunde her turda cagrilir. ISTEK varsa: bellegi birak, ONAY yaz,
    ISTEK silinene kadar DUR. Dondurdugu: durdugu saniye (0.0 = istek yoktu).

    bosalt: bellegi birakan cagrilabilir (orn. torch.cuda.empty_cache)."""
    if not ACIK or not os.path.exists(ISTEK):
        return 0.0
    if bosalt is not None:
        try:
            bosalt()
        except Exception:
            pass
    _yaz(ONAY)
    t0 = time.time()
    while os.path.exists(ISTEK) and (time.time() - t0) < WORKER_BEKLE:
        time.sleep(0.01)
    _sil(ONAY)
    gecen = time.time() - t0
    if log is not None:
        log(gecen)
    return gecen


def temizle():
    """Oturum basi: bayat kuyruk dosyalarini sil."""
    _sil(ISTEK)
    _sil(ONAY)
