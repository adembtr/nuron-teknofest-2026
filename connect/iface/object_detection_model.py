import logging
import os
import sys
import time

import cv2
import requests
import torch

from .constants import classes, landing_statuses, moving_statuses
from .detected_object import DetectedObject
from .detected_translation import DetectedTranslation
from .reference_prediction import ReferencePrediction

# --- NURON: NURON_DRONE cekirdegini import edilebilir yap (src.* ve connect.*) ---
# connect/iface/object_detection_model.py -> parents[2] = NURON_DRONE koku
_DRONE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _DRONE_ROOT not in sys.path:
    sys.path.insert(0, _DRONE_ROOT)

from src.common.config import ROOT                        # noqa: E402
from src.common import modality as MOD                     # noqa: E402
from src.common import runtime as RT                       # noqa: E402
from src.common import kosu as KOSU                         # noqa: E402  (kosular/folder<N>)
from src.common.gpu_kuyruk import gps_durdur                # noqa: E402  (GPU kuyrugu, 2026-09-10)
# ═══ A — PROAKTIF GPU KUYRUGU (2026-09-12): referans penceresi ICINDE her karede G3'ten ONCE worker'a "bellegi birak + dur"
#     denir (OOM BEKLENMEZ). 12 Eyl termal E2E: reaktif kuyruk 66 kez tetiklendi (her biri +1-2.6 sn, worker toplam 124 sn durdu).
#     Model / cozunurluk / esik / DPVO tamponu DEGISMEZ, kare atlanmaz. Kapatma: NURON_GPU_KUYRUK_PROAKTIF=0 (eski reaktif davranis).
GPU_KUYRUK_PROAKTIF = os.environ.get("NURON_GPU_KUYRUK_PROAKTIF", "1").strip().lower() not in ("0", "false", "kapali")
from src.task1_detection.pipeline import DetectionPipeline  # noqa: E402
from src.task1_detection.landing import (                  # noqa: E402
    LandingEstimator, UAP_CLS, UAI_CLS, OBSTACLE_CLS)
# GOREV 3 v2 (2026-09-01): eski v1 (session.ReferenceSession) KALDIRILDI.
# ReferenceSessionV2 modaliteye gore dagitir:  rgb -> rgb2 (M16 motoru)
#                                              termal -> termal2 (DOKUNULMADI)
from src.task3_reference.g3_v2 import ReferenceSessionV2 as ReferenceSession, _oid, stem_oid  # noqa: E402

VEHICLE_CLS = 0


class ObjectDetectionModel:
    """NURON entegrasyonu — resmi arayuz ile NURON_DRONE modellerini birlestirir.

    Bu SINIF sadece __init__/detect acisindan degistirildi; protokol (auth, kare
    gating, resume, rate-limit) connection_handler + main.py'de AYNEN korunur.

    Agir modeller ILK karede (modalite belli olunca) BIR KEZ yuklenir:
      G1  DetectionPipeline (YOLO + tasit hareket)
      G1  LandingEstimator  (UAP/UAI inis, PaDiM)
      G3  ReferenceSessionV2 (rgb: FastSAM+C-RADIOv3-H+DAM4SAM · termal: termal2)
      G2  GpsBridge         (DPVO worker'ina dosya-IPC; worker AYRI dpvo env'de)

    Not: G2 icin DPVO worker'i ayri konsolda (dpvo env) calismalidir. Worker yoksa
    GpsBridge graceful degrade eder (health=1'de GT echo, health=0'da son konumu tutar).
    """

    def __init__(self, evaluation_server_url):
        logging.info('Created Object Detection Model (NURON)')
        self.evaulation_server = evaluation_server_url
        # --- lazy state (ilk karede kurulur) ---
        self.modality = None
        self.pipe = None
        self.landing = None
        self.ref = None
        self.gps = None
        self.url_to_id = {}      # G3: sunucu referans url'si -> banka object_id (COK url -> 1 id OLABILIR)
        self.id_to_url = {}      # G3: banka object_id -> ILK url (geri uyumluluk; gonderimde KULLANILMAZ)
        self._bank_path = None   # kurulmus ref_bank.npz yolu
        self._url_to_stem = {}   # referans url -> dosya-adi(stem); ayni dosya birden fazla url'de olabilir
        self._ref_dir = None     # indirilen ham referans klasoru (stem -> oid icin)
        self._son_aralik = None  # G3: son karede aktif pencere url'leri; degisince takip sifirlanir
        self._oom_n = 0          # GPU kuyrugu: OOM sayisi
        self._oom_ok = 0         # kuyruk devreye girip is KURTARILAN kare sayisi
        self._oom_fail = 0       # kuyruktan sonra da yer bulunamayan kare sayisi
        self._pro_n = 0          # A (2026-09-12): proaktif kuyruk sayisi (pencere ici G3 cagrisi)
        self._pro_oom = 0        # proaktif durdurmaya ragmen OOM (beklenen 0)
        self._pro_kapali = False # worker ONAY vermezse proaktif mod kapanir -> reaktif
        self._bank_cropped = False   # HQ-SAM crop + banka npz kuruldu mu (modaliteden bagimsiz)
        self._ref_session_ready = False  # ReferenceSession (modaliteye bagli) hazir mi
        self.frame_count = 0

    # ---------------------------------------------------------------
    # Resmi ornekten AYNEN: kare goruntusunu indir (tekrar indirmez).
    # ---------------------------------------------------------------
    @staticmethod
    def download_image(img_url, images_folder, images_files, retries=3, initial_wait_time=0.1, auth_token=None):
        t1 = time.perf_counter()
        wait_time = initial_wait_time
        image_name = img_url.split("/")[-1]
        if image_name not in images_files:
            headers = {'Authorization': f'Token {auth_token}'} if auth_token else {}
            for attempt in range(retries):
                try:
                    response = requests.get(img_url, headers=headers, timeout=60)
                    response.raise_for_status()
                    img_bytes = response.content
                    with open(images_folder + image_name, 'wb') as img_file:
                        img_file.write(img_bytes)
                    t2 = time.perf_counter()
                    logging.info(f'{img_url} - Download Finished in {t2 - t1} seconds to {images_folder + image_name}')
                    return
                except requests.exceptions.RequestException as e:
                    logging.error(f"Download failed for {img_url} on attempt {attempt + 1}: {e}")
                    logging.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    wait_time *= 2
            logging.error(f"Failed to download image from {img_url} after {retries} attempts.")
        else:
            logging.info(f'{image_name} already exists in {images_folder}, skipping download.')

    def process(self, prediction, evaluation_server_url, health_status, images_folder, images_files,
                active_refs=None, ref_image_paths=None, auth_token=None):
        # Kareyi indir (resmi ornekten aynen), sonra self.detect() ile 3 gorevi calistir.
        self.download_image(evaluation_server_url + "media" + prediction.image_url,
                            images_folder, images_files, auth_token=auth_token)
        frame_image_path = images_folder + prediction.image_url.split("/")[-1]
        frame_results = self.detect(prediction, health_status,
                                    active_refs=active_refs or [],
                                    ref_image_paths=ref_image_paths or {},
                                    frame_image_path=frame_image_path)
        return frame_results

    # ---------------------------------------------------------------
    # REFERANS: INER INMEZ CROP (kare dongusunden ONCE, modaliteden bagimsiz)
    # ---------------------------------------------------------------
    def prepare_references(self, ref_image_paths):
        """main.py referanslari indirir indirmez cagirir → HQ-SAM ile HEMEN crop edip
        nesneleri cikartir ve ref_bank.npz'i kurar (renkli+gri embedding). Modalite
        gerektirmez; segmenter/eslesme ilk karede modalite belli olunca baglanir."""
        if self._bank_cropped:
            return
        self._bank_cropped = True
        if not ref_image_paths:
            logging.info("[NURON] Referans yok → G3 kapali.")
            return
        try:
            paths = list(ref_image_paths.values())
            ref_dir = os.path.dirname(paths[0])
            self._ref_dir = ref_dir
            self._bank_path = os.path.join(KOSU.alt("offline_data"), "ref_bank_online.npz")
            os.makedirs(os.path.dirname(self._bank_path), exist_ok=True)
            # url -> stem (stem -> url DEGIL). Sunucu AYNI goruntuyu 2-3 ayri aralikta
            # ayri url ile isteyebilir; stem anahtarli sozluk yalniz SON url'yi tutuyordu,
            # onceki aralik(lar) hic islenmiyordu. E2E testte bulundu (2026-09-02):
            # Referans_Nesne_10 -> REF02 97-135 bos gitti, REF04 382-427 calisti.
            self._url_to_stem = {url: os.path.splitext(os.path.basename(p))[0]
                                 for url, p in ref_image_paths.items()}
            print(f"[NURON] {len(paths)} referans indi → HQ-SAM ile ANINDA crop + banka...")
            ReferenceSession.build_bank(ref_dir, out_path=self._bank_path, save_crops=True)
            logging.info(f"[NURON] Referans banka npz kuruldu: {self._bank_path}")
        except Exception as e:
            logging.error(f"[NURON] Referans crop/banka kurulamadi: {e}")
            self._bank_path = None

    # ---------------------------------------------------------------
    # ILK-KARE KURULUM
    # ---------------------------------------------------------------
    def _lazy_init(self, image_bgr, ref_image_paths):
        """Tum modelleri BIR KEZ yukle. YARISMA 2026: oturum SADECE RGB (resmi duyuru
        16.07.2026) → modalite otomatik-tespiti KALDIRILDI, dogrudan RGB sabit.

        YEREL TEST: NURON_MODALITY=termal verilirse termal boru hatti yuklenir.
        Ortam degiskeni YOKSA davranis DEGISMEZ (rgb) → yarisma akisi aynen korunur."""
        self.modality = os.environ.get("NURON_MODALITY", "rgb").strip().lower()
        if self.modality not in ("rgb", "termal"):
            self.modality = "rgb"
        _sabit = " (yarisma sabit — termal yok)" if self.modality == "rgb" else " (YEREL TEST)"
        logging.info(f"[NURON] Modalite: {self.modality.upper()}{_sabit}")
        print(f"[NURON] Oturum modalitesi: {self.modality.upper()}{_sabit} — modeller yukleniyor...")

        # G1: tespit + hareket
        self.pipe = DetectionPipeline(self.modality)
        # Gorev 1 hareket kurali v2 (2026-09-05): profil MODALITEYE gore secilir (MOTION_PROFILE[rgb|termal]),
        # kare is_w genisligine olceklenir (rgb 960 / termal 640). Acilista hangi profilin aktif oldugunu goster.
        _mp = self.pipe.motion.p
        print(f"[NURON] Hareket kurali v2 aktif — profil {self.modality}: is_w={_mp.get('is_w')} "
              f"perp={_mp['perp_floor']:.1f}px along={_mp['along_floor']:.1f}px (motion.py)")
        # G1: inis (PaDiM) — VRAM darsa NURON_PADIM_DEVICE=cpu (varsayilan cuda, davranis AYNI).
        # 2026-09-10: termal oturumda G3 HIBRIT (WebSSL + C-RADIOv3-H) tepe 5.7 GB; 8 GB karta
        # DPVO worker ile birlikte sigmiyordu (G1/G3 CUDA OOM). PaDiM iki modeli ~0.4 GB tutuyor
        # (cov_inv 196x448x448 x2) ve yalniz UAP/UAI kutusu varken cagriliyor -> CPU'da tutmak
        # kararlari DEGISTIRMEZ, yalnizca o kutularda ~50 ms ekler.
        _padim_dev = os.environ.get("NURON_PADIM_DEVICE", "cuda").strip().lower()
        if _padim_dev not in ("cuda", "cpu"):
            _padim_dev = "cuda"
        self.landing = LandingEstimator(self.modality, device=_padim_dev)
        if _padim_dev == "cpu":
            print("[NURON] PaDiM (inis) CPU'da — VRAM tasarrufu ~0.4 GB (NURON_PADIM_DEVICE=cpu)")
        # G2: GPS koprusu (DPVO worker AYRI surecte; yoksa graceful degrade)
        from src.client.gps_bridge import GpsBridge
        self.gps = GpsBridge(self.modality, enable_worker=True)
        # G3: referans oturumunu (modaliteye bagli segmenter/eslesme) bagla
        self._finalize_reference_session(ref_image_paths)
        print("[NURON] Tum modeller hazir. Oturum basliyor.")

    def _finalize_reference_session(self, ref_image_paths):
        """Modalite belli olunca ReferenceSession'i kur, bankayi yukle, url<->id esle.
        Banka daha once crop edilmediyse (prepare cagrilmadiysa) burada kurulur (yedek)."""
        self._ref_session_ready = True
        if not self._bank_cropped:                      # yedek: prepare cagrilmadi
            self.prepare_references(ref_image_paths)
        # v2 BANKASI NPZ DEGIL: build_bank yalnizca <kok>/ref_crops/<oid>.png yazar.
        # (v1 npz uretiyordu; npz varligini arayan eski kontrol referans oturumunu
        #  SESSIZCE kapatiyordu -> reference_predictions hep bos kaliyordu.)
        _crop = os.path.join(os.path.dirname(self._bank_path or ""), "ref_crops")
        if not self._bank_path or not (os.path.isdir(_crop) and os.listdir(_crop)):
            logging.error(f"[NURON] Referans kirpimi yok ({_crop}) -> G3 KAPALI.")
            print(f"[NURON] UYARI: referans kirpimi yok ({_crop}) -> Gorev 3 KAPALI.")
            self.ref = None
            return
        try:
            self.ref = ReferenceSession(self.modality, bank_path=self._bank_path, load_models=True)
            self.ref.load_bank(self._bank_path)
            # v2 kirpimlari <oid>.png olarak yazilir; url -> stem -> oid. stem -> oid
            # kurali build_bank ile AYNI kaynaktan (g3_v2.stem_oid) gelir. Birden fazla
            # url ayni oid'ye gidebilir (ayni nesne, ayri araliklar).
            self.url_to_id, self.id_to_url = {}, {}
            _idler = set(self.ref.ref_ids)
            try:
                _stem_oid = stem_oid(self._ref_dir) if self._ref_dir else {}
            except Exception as e:
                logging.warning(f"[NURON] stem_oid okunamadi ({e}) -> addaki sayi kullanilacak")
                _stem_oid = {}
            for url, stem in self._url_to_stem.items():
                rid = _stem_oid.get(stem, _oid(stem))
                if rid in _idler:
                    self.url_to_id[url] = rid
                    self.id_to_url.setdefault(rid, url)
            _eksik = sorted(set(self._url_to_stem) - set(self.url_to_id))
            _coklu = sorted({i for i in self.url_to_id.values()
                             if sum(1 for v in self.url_to_id.values() if v == i) > 1})
            logging.info(f"[NURON] Referans oturum hazir: ids={self.ref.ref_ids} "
                         f"eslesen_url={len(self.url_to_id)}/{len(self._url_to_stem)} "
                         f"| coklu-aralik id={_coklu or 'yok'}")
            if _eksik:
                logging.error(f"[NURON] ESLESMEYEN referans url'leri (bu araliklar BOS gider): {_eksik}")
        except Exception as e:
            logging.error(f"[NURON] Referans oturum kurulamadi: {e}")
            self.ref = None

    # ---------------------------------------------------------------
    # GPU KUYRUGU (2026-09-10) — OOM aninda GPS'i durdur, isi bitir, sonra devam
    # ---------------------------------------------------------------
    def _oom_kuyruk(self, ad, is_fn):
        """is_fn()'i calistirir; CUDA OOM olursa isi SIRAYA alir ve TEKRAR dener.

        Neden: kart 8 GB ve IKI surec paylasiyor (istemci G1+G3 tepe ~5.5 GB,
        DPVO worker oturum boyunca 0.6 -> 2.1 GB). Toplam kapasiteyi asinca
        istemcide OOM olur. `empty_cache()` yalniz KENDI onbellegini birakir;
        eksik bellek OTEKI surecte oldugu icin tek basina kurtarmaz
        (olculdu 2026-09-10: 130 tekrarin 130'u yine patladi).

        Kuyruk (hicbir model bosaltilmaz, hicbir kare atlanmaz):
          1) kendi PyTorch onbellegini birak
          2) GPS worker'ina "bellegini birak ve DUR" de (dosya-IPC), ONAY bekle
          3) isi TEKRAR dene -> basarili olursa kare tam sonuc verir
          4) ISTEK kalkar -> worker kaldigi yerden devam eder -> G2 calisir -> gonderim
        G3 calisirken worker o karenin girdisini henuz almamistir (girdi G2
        adiminda yazilir), yani durdurmak akisi YAVASLATMAZ.
        Kapatma: NURON_GPU_KUYRUK=0 -> eski davranis (empty_cache + tek tekrar)."""
        try:
            return is_fn()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            self._oom_n += 1
            with gps_durdur(ad) as k:
                logging.warning(f"[NURON] {ad} OOM #{self._oom_n} -> GPU kuyrugu: GPS durduruldu "
                                f"(onay={k.onay}, {k.gecen*1000:.0f} ms) -> is tekrarlaniyor")
                if self._oom_n <= 3 or self._oom_n % 25 == 0:
                    print(f"[NURON] {ad} OOM #{self._oom_n} -> GPU kuyrugu devrede "
                          f"(GPS durdu, onay={'VAR' if k.onay else 'YOK'}) -> tekrar", flush=True)
                try:
                    sonuc = is_fn()
                except torch.cuda.OutOfMemoryError:
                    self._oom_fail += 1
                    logging.error(f"[NURON] {ad} OOM: kuyruktan SONRA da yer yok "
                                  f"(kurtarilan {self._oom_ok} / kayip {self._oom_fail})")
                    raise
                self._oom_ok += 1
                logging.info(f"[NURON] {ad} kuyrukla KURTARILDI "
                             f"(kurtarilan {self._oom_ok} / kayip {self._oom_fail})")
                return sonuc

    # ---------------------------------------------------------------
    # HER KARE
    # ---------------------------------------------------------------
    def detect(self, prediction, health_status, active_refs=None, ref_image_paths=None, frame_image_path=None):
        active_refs = active_refs or []
        ref_image_paths = ref_image_paths or {}

        image_bgr = cv2.imread(frame_image_path)
        if image_bgr is None:
            logging.error(f"[NURON] Kare okunamadi: {frame_image_path} → bos tahmin.")
            return prediction

        # ---- ilk karede modelleri kur (referans bankasi genelde main.py'de
        #      prepare_references ile ONCEDEN crop edilmis olur) ----
        if self.pipe is None:
            self._lazy_init(image_bgr, ref_image_paths)

        # ---- sunucudan gelen kareyi KAYDET: frames/new'e yaz, onceki new → frames/old ----
        # (islenmekte olan kare new'de; SONRAKI kare gelince old'a gecer.)
        try:
            frame_key = os.path.splitext(os.path.basename(frame_image_path))[0]
            RT.save_new_frame(image_bgr, frame_key)
        except Exception as e:
            logging.error(f"[NURON] Kare kaydedilemedi (frames/new): {e}")

        # ============ GOREV 1 — Nesne Tespiti + Tasit Hareketi ============
        objs = []
        obstacle_boxes = []
        try:
            objs = self._oom_kuyruk("G1", lambda: self.pipe.process(image_bgr))   # schema.DetectedObject (motion dolu)
            for o in objs:
                if o.cls in OBSTACLE_CLS:                 # tasit/insan → inis engeli
                    obstacle_boxes.append((o.top_left_x, o.top_left_y,
                                           o.bottom_right_x, o.bottom_right_y))
        except Exception as e:
            logging.error(f"[NURON] G1 tespit hatasi: {e}")

        # ---- G1: UAP/UAI inis durumu ----
        try:
            for o in objs:
                if o.cls in (UAP_CLS, UAI_CLS) and self.landing is not None:
                    box = (o.top_left_x, o.top_left_y, o.bottom_right_x, o.bottom_right_y)
                    o.landing_status = self._oom_kuyruk(
                        "G1-inis", lambda: self.landing.status_for(image_bgr, box, o.cls, obstacle_boxes))
        except Exception as e:
            logging.error(f"[NURON] G1 inis hatasi: {e}")

        # ---- schema nesneleri → resmi DetectedObject (cls TUPLE, index+1 arayuzde) ----
        for o in objs:
            cls_tuple = (int(o.cls),)                     # arayuz cls[0] okur → classes/(idx+1)/
            d_obj = DetectedObject(
                cls_tuple,
                str(o.landing_status),                   # "1"/"0"/"-1"
                str(o.motion_status),                    # "1"/"0"/"-1"
                o.top_left_x, o.top_left_y,
                o.bottom_right_x, o.bottom_right_y,
            )
            prediction.add_detected_object(d_obj)

        # ---- G2 / G3 SIRASI ----
        # RGB  : yarismadaki ORIJINAL sira (G2 -> G3). 6.44/8 GB, rahat sigiyor. DOKUNULMADI.
        # TERMAL: G3 -> G2. Cunku G3'un kare segmenteri modaliteye bagli ve termalde
        #         SAM2 hiera-large (857 MB, RGB'deki CropFormer'in 4.5 kati) yukleniyor;
        #         G2'nin DPVO'su AYRI SURECTE ayni 8 GB VRAM'i paylasiyor ve OOM oluyor.
        #         G3 tepe kullanimini ONCE bitirip empty_cache ile PyTorch'un
        #         onbellegini surucuye geri verince DPVO yer buluyor.
        #         empty_cache MODELLERI BOSALTMAZ — agirliklar GPU'da yuklu kalir,
        #         sadece kullanilmayan ara-tampon bloklari geri verilir (ms mertebesi).
        #         Referans penceresi DISINDA G3 zaten calismaz -> ucu de rahat sigar.
        if self.modality == "termal":
            self._gorev3(prediction, image_bgr, active_refs)
            torch.cuda.empty_cache()      # onbellegi birak, DPVO (ayri surec) alsin
            self._gorev2(prediction, image_bgr, health_status, frame_key)
        else:
            self._gorev2(prediction, image_bgr, health_status, frame_key)
            self._gorev3(prediction, image_bgr, active_refs)

        self.frame_count += 1
        return prediction

    # ------------------------------------------------------------------
    def _gorev3(self, prediction, image_bgr, active_refs):
        """Referans Nesne Tespiti. G1 sonucu zaten eklendi -> burada hata/OOM
        olsa bile digerleri GONDERILIR, kare ATLANMAZ.

        Aralik <-> referans baglantisi ESNEK (2026-09-02): sunucu ayni referans
        goruntusunu 2-3 ayri aralikta (ayri url) isteyebilir. Bu yuzden
          (1) her aktif url KENDI id'sine cozulur (cok url -> 1 id),
          (2) kutu O ARALIGIN url'siyle gonderilir (id_to_url ile DEGIL),
          (3) aktif url kumesi degisince takip sifirlanir (id ayni kalsa bile)."""
        try:
            if self.ref is None or not active_refs:
                return
            aktif = [(r['url'], self.url_to_id[r['url']]) for r in active_refs
                     if r.get('url') in self.url_to_id]
            if not aktif:
                return
            anahtar = tuple(sorted(u for u, _ in aktif))
            if anahtar != self._son_aralik:
                self._son_aralik = anahtar
                self.ref.aralik_degisti()
                logging.info(f"[NURON] G3 yeni aralik {list(anahtar)} -> id "
                             f"{sorted({i for _, i in aktif})} (takip sifirlandi)")
            active_ids = sorted({i for _, i in aktif})
            is_fn = lambda: self.ref.process(image_bgr, active_ids=active_ids)
            if GPU_KUYRUK_PROAKTIF and not getattr(self, "_pro_kapali", False):
                matches = self._g3_proaktif(is_fn)                      # A: once worker'i durdur, sonra G3
            else:
                matches = self._oom_kuyruk("G3", is_fn)                 # eski: OOM olursa kuyruk
            for m in (matches or []):
                for url, rid in aktif:            # ayni nesne ayni anda 2 url'de ise ikisine de
                    if rid == int(m.object_id):
                        prediction.add_reference_prediction(ReferencePrediction(
                            url, prediction.frame_url,
                            m.top_left_x, m.top_left_y, m.bottom_right_x, m.bottom_right_y))
        except Exception as e:
            logging.error(f"[NURON] G3 referans hatasi: {e}")

    # ------------------------------------------------------------------
    def _g3_proaktif(self, is_fn):
        """A (2026-09-12) — PROAKTIF GPU KUYRUGU: G3'u calistirmadan ONCE GPS worker'ina ISTEK yaz
        (worker empty_cache + DUR, ONAY ~30 ms), G3 bitince ISTEK kalkar -> worker devam eder.
        Worker bu anda zaten BOS bekler (o karenin girdisi G2 adiminda yazilir), yani akis yavaslamaz;
        bedel yalnizca ONAY beklemesi + worker'in sonraki karede yeniden bellek ayirmasi.
        ONAY gelmezse (worker yok / yanit vermiyor) proaktif mod bu oturum icin KAPANIR, eski reaktif
        kuyruk (_oom_kuyruk) devam eder -> hic bir durumda eski surumden daha kotu degil.
        Proaktif durdurmaya ragmen OOM olursa: empty_cache + tek tekrar (worker zaten durmus)."""
        n = getattr(self, "_pro_n", 0) + 1; self._pro_n = n
        with gps_durdur("G3-proaktif") as k:
            if k.onay:
                if n == 1 or n % 250 == 0:
                    logging.info(f"[NURON] G3 proaktif kuyruk #{n}: worker durdu (onay {k.gecen*1000:.0f} ms)")
                    if n == 1:
                        print("[NURON] G3 PROAKTIF GPU kuyrugu AKTIF: pencere icinde her karede worker bellegini birakip bekler", flush=True)
                try:
                    return is_fn()
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    self._pro_oom = getattr(self, "_pro_oom", 0) + 1
                    logging.warning(f"[NURON] G3 proaktif kuyrukta YINE OOM #{self._pro_oom} -> empty_cache + tekrar")
                    return is_fn()
        # ONAY gelmedi: worker yok ya da yanit vermiyor -> proaktif kapat, eski yol
        self._pro_kapali = True
        logging.warning(f"[NURON] G3 proaktif kuyruk: worker ONAY vermedi ({k.gecen*1000:.0f} ms) -> proaktif KAPANDI, reaktif kuyruk devrede")
        print("[NURON] G3 proaktif kuyruk: worker yanit vermedi -> reaktif moda donuldu", flush=True)
        return self._oom_kuyruk("G3", is_fn)

    # ------------------------------------------------------------------
    def _gorev2(self, prediction, image_bgr, health_status, frame_key):
        """Konum Kestirimi.
        health_status: '1' (ilk ~450, GT var) / '0' (GPS yok) / None (ceviri yok)."""
        logging.info(f"[NURON] HEALTH frame#{self.frame_count} health_status={health_status}")
        try:
            if health_status is None:
                logging.info("[NURON] G2: health_status None → konum uretilmez.")
            elif health_status == '1':
                # GT var → kalibrasyonu besle + GT'yi AYNEN geri gonder (bedava puan).
                gt = None
                if None not in (prediction.gt_translation_x, prediction.gt_translation_y,
                                prediction.gt_translation_z):
                    gt = (float(prediction.gt_translation_x),
                          float(prediction.gt_translation_y),
                          float(prediction.gt_translation_z))
                if gt is not None and self.gps is not None:
                    world = self.gps.process(image_bgr, self.frame_count, gt, 1, frame_key=frame_key)
                    prediction.add_translation_object(DetectedTranslation(*world))
                elif gt is not None:
                    prediction.add_translation_object(DetectedTranslation(*gt))
                else:
                    logging.info("[NURON] G2: saglikli kare ama GT null → atlaniyor.")
            else:  # health_status == '0'  → kendi kestirimimiz ZORUNLU
                if self.gps is not None:
                    world = self.gps.process(image_bgr, self.frame_count, None, 0, frame_key=frame_key)
                    if world is not None:
                        prediction.add_translation_object(DetectedTranslation(*world))
        except Exception as e:
            logging.error(f"[NURON] G2 konum hatasi: {e}")
