import logging
import os
import time
from datetime import datetime
from pathlib import Path
from decouple import config
from tqdm import tqdm
# NURON: arayuz paketi connect/iface altina alindi (kendi src/ ile cakismamasi icin).
# Protokol mantigi (auth/gating/resume/rate-limit) DEGISTIRILMEDI.
from connect.iface.connection_handler import ConnectionHandler
from connect.iface.frame_predictions import FramePredictions
from connect.iface.object_detection_model import ObjectDetectionModel
from src.common import kosu as KOSU              # NURON: her kosu kendi klasorune (kosular/folder<N>)

# Minimum seconds between frames. Keeps GET /frames/ well under the 300/m rate limit
# even when image downloads are fast. 300/m = 5/s; 0.25s floor → max ~4/s with headroom.
MIN_FRAME_INTERVAL = 0.25


def configure_logger(team_name):
    log_folder = KOSU.alt("_logs") + "/"        # NURON: kosular/folder<N>/_logs/
    Path(log_folder).mkdir(parents=True, exist_ok=True)
    log_filename = datetime.now().strftime(log_folder + team_name + '_%Y_%m_%d__%H_%M_%S_%f.log')
    logging.basicConfig(filename=log_filename, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')


def run():
    print("Started...")
    # .env connect/config/ altinda (NURON_DRONE kokunden `python -m connect.main` calisir).
    config.search_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")
    team_name = config('TEAM_NAME')
    password = config('PASSWORD')
    evaluation_server_url = config("EVALUATION_SERVER_URL")

    configure_logger(team_name)

    detection_model = ObjectDetectionModel(evaluation_server_url)

    server = ConnectionHandler(evaluation_server_url, username=team_name, password=password)

    # Check where we left off — safe to call after reconnect.
    # get_progress() returns None ONLY when the server is unreachable (after retries);
    # a genuine "no active session" comes back as a dict with session_name == None.
    progress = server.get_progress()
    if progress is None:
        print("Could not reach the evaluation server (progress check failed). "
              "Check your connection and try again.")
        return
    if not progress['session_name']:
        print("No active session found. Exiting.")
        return

    if progress['completed']:
        print(f"All {progress['total_frames']} frames already submitted. Nothing to do.")
        return

    session_name = progress['session_name']
    total_frames = progress['total_frames']
    start_index = progress['frame_index']
    print(f"Session: {session_name} — resuming from frame {start_index + 1} of {total_frames}")

    # Prepare image storage
    server.video_name = session_name + "/"
    server.create_img_folder(server.video_name)
    images_folder = os.path.join(server.img_save_path, server.video_name)

    references_folder = os.path.join(images_folder, "references") + os.sep
    Path(references_folder).mkdir(parents=True, exist_ok=True)

    # Gorev 3: TUM referans nesnelerini ve [frame_start, frame_end] araliklarini en
    # bastan TEK seferde cek (GET /reference/, 5/m limit).
    all_references = server.get_reference_objects(force_download=True) or []
    logging.info(f"Loaded {len(all_references)} reference object(s) for the session.")
    # NURON: references.json geldi mi + beklenen anahtarlar (frame_start/end_image_url) var mi?
    # Canli testte sunucu semasini DOGRULAMAK icin (komite 2026-07-14). Yanlis isim gelirse UYARIR.
    if all_references:
        _keys = sorted(all_references[0].keys())
        _win = all(("frame_start_image_url" in r and "frame_end_image_url" in r)
                   for r in all_references)
        _msg = (f"[NURON-REF] references.json: {len(all_references)} ref | anahtarlar={_keys} | "
                f"pencere_alanlari={'VAR' if _win else 'YOK'}")
        logging.info(_msg); print(_msg)
        if not _win:
            _w = ("[NURON-REF] UYARI: frame_start_image_url/frame_end_image_url yok — "
                  "active_refs bos kalabilir, references.json semasini kontrol et!")
            logging.warning(_w); print(_w)
    else:
        logging.info("[NURON-REF] references.json bos/gelmedi.")
    ref_image_paths = {}

    # Gorev 3: TUM referans goruntularini pencereyi beklemeden en bastan indir.
    # NOT: Bu yalnizca sunucu media-auth kapisi referans goruntulerini ilerlemeden
    # bagimsiz actiysa calisir; aksi halde penceresi henuz baslamayan referanslar
    # icin sunucu 403 doner. Her indirme bir media-auth alt-istegi uretir (300/m
    # limit); referans sayisi cok yuksekse bu baslangic patlamasini limit altinda tutun.
    for ref in all_references:
        ref_image_url = (ref['image_url'] if ref['image_url'].startswith('http')
                         else evaluation_server_url + "media" + ref['image_url'])
        detection_model.download_image(ref_image_url, references_folder,
                                       os.listdir(references_folder),
                                       auth_token=server.auth_token)
        ref_image_paths[ref['url']] = references_folder + ref_image_url.split("/")[-1]
    logging.info(f"Pre-downloaded {len(ref_image_paths)} reference image(s) up front.")

    # NURON: referanslar iner inmez HEMEN crop et + banka kur (kare dongusunden ONCE).
    detection_model.prepare_references(ref_image_paths)

    # tqdm progress bar — mutlak yarisma ilerlemesi (resume'de gercek yuzdeyi gosterir;
    # orn. 1000'in 500'unden devam edince %50'den baslar). ETA degismez (kalan = total - n).
    stuck_image_url = None
    stuck_count = 0
    with tqdm(total=total_frames, initial=start_index, desc="Frames") as pbar:
        while True:
            frame_start = time.monotonic()
            # Sunucu her çağrıda yalnızca sıradaki tek kareyi döner.
            frame = server.get_current_frame()
            if frame is None:
                # NURON: frame None IKI anlama gelir → (a) internet KESILDI, (b) oturum BITTI.
                # get_progress ile ayır: gerçekten bittiyse ÇIK; internet gittiyse BEKLE ve
                # tekrar dene (connect.main ÖLMESIN → gps_worker canlı kalır, DPVO sürekliliği
                # korunur, RESUME glitch OLMAZ). Kullanıcı testinde 1 dk kesinti sonrası GPS
                # saçmalıyordu; bu bekle-devam et döngüsü onu çözer (kare akışını bekler gibi).
                prog = server.get_progress()
                if prog is not None and prog.get('completed'):
                    print("Session complete.")
                    break
                if prog is None:
                    logging.warning("[NURON] Sunucuya ulasilamiyor (internet kesintisi?) — "
                                    "10s bekle, connect.main ACIK kalir, tekrar dene.")
                    print("Baglanti yok — bekleniyor (connect.main kapanmiyor)...")
                    time.sleep(10)
                else:
                    # oturum aktif ama kare gelmedi (gecici) — kısa bekle, tekrar dene
                    time.sleep(2)
                continue

            if frame['image_url'] == stuck_image_url:
                stuck_count += 1
                if stuck_count >= 5:
                    logging.error(f"Frame {frame['image_url']} did not advance after "
                                  f"{stuck_count} submissions; aborting to avoid an infinite loop.")
                    print("Aborting: current frame is not advancing (check logs).")
                    break
            else:
                stuck_image_url = frame['image_url']
                stuck_count = 0

            translation = server.get_current_translation()
            if translation is None:
                logging.warning("Translation unavailable for current frame; "
                                "submitting a detection-only prediction to advance.")
                health_status = None
                gt_x = gt_y = gt_z = None
            else:
                health_status = translation['health_status']
                gt_x = translation['translation_x']
                gt_y = translation['translation_y']
                gt_z = translation['translation_z']

            images_files = os.listdir(images_folder)

            # Bu kare icin aktif (pencere-ici) referanslar. Goruntuleri yukarida en
            # bastan toptan indirildigi icin burada ayrica indirme yapilmaz; bu liste
            # yalnizca asagidaki detect() cagrisina iletilir.
            active_refs = [
                r for r in all_references
                if r.get('frame_start_image_url') and r.get('frame_end_image_url')
                and r['frame_start_image_url'] <= frame['image_url'] <= r['frame_end_image_url']
            ]

            predictions = FramePredictions(
                frame['url'], frame['image_url'], frame['video_name'],
                gt_x, gt_y, gt_z
            )

            predictions = detection_model.process(
                predictions, evaluation_server_url, health_status,
                images_folder, images_files,
                active_refs=active_refs,
                ref_image_paths=ref_image_paths,
                auth_token=server.auth_token,
            )

            server.send_prediction(predictions)
            pbar.update(1)

            elapsed = time.monotonic() - frame_start
            if elapsed < MIN_FRAME_INTERVAL:
                time.sleep(MIN_FRAME_INTERVAL - elapsed)


if __name__ == '__main__':
    run()
