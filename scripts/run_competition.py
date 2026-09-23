#!/usr/bin/env python3
"""
ANA DÖNGÜ — Konsol 2 (base env). Sunucudan kare çek → orkestratör → sonuç gönder.

İKİ KONSOL:
  Konsol 1 (dpvo env): python scripts/gps_worker.py --modality <rgb|termal>
  Konsol 2 (base env): python scripts/run_competition.py --modality <rgb|termal> [--mock]

Akış: authenticate → loop{ get_frame → indir → process_frame → send_result }.
Bozuk/kayıp kare → boş JSON gönder, geç. Bittiğinde (frame None) durur.
"""
import os
import sys
import time
import argparse

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.common.schema import FrameInfo, empty_result   # noqa: E402
from src.client.api_client import ApiClient, load_env    # noqa: E402
from src.client.orchestrator import Orchestrator         # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", required=True, choices=["rgb", "termal"])
    ap.add_argument("--mock_url", default="", help="yerel mock sunucu (örn http://127.0.0.1:5000)")
    ap.add_argument("--no_gps", action="store_true")
    ap.add_argument("--no_reference", action="store_true")
    ap.add_argument("--no_landing", action="store_true")
    ap.add_argument("--gps_no_worker", action="store_true",
                    help="dpvo worker'a bağlanma (degraded: health=1 GT echo)")
    ap.add_argument("--max_frames", type=int, default=0)
    args = ap.parse_args()

    env = load_env(os.path.join(ROOT, ".env"))
    if args.mock_url:
        env["EVALUATION_SERVER_URL"] = args.mock_url
    client = ApiClient(env).authenticate()
    print(f"[run] bağlandı: {client.base}  session={client.session}")

    orch = Orchestrator(
        args.modality,
        enable_gps=not args.no_gps,
        enable_reference=not args.no_reference,
        enable_landing=not args.no_landing,
        gps_enable_worker=not args.gps_no_worker,
    )

    n = 0; t0 = time.time()
    try:
        while True:
            frame = client.get_frame()
            if frame is None:
                print("[run] kare bitti.")
                break
            info = FrameInfo.from_json(frame)
            try:
                img = client.fetch_image(info.image_url)
                if img is None:
                    raise ValueError("görüntü indirilemedi")
                result = orch.process_frame(info, img)
            except Exception as e:                      # bozuk/kayıp kare → boş gönder
                print(f"[run] kare {n} hata ({e}) → boş sonuç")
                result = empty_result(info.url)
            ok = client.send_result(result)
            tm = orch.last_timing
            if n % 25 == 0:
                print(f"[run] kare {n} gönderildi ok={ok} "
                      f"süre={tm.get('total',0)*1000:.0f}ms "
                      f"(detect={tm.get('detect',0)*1000:.0f} ref={tm.get('reference',0)*1000:.0f} "
                      f"gps={tm.get('gps',0)*1000:.0f} landing={tm.get('landing',0)*1000:.0f})",
                      flush=True)
            n += 1
            if args.max_frames and n >= args.max_frames:
                break
    finally:
        orch.close()
        dt = time.time() - t0
        print(f"[run] BİTTİ: {n} kare, {dt:.1f}s, ort {1000*dt/max(n,1):.0f}ms/kare")


if __name__ == "__main__":
    main()
