#!/usr/bin/env python3
"""
TAM UÇTAN UCA TEST (offline, base env) — orkestratörü örnek video üzerinde 2250 kare
(stride-4) koşturur, GPS worker'ı (dpvo env) DIŞARIDAN çalışır varsayar.

Kaydeder (results/<run>/):
  frames/{i}.jpg   ← tespit çizili kare (kırmızı=hareketli, yeşil=durgun, mavi=uap/uai, sarı=referans)
  json/{i}.json    ← o karenin sunucu-çıktısı (Şekil 17)
  gps.csv          ← i, health, pred_xyz, gt_xyz  (GPS grafiği için)
  timing.csv       ← i, total, detect, reference, landing, gps

Çalıştırma:
  # Konsol 1 (dpvo env):  python scripts/gps_worker.py --modality rgb
  # Konsol 2 (base env):  python scripts/full_test.py --modality rgb --video ... --gt ... [--refs DIR]
"""
import sys, os, time, json, argparse, csv
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import cv2, numpy as np
from src.common.schema import FrameInfo
from src.client.orchestrator import Orchestrator

COL = {"mov": (0, 0, 255), "still": (0, 200, 0), "marker": (255, 128, 0),
       "ref": (0, 255, 255), "person": (200, 200, 0)}


def draw(img, res):
    for o in res["detected_objects"]:
        cls = int(o["cls"]); x1, y1 = int(o["top_left_x"]), int(o["top_left_y"])
        x2, y2 = int(o["bottom_right_x"]), int(o["bottom_right_y"])
        if cls == 0:
            c = COL["mov"] if o["motion_status"] == "1" else COL["still"]
            tag = ("HAREKET" if o["motion_status"] == "1" else "durgun")
        elif cls == 1:
            c = COL["person"]; tag = "insan"
        else:
            c = COL["marker"]
            ls = o["landing_status"]
            tag = ("UAP" if cls == 2 else "UAI") + (" inis+" if ls == "1" else " inis-" if ls == "0" else "")
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)
        cv2.putText(img, tag, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
    for u in res["detected_undefined_objects"]:
        x1, y1 = int(u["top_left_x"]), int(u["top_left_y"])
        x2, y2 = int(u["bottom_right_x"]), int(u["bottom_right_y"])
        cv2.rectangle(img, (x1, y1), (x2, y2), COL["ref"], 3)
        cv2.putText(img, f"REF#{u['object_id']}", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COL["ref"], 2)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", required=True, choices=["rgb", "termal"])
    ap.add_argument("--video", required=True)
    ap.add_argument("--gt", default="")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--gt_frames", type=int, default=450)
    ap.add_argument("--mid_gt", type=int, default=1800, help="orta-oturum health=1 GT karesi (drift reset)")
    ap.add_argument("--total", type=int, default=2250)
    ap.add_argument("--save_every", type=int, default=1, help="kaç karede bir frame görseli kaydet")
    ap.add_argument("--no_gps", action="store_true")
    ap.add_argument("--no_reference", action="store_true")
    ap.add_argument("--gps_no_worker", action="store_true")
    ap.add_argument("--landing_device", default="cuda")
    ap.add_argument("--out", default="results/run_rgb")
    args = ap.parse_args()

    outd = os.path.join(ROOT, args.out)
    for sub in ("frames", "json"):
        os.makedirs(os.path.join(outd, sub), exist_ok=True)

    gts = []
    if args.gt and os.path.exists(args.gt):
        with open(args.gt) as f:
            r = csv.reader(f)
            for row in r:
                try:
                    gts.append([float(row[0]), float(row[1]), float(row[2])])
                except Exception:
                    continue
    print(f"[full] GT satır={len(gts)}", flush=True)

    orch = Orchestrator(args.modality, enable_gps=not args.no_gps,
                        enable_reference=not args.no_reference, enable_landing=True,
                        gps_enable_worker=not args.gps_no_worker,
                        landing_device=args.landing_device)

    cap = cv2.VideoCapture(args.video)
    gcsv = open(os.path.join(outd, "gps.csv"), "w"); gw = csv.writer(gcsv)
    gw.writerow(["i", "health", "px", "py", "pz", "gx", "gy", "gz"])
    tcsv = open(os.path.join(outd, "timing.csv"), "w"); tw = csv.writer(tcsv)
    tw.writerow(["i", "total", "detect", "reference", "landing", "gps"])

    raw = 0; t0 = time.time(); nref = 0
    for i in range(args.total):
        cap.set(cv2.CAP_PROP_POS_FRAMES, raw)
        ok, img = cap.read()
        if not ok:
            break
        gt_true = gts[raw] if raw < len(gts) else None      # gerçek GT (kıyas için — her zaman)
        # ZAMANLAMA: health=1 sadece ilk gt_frames + mid_gt karesinde; diğerlerinde health=0
        is_gt = (i < args.gt_frames or i == args.mid_gt) and gt_true is not None
        health = 1 if is_gt else 0
        if health == 1:
            tx, ty, tz = gt_true                            # DOĞRU GT gönder
        else:
            # health=0 → sunucu YANLIŞ/bozuk gps yollar; kod health flag'e bakıp KULLANMAMALI
            tx, ty, tz = 999.0, -999.0, 888.0
        info = FrameInfo.from_json({
            "url": f"/frames/{i}/", "image_url": "x", "video_name": f"v_{i}", "session": "s",
            "translation_x": tx, "translation_y": ty, "translation_z": tz,
            "gps_health_status": health})
        res = orch.process_frame(info, img)
        tm = orch.last_timing

        # json kaydet
        with open(os.path.join(outd, "json", f"{i:05d}.json"), "w") as jf:
            json.dump(res, jf, ensure_ascii=False)
        # frame çiz + kaydet
        if i % args.save_every == 0:
            vis = draw(img.copy(), res)
            cv2.putText(vis, f"kare {i} raw{raw} h={health}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
            cv2.imwrite(os.path.join(outd, "frames", f"{i:05d}.jpg"), vis)
        # gps kaydet: i, health, gönderilen-pred, GERÇEK GT (kıyas için hep yazılır)
        tr = res["detected_translations"]
        p = (tr[0]["translation_x"], tr[0]["translation_y"], tr[0]["translation_z"]) if tr else ("", "", "")
        gg = gt_true if gt_true else ("", "", "")
        gw.writerow([i, health, *p, *gg])
        tw.writerow([i, f"{tm.get('total',0):.3f}", f"{tm.get('detect',0):.3f}",
                     f"{tm.get('reference',0):.3f}", f"{tm.get('landing',0):.3f}", f"{tm.get('gps',0):.3f}"])
        if res["detected_undefined_objects"]:
            nref += 1
        if i % 50 == 0:
            gcsv.flush(); tcsv.flush()
            el = time.time() - t0
            print(f"[full] {i}/{args.total} raw{raw} total={tm.get('total',0)*1000:.0f}ms "
                  f"ref_kare={nref} ({el:.0f}s, {el/(i+1)*1000:.0f}ms/kare ort)", flush=True)
        raw += args.stride

    gcsv.close(); tcsv.close(); orch.close()
    print(f"[full] BITTI: {i+1} kare, {time.time()-t0:.0f}s, referans-eşleşen kare={nref} → {outd}", flush=True)


if __name__ == "__main__":
    main()
