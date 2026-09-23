#!/usr/bin/env python3
"""
FINAL-11 RGB GPS — worker'siz OFFLINE tekrar-oynatma testi (base env).
Gercek kareler (ORB ObjectSizeZ) + onceden kosulmus DPVO ciktilari (rgb_gps cache) ile FinalKestirici'yi
kare kare besler; GPS-kapali karelerde eksen hatalarini yazar. Beklenen (2026, GPS 1400+1800): ~X 1.7 Y 1.4 Z 1.3 m.
    python3 scripts/test_gps_offline.py --ad 2026 --olaylar 1400,1800 [--limit N]
Veri: /home/adem/Desktop/server/oturum_rgb/<oturum>/frames + gps.csv ; DPVO cache: rgb_gps/trying/cache/traj_<ad>_derin.txt,
      rgb_gps/work/v3/cache/derin_<ad>_derin.npy (yama derinligi), rgb_gps/work/v5/cache/poz_<ad>.npz (yer-duzlemi normali).
"""
import os, sys, csv, time, argparse
import numpy as np, cv2
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.task2_position.final_kestirici import FinalKestirici                       # noqa: E402
from src.task2_position.z_termal import ObjectSizeZ                                 # noqa: E402
# Dataset roots. Override with NURON_GPS_ROOT / NURON_SERVER_ROOT; defaults reproduce the
# original development layout. The TEKNOFEST datasets themselves are not redistributable.
RGB    = os.environ.get("NURON_GPS_ROOT",    os.path.expanduser("~/Desktop/gps/rgb_gps"))
SERVER = os.environ.get("NURON_SERVER_ROOT", os.path.expanduser("~/Desktop/server"))
VERI = {"2026": f"{SERVER}/oturum_rgb/2026", "ot2": f"{SERVER}/oturum_rgb/2025_oturum_2", "ot3": f"{SERVER}/oturum_rgb/2025_oturum_3",
        "ornek1": f"{SERVER}/no_gps/2025_ornek_1", "ornek2": f"{SERVER}/no_gps/2025_ornek_2", "ot1": f"{SERVER}/no_gps/2025_oturum_1"}
G = 450

def orb_kare(img, w=960):
    h, ww = img.shape[:2]
    return img if ww == w else cv2.resize(img, (w, int(h * w / ww)), interpolation=cv2.INTER_AREA)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ad", default="2026"); ap.add_argument("--olaylar", default="1400,1800"); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--koruma", default="", help="anahtar=deger,... (cfg['koruma'] gecici override)")
    a = ap.parse_args(); kok = VERI[a.ad]
    files = sorted(f for f in os.listdir(os.path.join(kok, "frames")) if f.lower().endswith((".jpg", ".png", ".webp")))
    gt = np.array([[float(r["translation_x"]), float(r["translation_y"]), float(r["translation_z"])] for r in csv.DictReader(open(os.path.join(kok, "gps.csv")))])
    traj = np.loadtxt(os.path.join(RGB, "trying", "cache", f"traj_{a.ad}_derin.txt"))[:, 1:4]
    D = np.load(os.path.join(RGB, "work", "v3", "cache", f"derin_{a.ad}_derin.npy"))[:, 1]
    pz = os.path.join(RGB, "work", "v5", "cache", f"poz_{a.ad}.npz")
    if not os.path.exists(pz): pz = os.path.join(RGB, "work", "v5", "cache", f"poz2_{a.ad}.npz")
    poz = np.load(pz)
    n = min(len(files), len(gt), len(traj)); n = min(n, a.limit) if a.limit else n
    olaylar = set(int(x) for x in a.olaylar.split(",") if x)
    zf = ObjectSizeZ(f=694.85, gt_frames=G); zf.CLIP_LO, zf.CLIP_HI = 0.98, 1.02; zf.MED_W = 3; zf.NEDENSEL = True; zf.OFSET, zf.OFSET_N = "son", 10
    fk = FinalKestirici(G=G); pred = np.full((n, 3), np.nan); gz_kal = None; t0 = time.time()
    for kv in [x for x in a.koruma.split(",") if x]:
        k, v = kv.split("="); fk.cfg["koruma"][k] = float(v)
    for i in range(n):
        img = cv2.imread(os.path.join(kok, "frames", files[i])); h = 1 if (i < G or i in olaylar) else 0
        zf.feed(orb_kare(img))
        zA = None; zA_kal = None
        if i >= G:
            if gz_kal is None:                              # kalibrasyon bitti: ORB h0 + zA dizisi
                gz_kal = gt[:G, 2]
                try:
                    zf.calibrate_h0(gt[:G, :2]); zA_kal = zf.z(gz_kal)[:G]
                except Exception as e:
                    print("[test] ORB h0 kalibre olmadi:", e)
            if zf.h0 is not None:
                zA = float(zf.z(gz_kal)[-1])
        out = fk.adim(traj[i] if np.all(np.isfinite(traj[i])) else None, d_med=D[i] if np.isfinite(D[i]) else None,
                      n_cam=poz["n_cam"][i] if i < G else None, plan_rms=poz["plan_rms"][i] if i < G else None,
                      zA_ham=zA, gt_xyz=gt[i] if h else None, health=h, zA_kal=zA_kal)
        if out is not None: pred[i] = out
        if (i + 1) % 500 == 0: print(f"  {i+1}/{n} ({time.time()-t0:.0f} sn)", flush=True)
    m = np.ones(n, bool); m[:G] = False; m[list(olaylar & set(range(n)))] = False; m &= np.isfinite(pred).all(1)
    e = np.abs(pred[m] - gt[m]).mean(0); e3 = np.linalg.norm(pred[m] - gt[m], axis=1).mean()
    print(f"[{a.ad}] olaylar {sorted(olaylar)}: X {e[0]:.2f} Y {e[1]:.2f} Z {e[2]:.2f} 3B {e3:.2f} | nadir={fk.nadir} aci {fk.aci:.1f} rms {fk.rms:.3f} | saglik {fk.saglik} | uyari {fk.uyarilar}")
    np.savez(os.path.join(ROOT, "logs", f"test_gps_offline_{a.ad}.npz"), pred=pred, gt=gt[:n], olaylar=sorted(olaylar))

if __name__ == "__main__":
    main()
