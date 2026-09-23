#!/usr/bin/env python3
"""
FINAL-TERMAL GPS — worker'siz/sunucusuz OFFLINE tekrar-oynatma testi (dpvo env; GPU).
Gercek kareler diskten sirayla; gps_worker ile AYNI DPVO yolu (K ornek, tekrar-kare atlama, gercek indeks tstamp, damping 1.0) ve
gps_bridge ile AYNI kestirim yolu (TermalKopru). health=1: ilk 450 + --olaylar. GPS-kapali karelerde eksen hatalarini yazar.
Beklenen (canli_replay ile): 2026 1400+1800 ~ X 1.7 Y 13.6 Z 1.9 ; ot4 1400+1800 ~ 1.3 / 1.3 / 1.7.
    conda run -n dpvo python -u scripts/test_gps_offline_termal.py --ad 2026 --olaylar 1400,1800 [--limit N] [--K 2]
"""
import os, sys, csv, time, argparse
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.task2_position import paths as P                    # noqa: E402
sys.path.insert(0, P.DPVO_REPO)
import cv2                                                    # noqa: E402
from src.task2_position.dpvo_runner import DPVOTracker        # noqa: E402
from src.client.gps_bridge import TermalKopru                 # noqa: E402
# Dataset root. Override with NURON_SERVER_ROOT; default reproduces the original layout.
SERVER = os.environ.get("NURON_SERVER_ROOT", os.path.expanduser("~/Desktop/server"))
VERI = {"2026": f"{SERVER}/oturum_termal/2026", "ot4": f"{SERVER}/oturum_termal/2025_oturum_4",
        "ornek1": f"{SERVER}/no_gps/2025_ornek_1", "ornek2": f"{SERVER}/no_gps/2025_ornek_2"}
G = 450

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ad", default="2026"); ap.add_argument("--olaylar", default="1400,1800"); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--K", type=int, default=0)
    a = ap.parse_args(); kok = VERI[a.ad]; prof = P.GPS_PROFILE["termal"]
    K = a.K or int(prof.get("K", 1)); esik = float(prof.get("tekrar_esik", 0.0))
    files = sorted(f for f in os.listdir(os.path.join(kok, "frames")) if f.lower().endswith((".jpg", ".png", ".webp")))
    gt = np.array([[float(r["translation_x"]), float(r["translation_y"]), float(r["translation_z"])] for r in csv.DictReader(open(os.path.join(kok, "gps.csv")))])
    n = min(len(files), len(gt)); n = min(n, a.limit) if a.limit else n; gt = gt[:n]; olaylar = set(int(x) for x in a.olaylar.split(",") if x)
    trackers = [DPVOTracker("termal", ckpt=P.DPVO_CKPT("termal"), calib=P.CALIB("termal"), cfg_file=P.DPVO_CFG) for _ in range(K)]
    tk = TermalKopru(gt_frames=G); pred = np.full((n, 3), np.nan); son_satir = None; t0 = time.time(); n_tekrar = 0
    print(f"[test] {a.ad} K={K} tekrar_esik={esik} olaylar={sorted(olaylar)} kare={n}", flush=True)
    for i in range(n):
        img = cv2.imread(os.path.join(kok, "frames", files[i])); h = 1 if (i < G or i in olaylar) else 0
        tekrar = esik > 0 and trackers[0].tekrar_mi(img, esik)
        if tekrar and son_satir is not None:
            satir = son_satir.copy(); satir[0] = 1.0; n_tekrar += 1
        else:
            bl = []
            for k, tr in enumerate(trackers):
                xyz = tr.feed(img, tstamp=i, seed=k * 100003 + i); xyz = np.zeros(3) if xyz is None else xyz
                d_med, n_cam, rms = tr.ek_cikti(); bl.append(np.concatenate([np.asarray(xyz, float), [d_med], np.asarray(n_cam, float), [rms]]))
            satir = np.concatenate([[0.0]] + bl); son_satir = satir.copy()
        out = tk.adim(satir, img, gt[i] if h else None, h, i)
        pred[i] = gt[i] if h else (out if out is not None else pred[i - 1])
        if (i + 1) % 250 == 0:
            e = pred[i] - gt[i]; print(f"  {i+1}/{n} ({time.time()-t0:.0f} sn) hata {e[0]:+6.1f} {e[1]:+6.1f} {e[2]:+5.1f}  ens uyum {np.round(tk.ens.uyum,2) if tk.ens else '-'}", flush=True)
    m = np.ones(n, bool); m[:G] = False; m[list(olaylar & set(range(n)))] = False; m &= np.isfinite(pred).all(1)
    e = np.abs(pred[m] - gt[m]).mean(0); e3 = np.linalg.norm(pred[m] - gt[m], axis=1).mean()
    print(f"[test] kestirici: {tk.kestirici_bilgi()}  tekrar kare: {n_tekrar}", flush=True)
    print(f"[SONUC] {a.ad} K={K} olaylar={sorted(olaylar)} GPS-kapali {int(m.sum())} kare: X {e[0]:.2f}  Y {e[1]:.2f}  Z {e[2]:.2f}  (3B {e3:.2f})  {time.time()-t0:.0f} sn, {1000*(time.time()-t0)/n:.0f} ms/kare", flush=True)
    out_csv = os.path.join(ROOT, "logs", f"test_gps_offline_termal_{a.ad}_{a.olaylar.replace(',', '+') or 'yok'}.csv"); os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    np.savetxt(out_csv, np.column_stack([np.arange(n), pred, gt[:n]]), fmt="%.4f", header="kare px py pz gx gy gz"); print("[test] csv:", out_csv)

if __name__ == "__main__":
    main()
