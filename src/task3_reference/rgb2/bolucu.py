#!/usr/bin/env python3
"""FastSAM-x bolme + AYRIK bolutuleme + temas grafigi.

FastSAM boya ustune boya yapar: olculdu, piksel basina ORTALAMA 3.00 maske
katmani (bazi karelerde 7). Bu, arama agacini kendi kopyalariyla dolduruyordu.
`ayrik()` kareyi bolutuler: bir piksel yalnizca TEK nesneye ait olur.
Kural: KUCUK nesne kazanir, buyuge ARTAN kalir, tamamen yutulan dusulur.
Sonuc: ic-ice sayaci 331 -> 0, ortalama komsu sayisi 3.2-5.0.
"""
import numpy as np, cv2
from src.task3_reference.rgb2 import ayarlar as A

_MODEL = [None]


def _model():
    if _MODEL[0] is None:
        from ultralytics import FastSAM
        _MODEL[0] = FastSAM(A.FASTSAM_CKPT)
    return _MODEL[0]


def kutu(m):
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1)


def bol(img):
    """kare -> maske listesi (ayrik, minimum alan suzgecinden gecmis)"""
    r = _model()(img, device=A.DEVICE, retina_masks=True, imgsz=A.MAXSIDE,
                 conf=A.FASTSAM_CONF, iou=A.FASTSAM_IOU, verbose=False)
    if not r or r[0].masks is None:
        return []
    H, W = img.shape[:2]
    esik = max(50, int(A.MIN_ORAN * H * W))
    ms = []
    for m in r[0].masks.data.cpu().numpy():
        mm = m > 0.5
        if mm.shape != (H, W):
            mm = cv2.resize(mm.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
        if int(mm.sum()) >= esik:
            ms.append(mm)
    if A.AYRIK:
        ms = ayrik(ms, (H, W), esik)
    return [m for m in ms if kutu(m) is not None]


def ayrik(ms, sekil, esik):
    """Bir piksel = TEK nesne. Kucuk kazanir, buyuge artan kalir."""
    if not ms:
        return ms
    H, W = sekil
    sahip = np.zeros((H, W), bool)
    out = []
    for i in sorted(range(len(ms)), key=lambda i: int(ms[i].sum())):
        m = ms[i] & ~sahip
        if int(m.sum()) < esik:
            continue
        sahip |= m
        out.append(m)
    return out


def delik_kapa(m):
    """Nesnenin ICI ASLA DELIK KALMAZ — disariya acilmayan her bosluk nesnenindir."""
    u = (~m).astype(np.uint8)
    n, lab = cv2.connectedComponents(u)
    if n <= 1:
        return m
    dis = set(lab[0, :].tolist()) | set(lab[-1, :].tolist()) | \
          set(lab[:, 0].tolist()) | set(lab[:, -1].tolist())
    return m | (~np.isin(lab, list(dis)) & ~m)


def temas(ms):
    """Komsuluk grafigi. Ic-ice olanlar komsu SAYILMAZ; her dugumde
    yalnizca EN YAKIN KOM_UST komsu tutulur (merkez uzakligina gore)."""
    n = len(ms)
    ku = [kutu(m) or (0, 0, 0, 0) for m in ms]
    al = [max(int(m.sum()), 1) for m in ms]
    gen = [cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) for m in ms]
    out = {i: [] for i in range(n)}
    for i in range(n):
        xi1, yi1, xi2, yi2 = ku[i]
        for j in range(i+1, n):
            xj1, yj1, xj2, yj2 = ku[j]
            if xj2 < xi1-6 or xj1 > xi2+6 or yj2 < yi1-6 or yj1 > yi2+6:
                continue
            if not (gen[i] & ms[j]).any():
                continue
            if float((ms[i] & ms[j]).sum()) / min(al[i], al[j]) >= A.IC_ESIK:
                continue
            d = -float(np.hypot((xi1+xi2)/2-(xj1+xj2)/2, (yi1+yi2)/2-(yj1+yj2)/2))
            out[i].append((j, d)); out[j].append((i, d))
    for i in range(n):
        out[i] = [j for j, _ in sorted(out[i], key=lambda x: -x[1])[:A.KOM_UST]]
    for i in range(n):                       # simetriklestir
        for j in out[i]:
            if i not in out[j]:
                out[j].append(i)
    return out


def kirp_gen(img, m, k, gen):
    """Kutuyu ve maskeyi `gen` px genislet; olcut kutusu DEGISMEZ."""
    if not gen:
        return img[k[1]:k[3], k[0]:k[2]], m[k[1]:k[3], k[0]:k[2]]
    H, W = img.shape[:2]
    x1, y1 = max(0, int(k[0])-gen), max(0, int(k[1])-gen)
    x2, y2 = min(W, int(k[2])+gen), min(H, int(k[3])+gen)
    mm = cv2.dilate(m[y1:y2, x1:x2].astype(np.uint8),
                    np.ones((2*gen+1, 2*gen+1), np.uint8)).astype(bool)
    return img[y1:y2, x1:x2], mm


def girdi(c, m):
    """crop -> modele giren goruntu: HAM RENK, maske disi SIYAH, kareye siyahla doldur.
    (gri tonlama YOK — halisahada IoU 0.17 -> 0.93 farki bundandi)"""
    mm = m
    if mm is None or mm.all():
        mm = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY) > 10
        if mm.mean() < 0.02:
            mm = np.ones(c.shape[:2], bool)
    g = c.copy(); g[~mm] = 0
    h, w = g.shape[:2]; S = max(h, w)
    kare = np.zeros((S, S, 3), np.uint8)
    y, x = (S-h)//2, (S-w)//2
    kare[y:y+h, x:x+w] = g
    return kare
