#!/usr/bin/env python3
"""TURLU ARAMA — tek karede referansin karsiligini bulur.

tur 0 : butun ayrik tekiller TUR0_TAVAN'da taranir (isi: yalnizca siralama)
        -> CESITLI en iyi 5 (birbirinin komsusu degil, %30'dan az ortusen)
        -> bu 5'in ICI DOLDURULUR (delik kalmaz) ve tam cozunurlukte gomulur
tur 1+: her aday EN YAKIN 5 komsusuyla birlestirilir = tur basi 5x5 = 25 yeni aday,
        hepsi delik-kapali ve YENIDEN gomulur (agirlik birlestirme YOK)
durma : kazanan 2 tur degismezse

Ucuz filtreler: maskeyi <%3 buyuten birlesim atlanir; ayni maske imzasi bir kez.
"""
import numpy as np, cv2
from src.task3_reference.termal2 import ayarlar as A, bolucu as B

_nrm = lambda x: x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def _imza(m):
    return cv2.resize(m.astype(np.uint8), (64, 36), interpolation=cv2.INTER_AREA).tobytes()


def ara(img, gomme, banka, ms=None, kom=None, iz=None):
    """img: kosu olcegindeki kare · banka: (K,D) referans vektorleri (4 donme)
    iz: verilirse {'satir': kazanan banka satiri} doldurulur (teshis; davranis AYNI)
    -> (kutu, skor)  ·  bulunamazsa (None, -9.0)"""
    if ms is None:
        ms = B.bol(img)
    if not ms:
        return None, -9.0
    KOM = kom if kom is not None else B.temas(ms)

    def skorla(maskeler):
        cr, mk, ok = [], [], []
        for q, m in enumerate(maskeler):
            k = B.kutu(m)
            if k is None or k[2]-k[0] < 8 or k[3]-k[1] < 8:
                continue
            c2, m2 = B.kirp_gen(img, m, k, A.GEN)
            cr.append(B.girdi(c2, m2)); mk.append(m2); ok.append(q)
        out = np.full(len(maskeler), -9.0)
        if not cr:
            return out
        V = []
        for s0 in range(0, len(cr), 96):
            V.append(_nrm(gomme(cr[s0:s0+96]).numpy()))
        S = np.concatenate(V) @ banka.T
        out[np.asarray(ok)] = S.max(1)
        if iz is not None:
            for q, w in zip(ok, S.argmax(1)):
                iz[_imza(maskeler[q])] = int(w)
        return out

    # ── TUR 0 · ucuz tarama -> CESITLI ilk UST -> tam cozunurlukte yeniden gom ──
    gomme.tavan = A.TUR0_TAVAN
    tek = skorla(ms)
    gomme.tavan = A.HAM_TAVAN
    sec, secm = [], []
    for t in np.argsort(-tek):
        if tek[t] <= -8:
            continue
        if t in {y for x in sec for y in KOM[x]}:                  # yan yana olmasin
            continue
        if any(float((ms[t] & m2).sum()) /
               max(1, min(int(ms[t].sum()), int(m2.sum()))) > 0.3 for m2 in secm):
            continue                                               # ortusmesin
        sec.append(int(t)); secm.append(ms[t])
        if len(sec) >= A.UST:
            break
    if not sec:
        sec = [int(np.argmax(tek))]; secm = [ms[sec[0]]]
    # TUR 0 kazananlarinin ICI DOLDURULUR (2026-08-25): secilen nesnenin icinde
    # delik kalmaz, TAM haliyle tur 1'e girer. Yutulan tekiller anahtara eklenir.
    anah, dolu = [], []
    for q, t in enumerate(sec):
        md = B.delik_kapa(secm[q])
        yut = {u for u in range(len(ms)) if int((ms[u] & md).sum()) >= 0.9*int(ms[u].sum())}
        anah.append(frozenset({int(t)} | yut)); dolu.append(md)
    tam = skorla(dolu)
    HAVUZ = {}
    for q in range(len(anah)):
        if tam[q] > -8:
            HAVUZ[anah[q]] = (dolu[q], float(tam[q]))
    if not HAVUZ:
        HAVUZ[anah[0]] = (dolu[0], float(tek[sec[0]]))

    # ── TUR 1+ ──
    KS = set(range(len(ms)))
    IMZA = {_imza(m) for m in ms}
    kazanan = max(HAVUZ.items(), key=lambda x: x[1][1]); durgun = 0
    for _tur in range(1, A.MAKS_TUR + 1):
        ust = sorted(HAVUZ.items(), key=lambda x: -x[1][1])[:A.UST]
        yk, ym = [], []
        for key, (m0, _) in ust:
            md = B.delik_kapa(m0)                                  # ici delinmesin
            if int(md.sum()) > int(m0.sum()):
                yut = {t for t in range(len(ms))
                       if int((ms[t] & md).sum()) >= 0.9 * int(ms[t].sum())}
                kd = frozenset(set(key) | yut)
                if kd not in HAVUZ and _imza(md) not in IMZA:
                    IMZA.add(_imza(md)); yk.append(kd); ym.append(md)
                m0, key = md, kd
            kom = set()
            for x in key:
                if x < len(ms):
                    kom |= set(KOM[x])
            a0 = int(m0.sum())
            for j in (kom & KS) - set(key):
                k2 = frozenset(set(key) | {j})
                if k2 in HAVUZ:
                    continue
                m2 = B.delik_kapa(m0 | ms[j])
                if int(m2.sum()) < a0 * (1.0 + A.BUYUME):           # anlamsiz buyume
                    continue
                sg = _imza(m2)
                if sg in IMZA:
                    continue
                IMZA.add(sg); yk.append(k2); ym.append(m2)
        if not ym:
            break
        s2 = skorla(ym)
        for q, k in enumerate(yk):
            if s2[q] > -8:
                HAVUZ[k] = (ym[q], float(s2[q]))
        y = max(HAVUZ.items(), key=lambda x: x[1][1])
        if y[0] == kazanan[0]:
            durgun += 1
            if durgun >= A.SABIR:
                break
        else:
            kazanan, durgun = y, 0
    kazanan = max(HAVUZ.items(), key=lambda x: x[1][1])
    if iz is not None:
        iz['satir'] = iz.get(_imza(kazanan[1][0]), -1)
    return B.kutu(kazanan[1][0]), float(kazanan[1][1])
