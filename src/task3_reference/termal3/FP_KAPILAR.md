# TERMAL v3 — YANLIS-POZITIF (FP) KAPILARI  (2026-09-17)

Sorun: nesne kadrajda YOKKEN de pencereli kilit kuruluyor (bos araliklarda ~%79 kare
kutu gonderildi). Cozum: kilit ANINA ve kilit SONRASINA ayarlanabilir kapilar.
**HEPSI VARSAYILAN KAPALI** — env verilmezse motor eski davranisi BIREBIR kosar
(regresyon testi: sonda_radio_rg 1107 arama karesi, 0 fark).

## Parametreler (ayarlar.py, `_env` ile ezilir; `ozet()["FP_KAPI"]` kayda girer)

| env            | tip   | kapali | anlam |
|----------------|-------|--------|-------|
| `PEN_S1_MIN`   | float | `0.0`  | kazanan uyenin BU KAREDEKI skoru < esik -> kilit yok |
| `PEN_PUAN_MIN` | float | `0.0`  | kazanan kumenin puan toplami (p1) < esik -> kilit yok |
| `PEN_MARJ_MIN` | float | `0.0`  | kilit karesinde s1-s2 (sonda marj tanimi) < esik -> kilit yok |
| `PEN_Z_MIN`    | float | `0.0`  | kazananin ilk-K aday icindeki z-degeri < esik -> kilit yok (K=`SONDA_K` ya da 8; ort/sd TUM ilk-K skoru uzerinden, kazanan DAHIL, nufus sd; sd<=1e-6 -> z=-9 (ret); tek aday -> kapi calismaz) |
| `PEN_TAM`      | bool  | `0`    | kazanan kume `PEN_W` karenin HEPSINDE gorulmus olmali |
| `PEN_KARARLI`  | float | `0.0`  | kume uyelerinin (kare-major duz liste) ardisik ikili IoU ort < esik -> kilit yok; cift yoksa 1.0 |
| `PEN_KENAR`    | bool  | `0`    | kazanan kutu kare kenarina degiyorsa (`KENAR_PX`) kilit yok |
| `ISINMA_N`     | int   | `0`    | kilitten sonraki ilk N MODEL karesi izlenir |
| `ISINMA_M`     | int   | `1`    | o N karede en az M kez `_onay` uyusmasi sart; yoksa `kilit_birak("isinma")` |
| `ISINMA_HER`   | bool  | `0`    | isinma boyunca modeli HER karede kostur (hizli karar, EK GPU YUKU) |
| `SOGUMA_N`     | int   | `0`    | birakilan kutu N KARE boyunca yasakli (yeniden ayni yanlis nesneye kilit yok) |
| `SOGUMA_IOU`   | float | `0.5`  | yasakli kutuyla bu IoU'yu gecen kazanan kilit alamaz |

`PEN_S1_MIN` / `PEN_Z_MIN` / `PEN_KARARLI` tanimlari `arastirma/fp_kural_tara.py`
(`pencere_adim` tani + `kapi_gec`) ile BIREBIR ayni: s_min=`t["s"]`, z_min=`z`,
ic_iou_min=`t["ic_iou"]` (341 kilit-adayi karede 341/341 esit olculdu).
NOT: `PEN_KENAR` motorun kendi `KENAR_PX=3` payini kullanir, simulator `pay=2` kullanir.

Kapilar `_pencere_kilit` icinde `_fp_kapilar()` ile SIRAYLA bakilir; ilk kesen kapi
`sonda["kilit_ozet"]["ret"]` alanina yazilir (`s1|puan|marj|z|tam|kararli|kenar|soguma`).

## Sonda zenginlestirmesi (teshis, JSON)
`sonda` dict'ine eklendi: `n_aday` (skorlanan aday sayisi), `s_ort`, `s_sd` (tum aday
skorlari) ve `kilit_ozet` = `{p1, p2, oran, kareler, tam, s1, marj, z, kararli, ret?}`.
`kilit_ozet` **kilit denenen HER arama karesinde** (kilit olmasa da) doldurulur —
kilit-ani ozellikleri sonradan CPU'da taranabilir.
`birak_neden`'e yeni deger: `isinma`. (`soguma` bir birakma nedeni DEGIL, kilit-ani
kapisidir; `kilit_ozet.ret == "soguma"` olarak gorunur.)

## Kullanim
```bash
# tek kapi
PEN_S1_MIN=0.70 python3 -m arastirma.olcum_termal3.kosu ...
# birkac kapi birlikte + kilit sonrasi isinma + soguma
PEN_Z_MIN=1.0 PEN_MARJ_MIN=0.08 ISINMA_N=3 ISINMA_M=1 SOGUMA_N=40 python3 ... 
```

## Ilk gozlem (sonda_radio_rg, `_pencere_kilit` yeniden oynatildi)
- `400-460` (nesne cogu karede YOK): kapali 13 kilit (3 dogru) -> `PEN_Z_MIN=1.0` 10,
  `PEN_MARJ_MIN=0.10` 10, `PEN_S1_MIN=0.70` 2 kilit.
- `880-920` (nesne VAR): kapali 6/6 dogru -> `PEN_Z_MIN=1.0` 6/6 (kayip YOK),
  `PEN_KARARLI=0.7` 1/1, `PEN_MARJ_MIN=0.10` 3/3 (dogru kilitleri de kesiyor).
- Yani `PEN_Z_MIN` ve `PEN_S1_MIN` en umut verici; `PEN_TAM` tek basina hicbir sey
  kesmiyor (PEN_W=PEN_M=3 oldugu icin zaten "tam" sarti saglanmis oluyor).

## Riskler
- `ISINMA_HER=1` isinma penceresinde kare basina BIR EK tam-cozunurluk `model_kos`
  ekler (GPU yuku). `ISINMA_HER=0` iken ek yuk YOKTUR (mevcut DUZELT/DENETIM
  kosularini sayar) ama karar gec gelir.
- IKILI KARAR modunda (`gomme2` dolu): kapilar her modelin ATESINI engeller; ancak
  bir modelin ONCEKI karelerden kalan atesi hala gecerliyse ikili birlesim kilit
  kurabilir. Tek model (varsayilan t3) hattinda boyle bir bosluk yok.

## Ek (2026-09-17 22:50, orkestratör): `KAPI_SONRA`
`KAPI_SONRA=1` → kilit-anı kapıları (PEN_S1_MIN, PEN_Z_MIN, …) yalnız aralıkta EN AZ BİR kilit bırakıldıktan sonraki yeniden
kilitlerde uygulanır; ilk kilit taban kuralıyla gelir (BULMA/kilit karesi taban ile birebir). Gerekçe: gerçek nesnelerde kilit yalnız
aralık sonunda (`cikis`) bırakılıyor; boş aralıklarda ise 20-30 karede `sam_kayip`/`cikis` ile düşüp başka yere yeniden kilitleniyor
(10-80 ve 1080-1120'de 3'er kilit bölümü). Sayaç: `Motor.birak_say` (sifirla → 0, kilit_birak → +1). Varsayılan kapalı.

## Varsayılan değişikliği (2026-09-18 06:52): K3b varsayılan oldu
`KAPI_SONRA=True`, `PEN_S1_MIN=0.60` artık varsayılan (ayarlar.py). Eski hibrit A davranışı için `KAPI_SONRA=0`. Ölçüm:
2026 GERÇEK BULMA %83.3 (tabanla kare kare birebir), boş 3 aralıkta gönderim %79.0 → %38.5; 2025'te 7/7 aralık aynı BULMA, kesinlik +5 puan.
