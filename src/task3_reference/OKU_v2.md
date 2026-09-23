# GÖREV 3 v2 — REFERANS EŞLEME (RGB + TERMAL)

Eski DINOv3 tabanlı yol (`session.py`, `matcher.py`, `embedder.py`, `segmenters.py`)
**silinmedi**, olduğu gibi duruyor. v2 onun yanına kuruldu ve orkestratörde
varsayılan yapıldı.

## Açma / kapama

`src/client/orchestrator.py` içinde:

```python
REFERANS_V2 = bool(int(os.environ.get("REFERANS_V2", "1")))
```

- `REFERANS_V2=1` (varsayılan) → yeni boru hattı
- `REFERANS_V2=0` → eski DINOv3 yolu

## Dosyalar

| yol | ne |
|---|---|
| `g3_v2.py` | **orkestratör adaptörü** — eski `ReferenceSession` ile aynı arayüz, altta v2'yi çalıştırır |
| `rgb2/` | RGB boru hattı (eşik 0.44, 3 ardışık tespitte çıpala) |
| `termal2/` | **TERMAL boru hattı** (bu belgenin konusu) |

`termal2` kendi `bolucu/arama/gomme/takip` kopyalarını taşır. Sebep: termal
çalışmasında bu modüllerde iyileştirmeler yapıldı (komşuluk ölçütü, delik
kapatma yeri, bf16) ve bunlar RGB'nin **doğrulanmış** davranışını değiştirirdi.
RGB'ye dokunulmadı.

## Termal boru hattı

```
OTURUM BAŞI
  referans görselleri → HQ-SAM ile kırp → crop/ref_crops/<oid>.png
  (ref_crops/*_cut.png zaten kırpık, HQ-SAM'e girmez)
  her kırpım → GRİ'ye çevir → 4×90° döndür → WebSSL CLS → oid başına 4 vektör

HER KARE
  ARAMA modu:  FastSAM-x → ayrık bölme → delik kapatma
               temas grafiği (en yakın 5) + turlu birleşim (ilk 5, 2 tur sabır)
               WebSSL CLS → kosinüs → en iyi aday
               skor ≥ 0.45 VE önceki tespitle tutarlı → çıpa adayı
               4 ardışık aday → SAM ÇIPALANIR, mod TAKİP
               ** SADECE_TAKIP açık: bu modda HİÇBİR kutu gönderilmez **

  TAKİP modu:  DAM4SAM (SAM2.1 base_plus) kare kare sürer, eşleştirici durur
               her 5 karede eşleştirici DENETİM için çalışır
                 kutusu SAM ile IoU < 0.30 ise uyuşmazlık sayacı artar
                 2 kez üst üste uyuşmazsa → KİLİT AÇILIR
               SAM maskeyi kaybederse → KİLİT AÇILIR → ARAMA
```

## Ölçülen sonuç (2026 oturumu, 253 GT kutusu)

| | değer |
|---|---|
| doğru | 201 |
| **yanlış** | **6** |
| gönderilmedi | 46 |
| yanlış pozitif | 1 |
| doğruluk | %79.4 |
| **kesinlik** | **%97.1** |
| ortalama IoU | 0.871 |
| hız | ~1.0 sn/kare |

`SADECE_TAKIP` kapatılırsa: 225 doğru / **22 yanlış** / 6 boş → doğruluk %88.9,
kesinlik %91.1. Takas: 24 doğru verip 16 yanlıştan kurtulmak. Yanlış kutunun
ceza yediği varsayımıyla açık bırakıldı; `termal2/ayarlar.py` → `SADECE_TAKIP`.

## Gereken ağırlıklar (hepsi `models/` içinde mevcut)

| dosya | ne |
|---|---|
| `rgb_fastsam_x.pt` | FastSAM-x (modaliteden bağımsız, isim RGB'den kalma) |
| `webssl_dino300m_light2b/` | gömme |
| `sam2.1_hiera_base_plus.pt` | DAM4SAM takip |
| `sam_hq_vit_l.pth` | referans kırpma |

`third_party/DAM4SAM/` gerekli; `checkpoints/sam2.1_hiera_base_plus.pt`
sembolik bağı ilk çalıştırmada otomatik kurulur.

## Ayarlar

Hepsi `termal2/ayarlar.py` içinde, her birinin yanında ölçüm gerekçesi yazılı.
En kritik üçü:

```python
CIPA    = 0.45   # 0.42 -> 26 yanlış, 0.30 -> 37 yanlış
ARDISIK = 4      # 3 -> 36 yanlış, 2 -> 43 yanlış
BANKA   = "gri"  # renkli+gri %88.5, yalnız renkli %84.6, +CLAHE %82.6
```

Ayrıntılı gerekçe ve elenen yöntemler:
`/home/adem/Desktop/referance/nuron_referance_termal/YONTEM.md` ve `ELENENLER.md`


---

## Z3 guncellemesi (2026-09-02)

`rgb2/motor.py` + `rgb2/ayarlar.py`, gelistirme motorunun (nuron_referance_rgb/new_work/
mimari.py, kosu adi **Z3**) birebir portuna yukseltildi. `termal2/` DEGISMEDI.

**Yeni olanlar** (hepsi olcumle, iki veri setinin 1342 etiketli karesi):
- **Iki kip kurali** `ayarlar.KURAL`: referans TERMAL ise (R=G=B) skor esigi DUSER (0.62),
  marj (0.06) ve ardisiklik (N=3) YUKSELIR — termalde mutlak skor ayirmiyor
  (dogru medyan 0.750 / yanlis 0.740). RGB: 0.66 / 0.06 / N=2.
- **RAKIPSIZ kare kurali**: karede farkli nesneden rakip yoksa marj olculemez; eski kod
  bunu sonsuz marj sayip serbest gecis veriyordu — bos pencerelerde bosa gonderim 51 -> 1.
- **Cozunurluk kademesi**: ust uste 60 arama karesinde kilit yoksa FastSAM 512 -> 768.
- **Kilit sonrasi bekciler**: duzeltme penceresi (DUZELT=10), periyodik denetim
  (DENETIM_N=12, SAM savunmasi), kenar-giris dogrulamasi — onceki motor.py'de yoktu.
- **BITTI karari kipe gore**: termalde kapali (yanlis kilit "aralik bitti" ilan edip
  116 kare yakmisti).

**Sonuc** (sunucu pencereleri, elle etiket): 2026 %74.9/%74.6/IoU 0.94 ·
simulasyon %93.5/%71.3 · hiz 0.8-1.4 sn/kare · bos pencerede gonderim 1/123.
Dogrulama: `new_work/kapi_nuron.py` NURON motorunu ayni pencerelerde kosar,
`kkarsi.py Z3 NZ3` birebir karsilastirir. Tam gerekce: `new_work/KURALLAR_KIP.md`.
Eski motor: `rgb2/_ESKI/motor_M16_20260902.py`.

---

## Pencere-url eslemesi duzeltmesi (2026-09-02, uctan uca test)

Bulgu: sunucu AYNI referans goruntusunu 2 ayri aralikta istedi (2026: Referans_Nesne_10
-> REF02 97-135 ve REF04 382-427). `connect/iface/object_detection_model.py` "stem -> url"
sozlugu kurdugu icin REF02 dustu; 97-135 araliginda G3 hic calismadi (0.25 sn/kare, bos
gonderim). Gelistirme kosusu ayni pencerede 19 kare gonderiyordu. Yarismada tek referans
2-3 aralikta gelebilir -> baglanti ESNEK yapildi (rgb+termal ORTAK katman; `termal2/` ve
`rgb2/` DEGISMEDI):
- url -> stem -> oid: COK url -> 1 id serbest; eslesmeyen url loga ERROR duser.
- kutu, o anki ARALIGIN url'siyle gonderilir (id_to_url ile degil).
- aktif url kumesi degisince `ReferenceSessionV2.aralik_degisti()` -> takip sifirlanir
  (ayni id ardisik/ayrik iki aralikta gelse bile).
- `g3_v2.stem_oid()` TEK kaynak: build_bank ve url eslemesi ayni kurali kullanir
  (sayisiz dosya adi, cakisan sayi -> bos numara).
Yedekler: `connect/iface/object_detection_model.py.YEDEK_20260902_pencere`,
`src/task3_reference/g3_v2.py.YEDEK_20260902_pencere`.

---

## TERMAL HİBRİT (2026-09-05) — `termal3/hibrit.py`

Termal dalı artık **TermalOturumHibrit** (g3_v2.py, `NURON_TERMAL_HIBRIT=1` varsayılan; `=0` eski termal2.TermalOturum):

```
referans kırpımı (HQ-SAM, build_bank) → R=G=B mi?
   GRİ (termal fotoğraf) → termal2.TermalOturum   (WebSSL gri banka + turlu arama + 0.45/4 + SAM2.1-B, SADECE_TAKIP)  — DEĞİŞMEDİ
   RENKLİ (RGB fotoğraf)  → termal3.TermalOturum3  (C-RADIOv3-H + FastSAM temas büyümesi + ALAN_MAKS 0.65 + pencereli kilit
                                                   [son 3 arama karesi, ilk-3 aday, IoU 0.5 zinciri, ×1.6] + Z3 bekçileri)
```

Karar kuralı nesneye değil görüntü türüne bakar; iki hatta da referansa özel parametre yok.

**Ölçüm (2026 termal oturumu, elle etiket, gerçek nesne aralıkları 312 kutu; 1600-1700 benzeyen nesne — kullanıcı):**

| | termal2 (eski) | **hibrit** |
|---|---|---|
| kutu yakalama (IoU≥0.5 veya kapsama≥0.9 & alan≥0.4·GT) | %64.4 | **%83.3** |
| kesinlik | %97.1 | **%97.7** |
| takip (ilk doğru kareden sonra) | %97.0 | %95.8 |
| hız | 0.91 sn/kare | **0.46 sn/kare** |
| GPU tepe (yalnız G3 süreci) | ~2.4 GB | **5.2 GB** (HIBRIT_PAYLAS=1: FastSAM + DAM4SAM sam21pp-B iki hatta ortak; paylaşımsız 6.6 GB) |

Aralık bazında: 320-390 %91 [t2] · 400-460 %83 [t3, eski %0] · 570-630 %84 [t3] · 780-820 %57 [t2] · 880-920 %82 [t3] · 1760-1840 %89 [t2].

**Dikkat:** renkli hat, nesne kadrajda yokken/belirsizken de karelerin ~%80'inde kutu gönderir (kutusuz aralıklarda ölçüldü);
termal2 bunu yapmıyordu. Aralıklar nesnenin göründüğü karelerle tanımlı olduğu için yarışmada etkisi sınırlı, ama bilinmeli.
Daha temkinli seçenek: renkli hatta `GOMME_ADI=webssl_300m_l TAKIPCI=sam21pp-B` (gerçek nesnelerde %74, boş pencerede %26).

**Bellek:** termal E2E'de G3→G2 sırası ve `empty_cache` korunur. Hibrit tek başına tepe 5.2 GB; YOLO/PaDiM + DPVO (ayrı süreç) ile
8 GB'a sığması E2E'de doğrulanmalı. OOM olursa `_gorev3` içindeki sıralı fallback devreye girer; kalıcı çözüm `NURON_TERMAL_HIBRIT=0`.

Ek ortam değişkenleri (termal3/ayarlar.py): `HIBRIT_PAYLAS` (1), `HIBRIT_ZORLA` (t2|t3 zorla), `ALAN_MAKS`, `PEN_*`, `ONAY_K`, `GOMME_ADI`, `TAKIPCI`.
Gerekçe, bütün denemeler ve elenenler: `/home/adem/Desktop/referance/nuron_referance_termal/YONTEM_v3.md` ve `KOSU_LOG.md`.
Yedekler: `g3_v2.py.YEDEK_20260905_hibrit`, `OKU_v2.md.YEDEK_20260905_hibrit`.

### Termal hibrit — K3b güncellemesi (2026-09-18)
`termal3/` klasörü `nuron_referance_termal` ile eşitlendi. Yeni varsayılan **K3b**: `KAPI_SONRA=True`, `PEN_S1_MIN=0.60` (ayarlar.py) —
renkli hatta bir kilit bırakıldıktan sonraki YENİDEN kilitlerde kazanan adayın skoru 0.60 altındaysa kilit verilmez; ilk kilit kararı
değişmez. Ölçüm (nuron_referance_termal/YONTEM_v3.md §10): 2026 GERÇEK BULMA %83.3 / TAKİP-A %95.8 / kesinlik %97.7 (hibrit A ile kare
kare birebir), nesnesiz 3 aralıkta gönderim %79 → %38.5; 2025'te 7/7 aralık aynı BULMA, kesinlik +5 puan. Eski hibrit A davranışı:
ortam değişkeni `KAPI_SONRA=0`. Diğer kapılar (PEN_Z_MIN, ISINMA_*, SOGUMA_*) kapalı; belge `termal3/FP_KAPILAR.md`.
Yedekler: `termal3/*.py.YEDEK_20260918_hibritA`, `OKU_v2.md.YEDEK_20260918_hibritA`.
