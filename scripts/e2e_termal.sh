#!/bin/bash
# NURON — TERMAL UCTAN UCA TEST (tek komut).  Ayrinti ve konsol konsol surum: work.txt
#   bash scripts/e2e_termal.sh [oturum=termal|termal2025_4] [gps_extra=1200,1600] [hibrit=1] [K=1]
#
# SIRA: sunucu -> ISTEMCI (agir G1+G3 modelleri VRAM'i once alir) -> WORKER (kalan yere K DPVO).
# 8 GB kart: hibrit G3 tepe ~5.5 GB + worker (K=1) ~2.1 GB -> G3'te ara sira OOM olur
# (kod empty_cache ile toparlar, o karede referans kutusu gitmez; G1/G2 ETKILENMEZ).
# Tamamen OOM'suz kosu icin:  bash scripts/e2e_termal.sh termal 1200,1600 0 2   (G3 = eski termal2)
# Cikti: server/sonuclar/<cikti>/{predictions.jsonl,results.json,analiz_termal.json}
#        loglar logs/e2e_termal_*_<zaman>.log ; onceki cikti klasoru _ESKI_<zaman> olarak SAKLANIR.
set -u
OTURUM=${1:-termal}; EXTRA=${2:-1200,1600}; HIB=${3:-1}; K=${4:-1}
N="${NURON_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"; SRV="${NURON_SERVER_ROOT:-$HOME/Desktop/server}"
case $OTURUM in
  termal)       CIKTI=2026_termal;            VERI=$SRV/oturum_termal/2026;;
  termal2025_4) CIKTI=2025_oturum_4_termal;   VERI=$SRV/oturum_termal/2025_oturum_4;;
  *) echo "bilinmeyen oturum: $OTURUM"; exit 1;;
esac
TS=$(date +%Y%m%d_%H%M); mkdir -p $N/logs

ACIK=$(pgrep -af "connect\.main|gps_worker\.py|nuron_sim_server\.py" | grep -v "bash" | grep -v "e2e_termal")
[ -n "$ACIK" ] && { echo "HATA: onceki surecler acik:"; echo "$ACIK"; exit 2; }

[ -d $SRV/sonuclar/$CIKTI ] && mv $SRV/sonuclar/$CIKTI $SRV/sonuclar/${CIKTI}_ESKI_$TS && echo "onceki cikti saklandi: ${CIKTI}_ESKI_$TS"
# ref_crops SART: onceki modalitenin kirpimlari bankaya karisir. Digerleri gecici IPC dosyalari.
rm -rf $N/offline_data/ref_crops $N/offline_data/ref_bank_online.npz; mkdir -p $N/offline_data/ref_crops
rm -f $N/frames/new/* $N/runtime/gps/in/* $N/runtime/gps/out/* $N/runtime/gps/bridge_state.pkl* $N/runtime/gps/STOP 2>/dev/null
echo "[$(date +%T)] oturum=$OTURUM gps=$EXTRA hibrit=$HIB K=$K -> sonuclar/$CIKTI"

# --- 1) sunucu ---
( cd $SRV && exec nohup python3 nuron_sim_server.py --session $OTURUM --gps-extra $EXTRA --reset \
    > $N/logs/e2e_termal_server_$TS.log 2>&1 ) & SPID=$!
sleep 5; curl -sf http://127.0.0.1:5000/ >/dev/null || { echo "SUNUCU KALKMADI"; tail -5 $N/logs/e2e_termal_server_$TS.log; exit 3; }

# --- 2) istemci (G1 + G3 + arayuz; PaDiM CPU'da -> ~0.4 GB VRAM serbest) ---
( cd $N && exec env TEAM_NAME=nuron PASSWORD=local EVALUATION_SERVER_URL="http://127.0.0.1:5000/" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NURON_MODALITY=termal \
    NURON_PADIM_DEVICE=cpu NURON_TERMAL_HIBRIT=$HIB NURON_TERMAL_K=$K \
    nohup python -u -m connect.main > $N/logs/e2e_termal_client_$TS.log 2>&1 ) & CPID=$!
for i in $(seq 1 240); do grep -q "Tum modeller hazir\|Traceback" $N/logs/e2e_termal_client_$TS.log 2>/dev/null && break; sleep 1; done
grep -h "NURON" $N/logs/e2e_termal_client_$TS.log | tr '\r' '\n' | grep "NURON" | cut -c1-140

# --- 3) worker (dpvo env: cuda_corr yalniz orada derli) ---
( cd $N && exec env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NURON_TERMAL_K=$K \
    nohup conda run --no-capture-output -n dpvo python -u scripts/gps_worker.py --modality termal \
    > $N/logs/e2e_termal_worker_$TS.log 2>&1 ) & WPID=$!
for i in $(seq 1 180); do grep -q "hazır\|hazir\|Traceback" $N/logs/e2e_termal_worker_$TS.log 2>/dev/null && break; sleep 1; done
echo "[$(date +%T)] worker: $(grep -h 'hazır\|hazir\|açılamadı' $N/logs/e2e_termal_worker_$TS.log | tr '\n' '|')"

# --- kosu bitene kadar bekle (istemci kendisi cikar) ---
wait $CPID; echo "[$(date +%T)] istemci bitti"
touch $N/runtime/gps/STOP; sleep 3
kill $WPID 2>/dev/null; kill $SPID 2>/dev/null; wait $WPID 2>/dev/null; wait $SPID 2>/dev/null
rm -f $N/runtime/gps/STOP

# --- olcum + video ---
python3 $SRV/e2e_analiz.py --cikti $CIKTI --veri $VERI --ekstra $EXTRA \
        --json $SRV/sonuclar/$CIKTI/analiz_termal.json | tee $N/logs/e2e_termal_analiz_$TS.txt
L=$(ls -t $N/_logs/*.log | head -1)
echo "OOM kare -> G1: $(grep -c 'G1 tespit hatasi: CUDA out of memory' $L)  G3: $(grep -c 'G3 referans hatasi: CUDA out of memory' $L)"
( cd $SRV && python3 sonuc_video.py termal )    # -> ~/Desktop/nuron_sonuc_termal.mp4
