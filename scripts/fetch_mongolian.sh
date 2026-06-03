#!/usr/bin/env bash
set -euo pipefail

BASE="/Volumes/拓展盘/安联的扫地僧/SCI/计算机交叉/基于⺠族⾳乐交流的社会技能提升预测/数据/真实锚定数据集/audio_corpora/mongolian_yt"
mkdir -p "$BASE"/{khoomei,morin_khuur,long_song,urtyn_duu,mongolian_yt}

MAX=40
FILTER="duration>60 & duration<1500"
FMT_WAV="-x --audio-format wav --audio-quality 0"
SLEEP="--sleep-interval 2 --max-sleep-interval 6"
RETRY="-c --retries 10 --fragment-retries 10"

echo "========== 蒙古族呼麦（B站） =========="
yt-dlp $FMT_WAV --max-downloads $MAX --match-filter "$FILTER" \
  $SLEEP $RETRY \
  -o "$BASE/khoomei/%(uploader)s_%(title).80s_%(id)s.%(ext)s" \
  "bilisearch:蒙古族呼麦" || true

yt-dlp $FMT_WAV --max-downloads 30 --match-filter "$FILTER" \
  $SLEEP $RETRY \
  -o "$BASE/khoomei/%(uploader)s_%(title).80s_%(id)s.%(ext)s" \
  "bilisearch:呼麦 原生态" || true

echo "========== 蒙古族马头琴（B站） =========="
yt-dlp $FMT_WAV --max-downloads $MAX --match-filter "$FILTER" \
  $SLEEP $RETRY \
  -o "$BASE/morin_khuur/%(uploader)s_%(title).80s_%(id)s.%(ext)s" \
  "bilisearch:马头琴 蒙古族" || true

echo "========== 蒙古族长调（B站） =========="
yt-dlp $FMT_WAV --max-downloads $MAX --match-filter "$FILTER" \
  $SLEEP $RETRY \
  -o "$BASE/long_song/%(uploader)s_%(title).80s_%(id)s.%(ext)s" \
  "bilisearch:蒙古族长调" || true

yt-dlp $FMT_WAV --max-downloads 20 --match-filter "$FILTER" \
  $SLEEP $RETRY \
  -o "$BASE/long_song/%(uploader)s_%(title).80s_%(id)s.%(ext)s" \
  "bilisearch:乌尔汀哆" || true

echo "========== 蒙古族短调（B站） =========="
yt-dlp $FMT_WAV --max-downloads 25 --match-filter "$FILTER" \
  $SLEEP $RETRY \
  -o "$BASE/urtyn_duu/%(uploader)s_%(title).80s_%(id)s.%(ext)s" \
  "bilisearch:蒙古族短调民歌" || true

echo "========== YouTube补充源 =========="
yt-dlp $FMT_WAV --max-downloads $MAX \
  $SLEEP $RETRY \
  -o "$BASE/mongolian_yt/%(title).80s_%(id)s.%(ext)s" \
  "ytsearch30:Mongolian khoomei throat singing" || true

yt-dlp $FMT_WAV --max-downloads 20 \
  $SLEEP $RETRY \
  -o "$BASE/mongolian_yt/%(title).80s_%(id)s.%(ext)s" \
  "ytsearch20:Mongolian long song urtiin duu" || true

echo "========== 统一转16kHz单声道 =========="
cd "$BASE"
COUNT=0
for f in khoomei/*.wav morin_khuur/*.wav long_song/*.wav urtyn_duu/*.wav mongolian_yt/*.wav; do
  [ -f "$f" ] || continue
  OUT="${f%.wav}_16k.wav"
  [ -f "$OUT" ] && continue
  ffmpeg -i "$f" -ar 16000 -ac 1 "$OUT" -y -loglevel error
  COUNT=$((COUNT+1))
done
echo "转换完成：${COUNT} 个文件"

echo "========== 统计 =========="
echo "呼麦: $(ls khoomei/*.wav 2>/dev/null | wc -l) 个文件"
echo "马头琴: $(ls morin_khuur/*.wav 2>/dev/null | wc -l) 个文件"
echo "长调: $(ls long_song/*.wav 2>/dev/null | wc -l) 个文件"
echo "短调: $(ls urtyn_duu/*.wav 2>/dev/null | wc -l) 个文件"
echo "YouTube: $(ls mongolian_yt/*.wav 2>/dev/null | wc -l) 个文件"
echo "16kHz版本: $(find . -name '*_16k.wav' | wc -l) 个文件"
du -sh khoomei morin_khuur long_song urtyn_duu mongolian_yt 2>/dev/null
echo "蒙古族音频采集完成"
