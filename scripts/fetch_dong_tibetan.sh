#!/usr/bin/env bash
set -euo pipefail

BASE="/Volumes/拓展盘/安联的扫地僧/SCI/计算机交叉/基于⺠族⾳乐交流的社会技能提升预测/数据/真实锚定数据集/audio_corpora/dong_tibetan_yt"
mkdir -p "$BASE"/{dong,tibetan,dong_yt,tibetan_yt}

MAX=30
FILTER="duration>60 & duration<1200"
FMT_WAV="-x --audio-format wav --audio-quality 0"
SLEEP="--sleep-interval 2 --max-sleep-interval 6"
RETRY="-c --retries 10 --fragment-retries 10"

echo "========== 第一步：侗族（B站） =========="

yt-dlp $FMT_WAV --max-downloads $MAX --match-filter "$FILTER" \
  $SLEEP $RETRY \
  -o "$BASE/dong/%(uploader)s_%(title).80s_%(id)s.%(ext)s" \
  "bilisearch:侗族大歌" || true

yt-dlp $FMT_WAV --max-downloads 20 --match-filter "$FILTER" \
  $SLEEP $RETRY \
  -o "$BASE/dong/%(uploader)s_%(title).80s_%(id)s.%(ext)s" \
  "bilisearch:侗族琵琶歌" || true

yt-dlp $FMT_WAV --max-downloads 15 --match-filter "$FILTER" \
  $SLEEP $RETRY \
  -o "$BASE/dong/%(uploader)s_%(title).80s_%(id)s.%(ext)s" \
  "bilisearch:侗族小黄村" || true

echo "========== 第二步：藏族（B站） =========="

yt-dlp $FMT_WAV --max-downloads $MAX --match-filter "$FILTER" \
  $SLEEP $RETRY \
  -o "$BASE/tibetan/%(uploader)s_%(title).80s_%(id)s.%(ext)s" \
  "bilisearch:藏族民歌" || true

yt-dlp $FMT_WAV --max-downloads $MAX --match-filter "$FILTER" \
  $SLEEP $RETRY \
  -o "$BASE/tibetan/%(uploader)s_%(title).80s_%(id)s.%(ext)s" \
  "bilisearch:藏族山歌 原生态" || true

yt-dlp $FMT_WAV --max-downloads 20 --match-filter "$FILTER" \
  $SLEEP $RETRY \
  -o "$BASE/tibetan/%(uploader)s_%(title).80s_%(id)s.%(ext)s" \
  "bilisearch:康巴弦子" || true

echo "========== 第三步：侗族（YouTube） =========="

yt-dlp $FMT_WAV --max-downloads $MAX \
  $SLEEP $RETRY \
  -o "$BASE/dong_yt/%(title).80s_%(id)s.%(ext)s" \
  "ytsearch30:Kam Grand Choir Dong ethnic China" || true

echo "========== 第四步：藏族（YouTube） =========="

yt-dlp $FMT_WAV --max-downloads $MAX \
  $SLEEP $RETRY \
  -o "$BASE/tibetan_yt/%(title).80s_%(id)s.%(ext)s" \
  "ytsearch30:Tibetan folk song traditional" || true

echo "========== 第五步：统一转16kHz单声道 =========="

cd "$BASE"
COUNT=0
for f in dong/*.wav tibetan/*.wav dong_yt/*.wav tibetan_yt/*.wav; do
  [ -f "$f" ] || continue
  OUT="${f%.wav}_16k.wav"
  [ -f "$OUT" ] && continue
  ffmpeg -i "$f" -ar 16000 -ac 1 "$OUT" -y -loglevel error
  COUNT=$((COUNT+1))
done
echo "转换完成：${COUNT} 个文件"

echo "========== 统计 =========="
echo "侗族（B站）: $(ls dong/*.wav 2>/dev/null | wc -l) 个文件"
echo "藏族（B站）: $(ls tibetan/*.wav 2>/dev/null | wc -l) 个文件"
echo "侗族（YouTube）: $(ls dong_yt/*.wav 2>/dev/null | wc -l) 个文件"
echo "藏族（YouTube）: $(ls tibetan_yt/*.wav 2>/dev/null | wc -l) 个文件"
echo "16kHz版本: $(find . -name '*_16k.wav' | wc -l) 个文件"
du -sh dong tibetan dong_yt tibetan_yt 2>/dev/null
echo "全部完成"
