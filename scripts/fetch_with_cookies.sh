#!/usr/bin/env bash
set -euo pipefail

BASE="/Volumes/拓展盘/安联的扫地僧/SCI/计算机交叉/基于⺠族⾳乐交流的社会技能提升预测/数据/真实锚定数据集/audio_corpora"
COOKIES="/Volumes/拓展盘/安联的扫地僧/SCI/计算机交叉/基于⺠族⾳乐交流的社会技能提升预测/实验/scripts/bilibili_cookies.txt"

DONG_DIR="$BASE/dong_tibetan_yt/dong_v2"
TIB_DIR="$BASE/dong_tibetan_yt/tibetan_v2"
MON_DIR="$BASE/mongolian_yt/mongolian_v2"
mkdir -p "$DONG_DIR" "$TIB_DIR" "$MON_DIR"

if [ ! -f "$COOKIES" ]; then
  echo "[ERROR] Cookie 文件不存在：$COOKIES"
  echo "请先用浏览器扩展导出 B 站 cookies，保存为 Netscape 格式"
  exit 1
fi

MAX=40
FILTER="duration>60 & duration<1500"
FMT_WAV="-x --audio-format wav --audio-quality 0"
SLEEP="--sleep-interval 3 --max-sleep-interval 8"
RETRY="-c --retries 10 --fragment-retries 10"
COOK="--cookies $COOKIES"
UA='--user-agent "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"'

fetch() {
  local query="$1"
  local outdir="$2"
  local max="${3:-$MAX}"
  echo "========== 搜索: $query =========="
  eval yt-dlp $FMT_WAV --max-downloads $max --match-filter \"$FILTER\" \
    $SLEEP $RETRY $COOK $UA \
    -o "\"$outdir/%(uploader)s_%(title).80s_%(id)s.%(ext)s\"" \
    "\"bilisearch:$query\"" || true
}

echo "==================== 侗族 ===================="
fetch "侗族大歌" "$DONG_DIR" 40
fetch "侗族琵琶歌" "$DONG_DIR" 25
fetch "侗族小黄村大歌" "$DONG_DIR" 15
fetch "侗族多耶" "$DONG_DIR" 15

echo "==================== 藏族 ===================="
fetch "藏族民歌 原生态" "$TIB_DIR" 40
fetch "康巴弦子舞" "$TIB_DIR" 25
fetch "藏族山歌" "$TIB_DIR" 25
fetch "藏族锅庄" "$TIB_DIR" 15

echo "==================== 蒙古族 ===================="
fetch "蒙古族呼麦" "$MON_DIR" 40
fetch "马头琴 民乐" "$MON_DIR" 25
fetch "蒙古族长调民歌" "$MON_DIR" 25
fetch "蒙古族短调" "$MON_DIR" 15

echo "==================== 16kHz 转码 ===================="
COUNT=0
for f in "$DONG_DIR"/*.wav "$TIB_DIR"/*.wav "$MON_DIR"/*.wav; do
  [ -f "$f" ] || continue
  OUT="${f%.wav}_16k.wav"
  [ -f "$OUT" ] && continue
  ffmpeg -i "$f" -ar 16000 -ac 1 "$OUT" -y -loglevel error
  COUNT=$((COUNT+1))
done
echo "16kHz 转码完成：$COUNT 个文件"

echo "==================== 统计 ===================="
echo "侗族 v2: $(ls "$DONG_DIR"/*.wav 2>/dev/null | wc -l)"
echo "藏族 v2: $(ls "$TIB_DIR"/*.wav 2>/dev/null | wc -l)"
echo "蒙古 v2: $(ls "$MON_DIR"/*.wav 2>/dev/null | wc -l)"
du -sh "$DONG_DIR" "$TIB_DIR" "$MON_DIR" 2>/dev/null
echo "全部完成"
