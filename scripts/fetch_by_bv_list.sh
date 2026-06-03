#!/usr/bin/env bash
set -euo pipefail

BASE="/Volumes/拓展盘/安联的扫地僧/SCI/计算机交叉/基于⺠族⾳乐交流的社会技能提升预测/数据/真实锚定数据集/audio_corpora"
LIST="/Volumes/拓展盘/安联的扫地僧/SCI/计算机交叉/基于⺠族⾳乐交流的社会技能提升预测/实验/scripts/bv_whitelist.csv"
COOKIES="/Volumes/拓展盘/安联的扫地僧/SCI/计算机交叉/基于⺠族⾳乐交流的社会技能提升预测/实验/scripts/bilibili_cookies.txt"

if [ ! -f "$LIST" ]; then
  echo "[ERROR] BV 号清单不存在：$LIST"
  echo "请将 bv_whitelist_template.csv 复制为 bv_whitelist.csv 并填充真实 BV 号"
  exit 1
fi

DONG_DIR="$BASE/dong_tibetan_yt/dong_bv"
TIB_DIR="$BASE/dong_tibetan_yt/tibetan_bv"
MON_DIR="$BASE/mongolian_yt/mongolian_bv"
mkdir -p "$DONG_DIR" "$TIB_DIR" "$MON_DIR"

FMT_WAV="-x --audio-format wav --audio-quality 0"
SLEEP="--sleep-interval 3 --max-sleep-interval 8"
RETRY="-c --retries 10 --fragment-retries 10"

if [ -f "$COOKIES" ]; then
  COOK="--cookies $COOKIES"
  echo "使用 Cookie 文件：$COOKIES"
else
  COOK=""
  echo "未提供 Cookie，使用匿名访问"
fi

OK=0
FAIL=0
SKIP=0
TOTAL=0

while IFS=',' read -r ethnicity category bv_id title uploader notes; do
  [ "$ethnicity" = "ethnicity" ] && continue
  [ -z "$bv_id" ] || [[ "$bv_id" == BV1xxxxxxxxx* ]] && { SKIP=$((SKIP+1)); continue; }

  case "$ethnicity" in
    dong) OUTDIR="$DONG_DIR" ;;
    tibetan) OUTDIR="$TIB_DIR" ;;
    mongolian) OUTDIR="$MON_DIR" ;;
    *) echo "[WARN] 未知族别 $ethnicity 跳过 $bv_id"; SKIP=$((SKIP+1)); continue ;;
  esac

  TOTAL=$((TOTAL+1))
  URL="https://www.bilibili.com/video/$bv_id"
  echo "[$TOTAL] $ethnicity / $category / $bv_id  $title"

  if yt-dlp $FMT_WAV $SLEEP $RETRY $COOK \
       -o "$OUTDIR/${ethnicity}_${category}_${bv_id}.%(ext)s" \
       "$URL" 2>&1 | tail -3; then
    OK=$((OK+1))
  else
    FAIL=$((FAIL+1))
    echo "[FAIL] $bv_id"
  fi
done < "$LIST"

echo "==================== 16kHz 转码 ===================="
COUNT=0
for f in "$DONG_DIR"/*.wav "$TIB_DIR"/*.wav "$MON_DIR"/*.wav; do
  [ -f "$f" ] || continue
  OUT="${f%.wav}_16k.wav"
  [ -f "$OUT" ] && continue
  ffmpeg -i "$f" -ar 16000 -ac 1 "$OUT" -y -loglevel error 2>/dev/null && COUNT=$((COUNT+1))
done
echo "16kHz 转码：$COUNT"

echo "==================== 汇总 ===================="
echo "目标   ：$TOTAL"
echo "成功   ：$OK"
echo "失败   ：$FAIL"
echo "跳过   ：$SKIP"
echo "侗族 BV ：$(ls "$DONG_DIR"/*.wav 2>/dev/null | wc -l)"
echo "藏族 BV ：$(ls "$TIB_DIR"/*.wav 2>/dev/null | wc -l)"
echo "蒙古 BV ：$(ls "$MON_DIR"/*.wav 2>/dev/null | wc -l)"
du -sh "$DONG_DIR" "$TIB_DIR" "$MON_DIR" 2>/dev/null
echo "全部完成"
