#!/bin/sh
# 웹스토어 제출용 zip 생성 — 확장 실행에 필요한 파일만 담는다.
# 사용법: sh scripts/build-zip.sh  →  dist/coupang-rank-finder-<버전>.zip
set -e
cd "$(dirname "$0")/.."

VERSION=$(sed -n 's/.*"version": "\([^"]*\)".*/\1/p' manifest.json)
OUT="dist/coupang-rank-finder-${VERSION}.zip"

mkdir -p dist
rm -f "$OUT"

# env.json(개인 키)은 절대 포함하지 않는다
zip -r "$OUT" \
  manifest.json \
  background.js \
  content.js \
  popup.html popup.css popup.js \
  api/ \
  icons/ \
  -x "*.DS_Store"

echo "생성됨: $OUT"
unzip -l "$OUT"
