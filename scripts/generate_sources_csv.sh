#!/usr/bin/env bash

set -euo pipefail

RAW_DIR="dataset/raw"
OUTPUT_CSV="dataset/metadata/sources.csv"

VALID_EXTENSIONS=("jpg" "jpeg" "png" "bmp")

mkdir -p "$(dirname "$OUTPUT_CSV")"

echo "image_id,source" > "$OUTPUT_CSV"

count=0

for file in "$RAW_DIR"/*; do
  [[ -f "$file" ]] || continue

  filename="$(basename "$file")"
  extension="${filename##*.}"
  extension="${extension,,}"

  valid=false

  for valid_ext in "${VALID_EXTENSIONS[@]}"; do
    if [[ "$extension" == "$valid_ext" ]]; then
      valid=true
      break
    fi
  done

  [[ "$valid" == true ]] || continue

  image_id="${filename%.*}"

  printf '%s,\n' "$image_id" >> "$OUTPUT_CSV"

  ((count+=1))
done

echo "Selesai."
echo "  Gambar ditemukan : $count"
echo "  Output            : $OUTPUT_CSV"