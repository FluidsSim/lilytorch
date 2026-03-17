#!/bin/bash
# ── run_pipeline.sh ─────────────────────────────────────────────────────────
# Wrapper that runs the three standalone pipeline steps in sequence:
#   1. correct_video1.py   – perspective correction + trim for video1
#   2. match_scale_video2.py – metric scale, anchor, sync for video2
#   3. stack_videos.sh     – ffmpeg vertical stack
#
# Usage:
#   ./run_pipeline.sh <video1> <video2> [--flip] [--no-homography]
#
#   video1          – experimental recording (e.g. GoPro .MP4)
#   video2          – simulation/reference video (.mp4)
#   --flip          – horizontally flip video1 after perspective correction
#   --no-homography – skip perspective correction (trim + flip only)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── parse arguments ──────────────────────────────────────────────────────────
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <video1> <video2> [--flip] [--no-homography]"
    exit 1
fi

VIDEO1="$(realpath "$1")"; shift
VIDEO2="$(realpath "$1")"; shift

EXTRA_ARGS=()
while [ "$#" -gt 0 ]; do
    EXTRA_ARGS+=("$1")
    shift
done

# Derive expected output names
BASE1="${VIDEO1%.*}"
EXT1="${VIDEO1##*.}"
VIDEO1_CORRECTED="${BASE1}_corrected.${EXT1}"
VIDEO1_CROPPED="${BASE1}_corrected_cropped.${EXT1}"
VIDEO2_CROPPED="${VIDEO2%.*}_cropped.${VIDEO2##*.}"
VIDEO2_CORRECTED="${VIDEO2%.*}_cropped_corrected.${VIDEO2##*.}"

# # ── Step 1: correct_video1.py ────────────────────────────────────────────────
# echo "========================================"
# echo "Step 1 – correct_video1.py"
# echo "========================================"
# python "$SCRIPT_DIR/correct_video1.py" "$VIDEO1" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"

# if [ ! -f "$VIDEO1_CORRECTED" ]; then
#     echo "ERROR: expected output not found: $VIDEO1_CORRECTED"
#     exit 1
# fi

# ── Step 2: crop & trim picker ───────────────────────────────────────────────
echo ""
echo "========================================"
echo "Step 2 – crop & trim"
echo "========================================"
python "$SCRIPT_DIR/crop_and_trim.py" "$VIDEO1_CORRECTED" "$VIDEO2"

# ── Step 3: match_scale_video2.py ────────────────────────────────────────────
echo ""
echo "========================================"
echo "Step 3 – match_scale_video2.py"
echo "========================================"
python "$SCRIPT_DIR/match_scale_video2.py" "$VIDEO1_CROPPED" "$VIDEO2_CROPPED"

# ── Step 4: vertical stack (ffmpeg) ──────────────────────────────────────────
echo ""
echo "========================================"
echo "Step 4 – stack videos"
echo "========================================"

OUT_DIR="$(dirname "$VIDEO1")"
STACKED="$OUT_DIR/stacked_output.mp4"

# Temporal sync offset written by match_scale_video2.py
VIDEO1_START=0
ENVFILE="$OUT_DIR/video_sync.env"
if [ -f "$ENVFILE" ]; then
    source "$ENVFILE"
    echo "Applying video1 start offset: ${VIDEO1_START}s"
fi

ffmpeg -y \
    -ss "$VIDEO1_START" -i "$VIDEO1_CROPPED" \
    -i "$VIDEO2_CORRECTED" \
    -filter_complex "[0:v]fps=30,scale=1920:-2[top];[1:v]fps=30,scale=1920:-2[bot];[top][bot]vstack=inputs=2[out]" \
    -map "[out]" \
    -c:v libx264 -crf 18 -preset fast \
    "$STACKED"

echo "Done: $STACKED"
echo ""
echo "Pipeline complete."
