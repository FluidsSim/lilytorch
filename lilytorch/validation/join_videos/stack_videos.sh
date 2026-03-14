#!/bin/bash
# ── stack_videos.sh ─────────────────────────────────────────────────────────
# Final step of the join_videos pipeline (standalone version of pipeline.py
# step 6).  Run after correct_video1.py and match_scale_video2.py.
#
# What it does:
#   Vertically stacks video1 (top, experiment) and video2 (bottom, simulation)
#   into stacked_output.mp4 using ffmpeg.  Both are scaled to 1920 px wide.
#   If video_sync.env exists (written by match_scale_video2.py), applies the
#   temporal offset so both videos start at the motion onset frame.
#
# Input:   video1_corrected.mp4 (or video1.MP4), video2_scaled.mp4 (or video2.mp4)
# Output:  stacked_output.mp4  (H.264, CRF 18, 30 fps)
#
# Notes:
#   -r 30 before -i forces both inputs to be treated as 30 fps, which
#   prevents the 90k time-base in GoPro video1 from exploding frame counts.

DIR="$(cd "$(dirname "$0")" && pwd)"


# Find *_corrected.* files for both experiment and simulation
EXP_CORRECTED=$(ls "$DIR"/*_corrected.* 2>/dev/null | head -n1)
SIM_CORRECTED=$(ls "$DIR"/*_corrected.* 2>/dev/null | tail -n1)

if [ -n "$EXP_CORRECTED" ]; then
    TOP="$EXP_CORRECTED"
    echo "Using corrected experiment video: $TOP"
else
    echo "No *_corrected.* experiment video found."
    exit 1
fi

if [ -n "$SIM_CORRECTED" ]; then
    BOT="$SIM_CORRECTED"
    echo "Using corrected simulation video: $BOT"
else
    echo "No *_corrected.* simulation video found."
    exit 1
fi

# Temporal sync: video_sync.env is written by match_scale_video2.py
VIDEO1_START=0
if [ -f "$DIR/video_sync.env" ]; then
    source "$DIR/video_sync.env"
    echo "Applying video1 start offset: ${VIDEO1_START}s"
fi

ffmpeg -y \
    -r 30 -ss "$VIDEO1_START" -i "$TOP" \
    -r 30 -i "$BOT" \
    -filter_complex "[0:v]scale=1920:-2[top];[1:v]scale=1920:-2[bot];[top][bot]vstack=inputs=2[out]" \
    -map "[out]" \
    -c:v libx264 -crf 18 -preset fast \
    "$DIR/stacked_output.mp4"

echo "Done: $DIR/stacked_output.mp4"
