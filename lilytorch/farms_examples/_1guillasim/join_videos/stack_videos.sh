#!/bin/bash
# Stack video1 (top) and video2 (bottom) vertically.
# Uses video1_corrected.mp4 if it exists (produced by correct_video1.py),
# otherwise falls back to the original video1.MP4.
# -r before -i forces both inputs to be treated as 30 fps,
# which prevents the 90k time-base in video1 from exploding frame counts.

DIR="$(cd "$(dirname "$0")" && pwd)"

if [ -f "$DIR/video1_corrected.mp4" ]; then
    TOP="$DIR/video1_corrected.mp4"
    echo "Using corrected video1: $TOP"
else
    TOP="$DIR/video1.MP4"
    echo "Using original video1: $TOP (run correct_video1.py to correct it)"
fi

if [ -f "$DIR/video2_scaled.mp4" ]; then
    BOT="$DIR/video2_scaled.mp4"
    echo "Using scaled video2: $BOT"
else
    BOT="$DIR/video2.mp4"
    echo "Using original video2: $BOT (run match_scale_video2.py to scale it)"
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
