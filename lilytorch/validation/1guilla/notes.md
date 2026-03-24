
Pipiline to match the particle dye and match exp/model videos

1. Generate particle images:

python plot_particles_robot.py --sim_dir /data/andreaferrario/ns_data/1guilla_dye_experiment/water2 --mode 3d_topview

2. Generate particle video:

python video_postprocess.py /data/andreaferrario/ns_data/1guilla_dye_experiment/water2 --fields particle_images

3. (optional - for better remembering) copy the generated video in a subfolder of


python pipeline.py water_dye_videos/exp_water.MP4 water_dye_videos/sim_water.mp4


# ====== slime in the dark video (Carboxymethylcellulose & Acid Fluoresceine) ====

python plot_particles_robot.py --sim_dir /data/andreaferrario/ns_data/2026-03-16T09:42:21.508718 --mode 3d_topview --particle_color "#CCFF00" --bg_color "#1A40E9" --body_color "#000000"



####


./run_pipeline.sh dye_videos/exp_water.MP4 dye_videos/sim_water.mp4 --no-homography

./run_pipeline.sh dye_videos/exp_slime.MP4 dye_videos/sim_slime.mp4 --no-homography







# Step 1 — compute homography interactively (saved as <video>_homography.npy)
python track_robot.py --homography --video /data/andreaferrario/1guilla_experiments/swim/videos/ms001mpt001.mp4

# Step 2 — calibrate scale on the corrected view
python track_robot.py --calibrate --video /data/andreaferrario/1guilla_experiments/swim/videos/ms001mpt001.mp4 \
    --homography_file /data/andreaferrario/1guilla_experiments/swim/videos/ms001mpt001_homography.npy

# Step 3 — batch process all videos
python3 track_robot.py \
    --video .../ms001mpt001.mp4 \
    --meters_per_pixel 0.002488 \
    --midline_method colwise \
    --homography_file /data/andreaferrario/1guilla_experiments/swim/videos/ms001mpt001_homography.npy



timeout 90 python3 track_robot.py \
    --preview \
    --video /data/andreaferrario/1guilla_experiments/swim/videos/ms001mpt001.mp4 \
    --save_preview /tmp/preview_detection.mp4 2>&1 | grep -v inotify