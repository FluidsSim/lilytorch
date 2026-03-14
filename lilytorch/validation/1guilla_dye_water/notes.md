
Pipiline to match the particle dye and match exp/model videos

1. Generate particle images:

python plot_particles_robot.py --sim_dir /data/andreaferrario/ns_data/1guilla_dye_experiment/water2 --mode 3d_topview

2. Generate particle video:

python video_postprocess.py /data/andreaferrario/ns_data/1guilla_dye_experiment/water2 --fields particle_images

3. (optional - for better remembering) copy the generated video in a subfolder of


python pipeline.py water_dye_videos/exp_water.MP4 water_dye_videos/sim_water.mp4