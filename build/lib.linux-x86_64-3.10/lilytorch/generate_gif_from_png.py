


import os
import imageio.v2 as imageio
from lilytorch.util.yaml_operations import yaml2pyobject


def make_video(
    results_path,
    quantity_name = "curluv",
    video_speed   = 1.0,
):
    ''' Make video from png files '''

    # Load parameters
    pars = yaml2pyobject(f"{results_path}/parameters.yaml")

    time_step = pars["solver"]["dt"]
    save_skip = pars["output"]["save_every"]
    video_fps = video_speed / ( time_step * save_skip )

    # Get available files (sorted by iteration)
    quantity_path = f"{results_path}/{quantity_name}"
    get_iteration = lambda f: int(f.split("_")[-1][:-4])

    quantity_files = [
        f"{quantity_path}/{f}"
        for f in os.listdir(quantity_path)
        if f.startswith(quantity_name) and f.endswith(".png")
    ]
    quantity_files = sorted(quantity_files, key=get_iteration)

    # Make video
    video_name= f'{results_path}/video_{quantity_name}_{video_speed:.2f}x.mp4'

    print(f"Making video: {video_name}")

    ims = [imageio.imread(f) for f in quantity_files]
    imageio.mimwrite(video_name, ims, fps=video_fps)

    return


def main():

    dir_path  = "/data/andreaferrario/ns_data/2025-03-06T15:56:06.822587/"
    dir_names = [
        ""
    ]

    for dir_name in dir_names:
        make_video(
            results_path  = f"{dir_path}/{dir_name}",
            quantity_name = "pressure",
            video_speed   = 1.0,
        )



if __name__ == "__main__":
    # main()
    import sys
    make_video(sys.argv[1].replace('\\', '/'), video_speed=float(sys.argv[2]))

