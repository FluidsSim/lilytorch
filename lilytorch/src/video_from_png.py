

import cv2
import os
import numpy as np
from lilytorch.util.yaml_operations import yaml2pyobject


dir         = "/data/andreaferrario/ns_data/2025-12-05T11:07:03.437250/curl/"
name        = "video"
img_name    = "curl"
format      = ".mp4"
dt          = 0.0001
slow_factor = 1
save_every  = 500
tstop       = 2000
video_name  = dir+name+format



# dir         = "/data/andreaferrario/ns_data/2025-10-16T19:46:01.461470/particle_images/"
# name        = "video"
# format      = ".mp4"
# img_name    = "particles"
# dt          = 0.0001
# slow_factor = 1
# save_every  = 50
# tstop       = 20
# video_name = dir+name+format



images = [img for img in os.listdir(dir) if img.split("_")[0]==img_name]
n=len(images)
images_sorted = [img_name+"_"+str(i*save_every)+".png" for i in range(n)]


# numbers = [img.split(".")[0].split("_")[1] for img in os.listdir(dir) if img.startswith(img_name+"_")]
# sorted_numbers = np.sort([num for num in numbers])
# images_sorted = [img_name+"_"+str(number)+".png" for number in sorted_numbers]

fps=slow_factor/(save_every*dt)

frame = cv2.imread(os.path.join(dir, images_sorted[0]))
height, width, layers = frame.shape

_fourcc = cv2.VideoWriter_fourcc(*'avc1')

video = cv2.VideoWriter(video_name, _fourcc, fps, (width,height))
font = cv2.FONT_HERSHEY_SIMPLEX
for idx, image in enumerate(images_sorted):
    frame = cv2.imread(os.path.join(dir, image))
    # from IPython import embed; embed()
    # print(asd)
    print(os.path.join(dir, image))
    iteration = idx*save_every
    time  = idx*save_every*dt*slow_factor
    if time>tstop:
        break
    cv2.putText(frame,
                'Time = {}, {}X '.format(round(time,1),round(slow_factor,2)),
                (int(width/2), 50),
                font, 1,
                (0, 0, 0),
                2,
                cv2.LINE_4)
    video.write(frame)

cv2.destroyAllWindows()
video.release()

