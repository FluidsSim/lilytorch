

import cv2
import os

dir         = "/data/andreaferrario/ns_data/2025-05-01T22:28:47.010218/curl/"
name        = "video"
format      = ".mp4"
img_name    = "curl"
dt          = 0.1
slow_factor =   1
save_every    = 50

video_name = dir+name+format

images = [img for img in os.listdir(dir) if img.split("_")[0]==img_name]
n=len(images)
fps=slow_factor/(save_every*dt)
images_sorted = [img_name+"_"+str(i*save_every)+".png" for i in range(n)]
frame = cv2.imread(os.path.join(dir, images_sorted[0]))
height, width, layers = frame.shape

_fourcc = cv2.VideoWriter_fourcc(*'avc1')

video = cv2.VideoWriter(video_name, _fourcc, fps, (width,height))
font = cv2.FONT_HERSHEY_SIMPLEX
for idx, image in enumerate(images_sorted):
    frame = cv2.imread(os.path.join(dir, image))
    time  = idx*save_every*dt*slow_factor
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

