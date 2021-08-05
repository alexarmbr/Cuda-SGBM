import cv2
import os
from glob import glob


files =  glob("2010_03_09_drive_0023/2010_03_09_drive_0023_Images/*.png")
resize_path = "2010_03_09_drive_0023/2010_03_09_drive_0023_Images_resize"
if not os.path.exists(resize_path):
    os.mkdir(resize_path)

scale_percent = 60 # percent of original size
for f in files:
    im = cv2.imread(f)
    width = int(im.shape[1] * scale_percent / 100)
    height = int(im.shape[0] * scale_percent / 100)
    dim = (width, height)
    im = cv2.resize(im, dim)
    cv2.imwrite(os.path.join(resize_path, os.path.basename(f)), im)
