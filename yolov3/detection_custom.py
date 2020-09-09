#================================================================
#
#   File name   : detection_demo.py
#   Author      : PyLessons
#   Created date: 2020-05-18
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#   Description : object detection image and video example
#
#================================================================
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import cv2
import numpy as np
import tensorflow as tf
from yolov3.yolov3 import Create_Yolov3
from yolov3.utils import load_yolo_weights, detect_image, detect_video, detect_realtime
from yolov3.configs import *

input_size = YOLO_INPUT_SIZE
Darknet_weights = YOLO_DARKNET_WEIGHTS
if TRAIN_YOLO_TINY:
    Darknet_weights = YOLO_DARKNET_TINY_WEIGHTS

image_path   = "./IMAGES/a.jpg"
video_path   = "./IMAGES/city.mp4"

yolo = Create_Yolov3(input_size=input_size, CLASSES="2_names.txt")
yolo.load_weights("./checkpoints/yolov3_custom") # use keras weights

orig, img, locs = detect_image(yolo, image_path, "x.jpg", input_size=input_size, show=True, CLASSES="2_names.txt", rectangle_colors=(255,0,0))
print("Pokemon(s) Found in Image at:")
print("[Start X, Start Y, End X, End Y]:", locs)
#detect_video(yolo, video_path, './IMAGES/detected.mp4', input_size=input_size, show=False, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
#detect_realtime(yolo, '', input_size=input_size, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255, 0, 0))
