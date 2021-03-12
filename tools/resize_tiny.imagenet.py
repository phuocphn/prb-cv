import os
import glob
import cv2

def resize_img(image_path, size=224):
    img = cv2.imread(image_path)
    img = cv2.resize(img,(size,size), interpolation = cv2.INTER_CUBIC)
    cv2.imwrite(image_path,img)

DATA_DIR = "/home/phuocphn/data/tiny-imagenet-200-224/*/*/*"
for img in glob.glob(DATA_DIR):
	resize_img(img)
