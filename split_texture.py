import glob
import os

import cv2
from random_name import random_name 

input_dir = './moss_textures'
output_dir = './result'
height, width = 256, 256

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

image_files = glob.glob(os.path.join(input_dir, '*'))
print(image_files)

for i_f in image_files:
    image = cv2.imread(i_f)
    im_height, im_width, _ = image.shape[:3]
    split_height, split_width = im_height//height, im_width//width

    for s_h in range(split_height):
        for s_w in range(split_width):
            w = s_w * height
            h = s_h * width
            c = image[h:h+height, w:w+width, :]
            cv2.imwrite(output_dir+'/moss_'+ random_name(10) +'.png', c)