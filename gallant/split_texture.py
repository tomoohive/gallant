import glob
import os

import cv2
from random_name import random_name


def _save_images(crops: list):
    for c in crops:
        cv2.imwrite(output_dir+'/moss_'+ random_name(10) +'.png', c)


def _crop_texture(image_files: list, height: int, width: int):
    crops = []
    for i_f in image_files:
        image = cv2.imread(i_f)
        im_height, im_width, _ = image.shape[:3]
        split_height, split_width = im_height//height, im_width//width

        for s_h in range(split_height):
            for s_w in range(split_width):
                w = s_w * height
                h = s_h * width
                c = image[h:h+height, w:w+width, :]
                crops.append(c)

    return crops


def split_texture(input_dir: str, output_dir: str, height: int, width: int):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    image_files = glob.glob(os.path.join(input_dir, '*'))
    crops = _crop_texture(image_files=image_files, height=height, width=width)
    return crops
    

if __name__ == '__main__':
    input_dir = './moss_textures'
    output_dir = './result'
    height, width = 512, 512

    split_texture(input_dir=input_dir, output_dir=output_dir, height=height, width=width)