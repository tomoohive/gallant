import numpy as np
import cv2
from random_name import random_name
from split_texture import split_texture


def _save_images(crops: list):
    for c in crops:
        cv2.imwrite(output_dir+'/moss_'+ random_name(10) +'.png', c)


# def _rotate_image(image: np.ndarray):
#     rotates = []
#     rotates.append(image)
#     for i in image:
#         rotates.append(np.rot90(i))
#         rotates.append(np.rot90(i, 2))
#         rotates.append(np.rot90(i, 3))
#     return rotates


def augment_image(crops: list):
    augments = []
    for c in crops:
        augments.append(c)
        augments.append(np.rot90(c))
        augments.append(np.rot90(c, 2))
        augments.append(np.rot90(c, 3))
    return augments


if __name__ == '__main__':
    input_dir = './moss_textures'
    output_dir = './result'
    height, width = 512, 512
    crops = split_texture(input_dir=input_dir, output_dir=output_dir, height=height, width=width)
    augments = augment_image(crops=crops)
    # print(augments)
    _save_images(augments)

