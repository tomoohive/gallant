import cv2
import numpy as np
import json

from scipy import ndimage
from skimage import measure
from skimage.segmentation import clear_border
from skimage.filters import threshold_otsu

import collections as cl
import sys
import os
import glob


def info():
    tmp = cl.OrderedDict()
    tmp["description"] = "This is Moss Dataset"
    tmp["url"] = "Nothing"
    tmp["version"] = "0.01"
    tmp["year"] = 2022
    tmp["contributer"] = "tomoohive"
    tmp["data_criation"] = "2022/01/30"
    return tmp


def licenses():
    tmp = cl.OrderedDict()
    tmp["id"] = 1
    tmp["url"] = "Nothing"
    tmp["name"] = "tomoohive"
    return tmp


def images(image_paths):
    tmps = []

    for i, file in enumerate(image_paths):
        img = cv2.imread(file, 0)
        height, width = img.shape[:3]

        tmp = cl.OrderedDict()
        tmp["license"] = 1
        tmp["id"] = i
        tmp["file_name"] = os.path.basename(file)
        tmp["file_path"] = file
        tmp["width"] = width
        tmp["height"] = height
        tmp["date_captured"] = ""
        tmp["coco_url"] = "Nothing"
        tmp["flickr_url"] = "Nothing"
        tmps.append(tmp)

    return tmps


def annotations(mask_image_paths):
    tmps = []

    files = mask_image_paths
    
    for i, file in enumerate(files):
        img = cv2.imread(file, 0)
        tmp = cl.OrderedDict()
        contours = measure.find_contours(img, 0.5)
        segmentation_list = []

        for contour in contours:
            for a in contour:
                segmentation_list.append(a[0])
                segmentation_list.append(a[1])


        mask = np.array(img)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]
        masks = mask == obj_ids[:, None, None]
        num_objs = len(obj_ids)
        boxes = []

        for j in range(num_objs):
            pos = np.where(masks[j])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        tmp["segmentation"] = [segmentation_list]
        tmp["id"] = str(i)
        tmp["image_id"] = i
        tmp["category_id"] = 1
        tmp["area"] = float(boxes[0][3] - boxes[0][1]) * float(boxes[0][2] - boxes[0][0])
        tmp["iscrowd"] = 0
        tmp["bbox"] =  [float(boxes[0][0]), float(boxes[0][1]), float(boxes[0][3] - boxes[0][1]), float(boxes[0][2] - boxes[0][0])]
        tmps.append(tmp)

    return tmps


def categories():
    tmps = []
    sup = ["moss"]
    cat = ["moss"]
    for i in range(len(sup)):
        tmp = cl.OrderedDict()
        tmp["id"] = i+1
        tmp["name"] = cat[i]
        tmp["supercategory"] = sup[i]
        tmps.append(tmp)
    return tmps


def _get_dataset_paths(directory):
    files = os.listdir(directory)
    file_paths = []
    for f in files:
        file_paths.append(os.path.join(directory, f))
    image_paths = []
    mask_image_paths = []
    for f in file_paths:
        mask_dir_path = os.path.join(f, "mask")
        mask_image_path = os.listdir(mask_dir_path)
        image_path = glob.glob(f+'/??????????.png')
        for m in mask_image_path:
            mask_image_paths.append(os.path.join(mask_dir_path, m))
            image_paths.append(image_path[0])

    return image_paths, mask_image_paths


def main(directory, json_name):
    image_paths, mask_image_paths = _get_dataset_paths(directory)
    query_list = ["info", "licenses", "images", "annotations", "categories", "segment_info"]
    js = cl.OrderedDict()
    for i in range(len(query_list)):
        tmp = ""

        if query_list[i] == "info":
            tmp = info()

        elif query_list[i] == "licenses":
            tmp = licenses()

        elif query_list[i] == "images":
            tmp = images(image_paths)

        elif query_list[i] == "annotations":
            tmp = annotations(mask_image_paths)

        elif query_list[i] == "categories":
            tmp = categories()

        js[query_list[i]] = tmp

    fw = open(json_name,'w')
    json.dump(js,fw,indent=2)

if __name__ == '__main__':
    args = sys.argv
    directory = args[1]
    json_name = args[2]
    main(directory, json_name)