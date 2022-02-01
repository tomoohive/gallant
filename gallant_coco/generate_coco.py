from fileinput import hook_encoded
from turtle import home
import cv2
import numpy as np
import json

from skimage import measure
from tqdm import tqdm

import collections as cl
import sys
import os
import glob
import random


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


def images(data, directory):
    tmps = []

    for i, file in tqdm(enumerate(data)):
        img = cv2.imread(file[0])
        height, width, _ = img.shape

        tmp = cl.OrderedDict()
        tmp["license"] = 1
        tmp["id"] = i
        tmp["file_name"] = os.path.basename(file[0])
        tmp["width"] = width
        tmp["height"] = height
        tmp["date_captured"] = ""
        tmp["coco_url"] = "Nothing"
        tmp["flickr_url"] = "Nothing"
        tmps.append(tmp)

        cv2.imwrite(os.path.join(directory, tmp["file_name"]), img)

    return tmps


def annotations(data):
    tmps = []
    
    for i, file in tqdm(enumerate(data)):
        img = cv2.imread(file[1], 0)
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
    image_and_mask_paths = []
    for f in file_paths:
        mask_dir_path = os.path.join(f, "mask")
        mask_image_path = os.listdir(mask_dir_path)
        image_path = glob.glob(f+'/??????????.png')
        for m in mask_image_path:
            image_and_mask_paths.append((image_path[0], os.path.join(mask_dir_path, m)))

    return image_and_mask_paths


def _make_directory(home_directory):
    coco_dir = home_directory + '/coco2017'
    train_dir = coco_dir + '/train2017'
    val_dir = coco_dir + '/val2017'
    ano_dir = coco_dir + '/annotations'

    os.makedirs(coco_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(ano_dir, exist_ok=True)

    return train_dir, val_dir, ano_dir


def make_coco_dict(data, directory):
    query_list = ["info", "licenses", "images", "annotations", "categories", "segment_info"]
    js = cl.OrderedDict()
    for i in range(len(query_list)):
        tmp = ""

        if query_list[i] == "info":
            tmp = info()

        elif query_list[i] == "licenses":
            tmp = licenses()

        elif query_list[i] == "images":
            tmp = images(data, directory)

        elif query_list[i] == "annotations":
            tmp = annotations(data)

        elif query_list[i] == "categories":
            tmp = categories()

        js[query_list[i]] = tmp

    return js


def main(directory, home_directory):
    image_and_mask_paths = _get_dataset_paths(directory)
    train_dir, val_dir, annotation_dir = _make_directory(home_directory)

    divide_num = int(len(image_and_mask_paths)*0.2)
    random.shuffle(image_and_mask_paths)
    train_data = image_and_mask_paths[:-divide_num]
    val_data = image_and_mask_paths[-divide_num:]
    
    train_json = make_coco_dict(train_data, train_dir)
    val_json = make_coco_dict(val_data, val_dir)

    fw = open(os.path.join(annotation_dir, 'train.json'),'w')
    json.dump(train_json,fw,indent=2)
    fw = open(os.path.join(annotation_dir, 'val.json'),'w')
    json.dump(val_json,fw,indent=2)
    

if __name__ == '__main__':
    directory = "/home/tomoohive/Pictures/test_result"
    home_direcotry = "/home/tomoohive/workspace/pytorch_solov2"
    main(directory, home_direcotry)