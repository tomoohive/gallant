import os
import glob
import numpy as np
from PIL import Image
import torch

class MossDataset(object):
    def __init__(self, directory):
        self.directory = directory

        self.image_paths, self.mask_image_paths = self._get_dataset_paths(directory)
        print(len(self.image_paths))

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_image_path = self.mask_image_paths[idx]

        img = Image.open(image_path).convert("RGB")

        mask = Image.open(mask_image_path)
        mask = np.array(mask)
        obj_ids = np.unique(mask)
        masks = mask == obj_ids[:, None, None]

        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float)
        labels = torch.ones((num_objs, ), dtype=torch.int64)
        masks = torch.as_tensor(masks, type=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs, ), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return img, target


    def _get_dataset_paths(self, directory):
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


    def __len__(self):
        return len(self.image_paths)



# directory = "/home/tomoohive/workspace/gallant/gallant/test_result"
# files = os.listdir(directory)
# file_paths = []
# for f in files:
#     file_paths.append(os.path.join(directory, f))

# image_paths = []
# mask_image_paths = []
# for f in file_paths:
#     mask_dir_path = os.path.join(f, "mask")
#     mask_image_path = os.listdir(mask_dir_path)
#     image_path = glob.glob(f+'/??????????.png')
#     for m in mask_image_path:
#         mask_image_paths.append(os.path.join(mask_dir_path, m))
#         image_paths.append(image_path[0])

mask_path = '/home/tomoohive/workspace/gallant/gallant/test_result/CAWHRW6PTY/mask/mask_crop8.png'
mask = Image.open(mask_path)
mask = np.array(mask)

obj_ids = np.unique(mask)
obj_ids = obj_ids[1:]
masks = mask == obj_ids[:, None, None]

pos = np.where(masks[0])
print(pos[0])

"""
1. ファイル名取得
2. パス取得
3. 元画像取得
4. マスク画像取得
5. 同じ数のペアになるように配列を取得
6. 結合
"""