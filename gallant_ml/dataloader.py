import os
import torch
import numpy as np
import glob
from PIL import Image


class MossDataset(object):
    def __init__(self, directory, transforms):
        self.directory = directory
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        # self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        # self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))
        self.imgs, self.masks = self._get_dataset_paths(self.directory)


    def __getitem__(self, idx):
        # load images and masks
        img_path = self.imgs[idx]
        mask_path = self.masks[idx]
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path).convert("L")

        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

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
        return len(self.imgs)


# directory = "/home/tomoohive/workspace/gallant/gallant/test_result"
# dataset = MossDataset(directory)