import torch
from dataloader import MossDataset
import transforms as T
from PIL import Image, ImageDraw

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from model import get_model_instance_segmentation
from get_transform import get_transform


if __name__ == "__main__":
    
    directory = "/home/tomoohive/workspace/gallant/gallant/test_result"
    dataset_test = MossDataset(directory, get_transform(train=False))
    device = torch.device('cpu')

    num_classes = 2
    model = get_model_instance_segmentation(num_classes)
    model.to(device)
    model.load_state_dict(torch.load('test9.pth', map_location=device))

    img, _ = dataset_test[0]
    model.eval()
    with torch.no_grad():
        prediction = model([img.to(device)])

    im = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())

    draw = ImageDraw.Draw(im)
    draw.rectangle(prediction[0]['boxes'][2].cpu().numpy())
    im.save('result.png', quality=100)

    im2 = Image.fromarray(prediction[0]['masks'][2, 0].mul(255).byte().cpu().numpy())
    im2.save('result1.png', quality=100)

    print(prediction[0]['masks'][0].cpu().numpy())