import torch
from dataloader import MossDataset
import transforms as T
from PIL import Image, ImageDraw

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


if __name__ == "__main__":
    
    directory = "/home/tomoohive/workspace/gallant/gallant/test_result"
    dataset_test = MossDataset(directory, get_transform(train=False))
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    num_classes = 2
    model = get_model_instance_segmentation(num_classes)
    model.to(device)
    model.load_state_dict(torch.load('test.pth'))

    img, _ = dataset_test[0]
    model.eval()
    with torch.no_grad():
        prediction = model([img.to(device)])

    im = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())

    draw = ImageDraw.Draw(im)
    draw.rectangle(prediction[0]['boxes'][0].cpu().numpy())
    im.save('result.png', quality=100)

    im2 = Image.fromarray(prediction[0]['masks'][1, 0].mul(255).byte().cpu().numpy())
    im2.save('result1.png', quality=100)

    print(prediction[0]['masks'][0].cpu().numpy())