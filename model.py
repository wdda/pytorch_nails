import torchvision
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
import os
import cv2
from nail_dataset import NailDataset
import transforms as T

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def collate_fn(batch):
    return tuple(zip(*batch))


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, progress=True)
# replace the classifier with a new one, that has
# num_classes which is user-defined
num_classes = 2  # 1 class (person) + background
# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

dataset = NailDataset('train', get_transform(train=True))

data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, num_workers=4,
    collate_fn=collate_fn)

# For Training
images, targets = next(iter(data_loader))
images = list(image for image in images)
print(images)

targets = [{k: v for k, v in t.items()} for t in targets]
output = model(images, targets)  # Returns losses and detections

# For inference
model.eval()
x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
predictions = model(x)  # Returns predictions

test = pd.read_csv('images/test_labels.csv')
train = pd.read_csv('images/train_labels.csv')
test_count = len(test) - 1
train_count = len(train) - 1


def show_box(image_name, folder):
    img = os.path.join('images/' + folder + '/', image_name)

    if folder == 'test':
        data = test
    else:
        data = train

    if os.path.isfile(img):
        rows_by_img = data.loc[data['filename'] == image_name]
        image = cv2.imread(img)

        for index, row in rows_by_img.iterrows():
            start_point = (int(row['xmin']), int(row['ymin']))
            end_point = (int(row['xmax']), int(row['ymax']))
            cv2.rectangle(image, start_point, end_point, (0, 0, 255), 2)

        cv2.imshow("Output", image)
        cv2.waitKey(0)
    else:
        print("Image not exist")
