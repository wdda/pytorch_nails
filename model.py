import torchvision
import torch
import pandas as pd
import os
import cv2
from nail_dataset import NailDataset
import helpers.transforms as T
from helpers.engine import train_one_epoch, evaluate
import helpers.utils as utils

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def collate_fn(batch):
    return tuple(zip(*batch))


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def testing_forward():
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


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    # in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    # hidden_layer = 256
    # and replace the mask predictor with a new one
    # model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,hidden_layer,num_classes)

    return model


# https://github.com/pytorch/vision/tree/master/references/detection
def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 1
    # use our dataset and defined transformations
    dataset = NailDataset('train', get_transform(train=True))
    dataset_test = NailDataset('test', get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 3

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    print("That's it!")

main()