import torch
import os
import pandas as pd
from PIL import Image
from torch.utils import data
import glob


class NailDataset(data.Dataset):
    def __init__(self, folder, transforms):
        self.folder = folder
        self.transforms = transforms
        self.root = 'images'
        self.data = pd.read_csv(os.path.join(self.root, folder + '_labels.csv'))
        all_img_pats = glob.glob(os.path.join(self.root, folder) + '/*.jpg')
        self.imgs = []

        for img_path in all_img_pats:
            img_name = img_path.split('/')[-1]
            self.imgs.append(img_name)

    def __getitem__(self, idx):
        boxes = []
        image_name = self.imgs[idx]

        img_path = os.path.join(self.root, self.folder, image_name)
        img = Image.open(img_path).convert("RGB")

        print(img)

        rows_by_img = self.data.loc[self.data['filename'] == image_name]
        num_objs = len(rows_by_img)

        for index, row in rows_by_img.iterrows():
            boxes.append([int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # there is only one class
        labels = torch.ones((len(rows_by_img),), dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
