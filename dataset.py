import os
from glob import glob

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

# image transform for train and test


class Transformer():

    __transform_set = [
        transforms.RandomRotation(90),
        transforms.ColorJitter(),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=(5, 9)),
        transforms.RandomInvert()
    ]

    # image transform for train
    __pre_data_transforms = transforms.Compose([
        transforms.RandomApply(__transform_set, p=0.5),
        transforms.Resize((224, 224)),
    ])

    # image transform for valid and test
    __data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    def pre_transform(self, img):
        return self.__pre_data_transforms(img)

    def data_transforms(self, img):
        return self.__data_transforms(img)


class MyDataset(Dataset):  # for training
    def __init__(self, path):

        self.data = []

        for img_name in glob(os.path.join(path, '*.jpg')):
            label = 1.0 if os.path.basename(img_name).split('.')[
                0] == 'dog' else 0.0
            self.data.append((img_name, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = Transformer().data_transforms(Image.open(self.data[idx][0]))
        return data, self.data[idx][1]


class Eval_TextDataset(Dataset):  # for evaluating dataset
    def __init__(self, path, csv):

        self.data = []

        for img_id in csv['id']:
            img_path = os.path.join(path, f'{img_id}.jpg')
            self.data.append(Image.open(img_path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = Transformer().data_transforms(self.data[idx])
        return data
