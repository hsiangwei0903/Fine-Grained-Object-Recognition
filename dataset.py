import os
import torch
import pandas as pd
import numpy as np
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, random_split, DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.utils import make_grid
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from collections import OrderedDict

# random_seed = 45
# torch.manual_seed(random_seed)
# train_size = len(dataset)
# print(train_size)

# img, label = dataset[20579]
# print(dataset.classes[label])
# plt.imshow(img)
# print(type(img))
# plt.show()


class DogBreedDataset(Dataset):

    def __init__(self, ds, transform=None):
        self.ds = ds
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, label = self.ds[idx]
        if self.transform:
            img = self.transform(img)
            return img, label


class Transform():
    def make_transform(type):
        if (type == "train"):
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224, padding=4, padding_mode='reflect'),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomRotation(degrees=30),
                transforms.ToTensor(),
            ])
            return transform
        if (type == "test" or type == "val"):
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
            return transform
