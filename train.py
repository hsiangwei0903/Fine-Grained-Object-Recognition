import tent.tent as tent
import torchvision.models as models
import timm
from dataset import DogBreedDataset, Transform

import urllib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import torch
import os
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision.utils import make_grid
from train_utils import train_model, print_epoch_result


dataset = ImageFolder(
    "../../../../../../../Downloads/StanfordDogDataset/images/Images")
breeds = []


def rename(name):
    return ' '.join(' '.join(name.split('-')[1:]).split('_'))


for n in dataset.classes:
    breeds.append(rename(n))

random_seed = 45
torch.manual_seed(random_seed)
train_ds, val_ds = random_split(dataset, [len(dataset)-2000, 2000])

train_dataset = DogBreedDataset(train_ds, Transform.make_transform("train"))
val_dataset = DogBreedDataset(val_ds, Transform.make_transform("val"))


batch_size = 16
train_dl = DataLoader(train_dataset, batch_size,
                      shuffle=True, num_workers=2, pin_memory=True)
val_dl = DataLoader(val_dataset, batch_size*2, num_workers=2, pin_memory=True)

device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')

model = timm.create_model('resnet50', pretrained=False, num_classes=120)
model = model.to(device)


# Cost function and optimzier
cost_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0003)

# Start Training
if __name__ == "__main__":
    torch.cuda.empty_cache()
    train_losses, train_acc, val_losses, val_acc = train_model(model=model,
                                                               cost_function=cost_function,
                                                               optimizer=optimizer,
                                                               train_dl=train_dl,
                                                               val_dl=val_dl,
                                                               device=device,
                                                               num_epochs=1)

# Print the result of 1 epoch


# print(model)


# if __name__ == '__main__':

#     def show_batch(dl):
#         for img, lb in dl:
#             fig, ax = plt.subplots(figsize=(16, 8))
#             ax.set_xticks([])
#             ax.set_yticks([])
#             ax.imshow(make_grid(img.cpu(), nrow=16).permute(1, 2, 0))
#             plt.show()
#             break

#     show_batch(train_dl)
# img, label = train_dataset[6]
# print(label)
# plt.imshow(img.permute(1, 2, 0))
# plt.show()

# # following github code for "tenting" model
# model = tent.configure_model(model)
# params, param_names = tent.collect_params(model)
# optimizer = torch.optim.SGD(params, lr=1e-3)
# print(tent.check_model(model))
# tented_model = tent.Tent(model, optimizer)
