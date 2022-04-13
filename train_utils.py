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


def print_epoch_result(train_loss, train_acc, val_loss, val_acc):
    print('loss: {:.3f}, acc: {:.3f}, val_loss: {:.3f}, val_acc: {:.3f}'.format(train_loss,
                                                                                train_acc,
                                                                                val_loss,
                                                                                val_acc))
# Main Training function


def train_model(model, cost_function, optimizer, train_dl, val_dl, device, num_epochs=5):
    train_losses = []
    val_losses = []
    train_acc = []
    val_acc = []

    # Metrics object
    # train_acc_object = metrics.Accuracy(compute_on_step=False)
    # val_acc_object = metrics.Accuracy(compute_on_step=False)

    for epoch in range(num_epochs):
        """
        On epoch start
        """
        print('-'*15)
        print('Start training {}/{}'.format(epoch+1, num_epochs))
        print('-'*15)

        # Training
        train_sub_losses = []
        model.train()
        i = 0
        for x, y in train_dl:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = cost_function(y_hat, y)
            loss.backward()
            optimizer.step()
            # lr_scheduler.step()
            # update loss sublist
            train_sub_losses.append(loss.item())
            i += 1
            print(i)
            # update accuracy object
            # train_acc_object(y_hat.cpu(), y.cpu())

        # Validation
        val_sub_losses = []
        model.eval()
        for x, y in val_dl:
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = cost_function(y_hat, y)
            val_sub_losses.append(loss.item())
            # val_acc_object(y_hat.cpu(), y.cpu())

        """
        On epoch end
        """
        # Update the loss list
        train_losses.append(np.mean(train_sub_losses))
        val_losses.append(np.mean(val_sub_losses))

        # Update the accuracy list and reset the metrics object
        # train_epoch_acc = train_acc_object.compute()
        # val_epoch_acc = val_acc_object.compute()
        # train_acc.append(train_epoch_acc)
        # val_acc.append(val_epoch_acc)
        # train_acc_object.reset()
        # val_acc_object.reset()

        # print the result of epoch
        # print_epoch_result(np.mean(train_sub_losses), train_epoch_acc, np.mean(
        #     val_sub_losses), val_epoch_acc)

    print('Finish Training.')
    return train_losses, train_acc, val_losses, val_acc
