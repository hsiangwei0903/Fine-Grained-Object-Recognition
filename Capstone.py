import torch
from __future__ import absolute_import, division, print_function
import models_trans.configs as configs
from models_trans.modeling import VisionTransformer, CONFIGS
from models_trans.dog_class import classes
import logging
import argparse
import os
import random
import numpy as np
import pandas as pd
import time
from torch.utils.data import Dataset
from torchvision import transforms
import anvil.server
import anvil.media
import anvil.mpl_util
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

detector = torch.hub.load('ultralytics/yolov5', 'yolov5s')

CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'testing': configs.get_testing(),
}

class FineGrainDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None, test=False):
        self.root_dir = root_dir  # 圖片本人路徑
        self.annotations = pd.read_csv(annotation_file)  # 上一步做的CSV
        self.transform = transform  # 定義要做的transform, 含有resize把圖片先resize成依樣
        self.test = test

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_id = self.annotations.iloc[index, 0]  # 取出image id => ex: 0003.jpg
        if self.test == False:
            img = Image.open(os.path.join(self.root_dir, img_id)).convert(
                "RGB")  # 取出image id 對應的圖片本人, 並且轉RGB(等等用transform來轉tensor)
            temp = self.annotations.iloc[index, 1]
            y_label = torch.tensor(self.annotations.iloc[index, 1]).long()
            img = self.transform(img)
            return (img, y_label - 1)
        else:
            # print("fetch testing image id: ", img_id)
            img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")
            img = self.transform(img)
            return (img, img_id)

#training model parameter setting
config = CONFIGS["ViT-B_16"]
config.slide_step = 12
config.split = 'overlap'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#inference data transform
test_transform = transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),transforms.CenterCrop((348, 348)),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

#Pretrained TransFG mdoel prepared
pretrained_model_path = "./output/transfg.bin"
model = VisionTransformer(config, 348, zero_head=True, num_classes=120, smoothing_value=0.0)
if pretrained_model_path is not None:
  pretrained_model = torch.load(pretrained_model_path)['model']
  model.load_state_dict(pretrained_model)
model.to(device)
model.eval()
print('model prepared')

anvil.server.connect("server_VFL2ORJJKT22YWR7YJEL2AVN-4TTOM7KQI4VYB57T")
print('anvil connected')

@anvil.server.callable
def dog_classifier(file):
  with anvil.media.TempFile(file) as filename: # getting upload images from the anvil server. 
    img = Image.open(filename).convert('RGB') 
    results = detector(img) # using yolov5 to get the detection result
    fig, ax = plt.subplots()
    ax.imshow(img)
    plt.axis('off')

  if len(results.xyxy[0]) == 0: # nothing detected
    plt.savefig('pics/{}{}.png'.format('blank',random.randint(0,1000))) # save images 
    return 'Nothing detected!', 0, anvil.mpl_util.plot_image()

  else:
    dog = 0
    dog_pred = []
    dog_prob = []
    for i in range(len(results.xyxy[0])):
      if results.pandas().xyxy[0].iloc[i]['name']=='dog': # dog detected in the image
        dog = 1
        if results.pandas().xyxy[0].iloc[i]['confidence'] > 0.1: # setting confidence threshold
          # get the bounding box
          xmin,ymin,xmax,ymax = results.pandas().xyxy[0].iloc[i]['xmin'],results.pandas().xyxy[0].iloc[i]['ymin'],results.pandas().xyxy[0].iloc[i]['xmax'],results.pandas().xyxy[0].iloc[i]['ymax'] 
          # crop the image
          img_crop = img.crop((results.pandas().xyxy[0].iloc[i]['xmin'],results.pandas().xyxy[0].iloc[i]['ymin'],results.pandas().xyxy[0].iloc[i]['xmax'],results.pandas().xyxy[0].iloc[i]['ymax']))
          # plot rectangle
          rect = patches.Rectangle((xmin,ymin),xmax-xmin,ymax-ymin, linewidth=3, edgecolor='r', facecolor='none')
          ax.add_patch(rect)
          # image transform
          x = test_transform(img_crop)
          # get prediction
          test_pred = model(x.unsqueeze(0).cuda())
          # get label
          test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
          probs = torch.nn.Softmax(dim=-1)(test_pred)
          dog_pred.append(classes[test_label[-1]])
          dog_prob.append(int(round(float(probs[0][int(test_label)]),3)*100))
          plt.text(xmin, ymin, classes[test_label[-1]]+' '+str(int(round(float(probs[0][int(test_label)]),3)*100))+'%', fontsize = 8,bbox = dict(facecolor = 'red', alpha = 0.5))
    
    if dog == 1:
      # save plot
      plt.savefig('pics/{}{}.png'.format('dog',random.randint(0,1000)))
      # return dog breed that is predicted by the model
      # return return_string,round(float(probs[0][int(test_label)]),3),anvil.mpl_util.plot_image()
      return str(len(dog_pred))+' dogs detected!',float(sum(dog_prob)/len(dog_prob))/100,anvil.mpl_util.plot_image()
    if dog == 0: # no dog detected in the image
      return_string = 'No dog detected, instead detect a {}!'.format(results.pandas().xyxy[0].iloc[0]['name']) # return the object with highest confidence score 
      plt.savefig('pics/{}{}.png'.format(results.pandas().xyxy[0].iloc[0]['name'],random.randint(0,1000)))
      xmin,ymin,xmax,ymax = results.pandas().xyxy[0].iloc[0]['xmin'],results.pandas().xyxy[0].iloc[0]['ymin'],results.pandas().xyxy[0].iloc[0]['xmax'],results.pandas().xyxy[0].iloc[0]['ymax']
      rect = patches.Rectangle((xmin,ymin),xmax-xmin,ymax-ymin, linewidth=3, edgecolor='g', facecolor='none')
      ax.add_patch(rect)
      return return_string , results.pandas().xyxy[0].iloc[0]['confidence'] ,anvil.mpl_util.plot_image()
