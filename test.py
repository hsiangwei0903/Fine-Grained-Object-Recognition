from __future__ import absolute_import, division, print_function
import torch
import models_trans.configs as configs
from models_trans.modeling import VisionTransformer, CONFIGS
from models_trans.dog_class import classes
import logging
import argparse
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import glob
#import cv2

# load yolov5s
detector = torch.hub.load('ultralytics/yolov5', 'yolov5s')
print('yolov5s loaded')

CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'testing': configs.get_testing(),
}

#training model parameter setting
config = CONFIGS["ViT-B_16"]
config.slide_step = 12
config.split = 'overlap'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#inference data transform
test_transform = transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),transforms.CenterCrop((348, 348)),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

#Pretrained TransFG mdoel prepared
pretrained_model_path = "./output/transfg.bin"
print("loading model from {}".format(pretrained_model_path))
model = VisionTransformer(config, 348, zero_head=True, num_classes=120, smoothing_value=0.0)
if pretrained_model_path is not None:
  pretrained_model = torch.load(pretrained_model_path)['model']
  model.load_state_dict(pretrained_model)
model.to(device)
model.eval()
print('model prepared')

test2 = glob.glob('/home/ubuntu/hsiangwei/Fine-Grained-Object-Recognition/wyze/imgs/*')
path_out = '/home/ubuntu/hsiangwei/Fine-Grained-Object-Recognition/wyze/result2/'

for i in range(len(test2)):
  name = test2[i].replace('/home/ubuntu/hsiangwei/Fine-Grained-Object-Recognition/wyze/imgs/','') # seq name
  if not os.path.exists(path_out+name):
    os.mkdir(path_out+name)
  print('testing ',name)
  test1 = test2[i]+'/*'
  test = sorted(glob.glob(test1))
  model.to(device)
  model.eval()
  for n,f in enumerate(test):
    img = Image.open(f).convert('RGB')
    fig, ax = plt.subplots()
    ax.imshow(img)
    plt.axis('off')
    results = detector(img)
    if len(results.xyxy[0])>0:
      for i in range(len(results.xyxy[0])):
        if results.pandas().xyxy[0].iloc[i]['name']=='dog': # dog detected in the image
          if results.pandas().xyxy[0].iloc[i]['confidence'] > 0.3: # setting confidence threshold
            # get the bounding box
            xmin,ymin,xmax,ymax = results.pandas().xyxy[0].iloc[i]['xmin'],results.pandas().xyxy[0].iloc[i]['ymin'],results.pandas().xyxy[0].iloc[i]['xmax'],results.pandas().xyxy[0].iloc[i]['ymax'] 
            # crop the image
            img_crop = img.crop((results.pandas().xyxy[0].iloc[i]['xmin'],results.pandas().xyxy[0].iloc[i]['ymin'],results.pandas().xyxy[0].iloc[i]['xmax'],results.pandas().xyxy[0].iloc[i]['ymax']))
            x = test_transform(img_crop)
            test_pred = model(x.unsqueeze(0).cuda())
            test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
            probs = torch.nn.Softmax(dim=-1)(test_pred)
            # plot rectangle
            rect = patches.Rectangle((xmin,ymin),xmax-xmin,ymax-ymin, linewidth=3, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            plt.text(xmin, ymin, classes[test_label[-1]]+' '+str(int(round(float(probs[0][int(test_label)]),3)*100))+'%', fontsize = 10,bbox = dict(facecolor = 'red', alpha = 0.5))
        elif results.pandas().xyxy[0].iloc[i]['name']=='cat': # cat detected in the image
          if results.pandas().xyxy[0].iloc[i]['confidence'] > 0.3: # setting confidence threshold
            # get the bounding box
            xmin,ymin,xmax,ymax = results.pandas().xyxy[0].iloc[i]['xmin'],results.pandas().xyxy[0].iloc[i]['ymin'],results.pandas().xyxy[0].iloc[i]['xmax'],results.pandas().xyxy[0].iloc[i]['ymax'] 
            # crop the image
            img_crop = img.crop((results.pandas().xyxy[0].iloc[i]['xmin'],results.pandas().xyxy[0].iloc[i]['ymin'],results.pandas().xyxy[0].iloc[i]['xmax'],results.pandas().xyxy[0].iloc[i]['ymax']))
            # plot rectangle
            rect = patches.Rectangle((xmin,ymin),xmax-xmin,ymax-ymin, linewidth=3, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            plt.text(xmin, ymin, 'cat'+' '+str(int((results.pandas().xyxy[0].iloc[i]['confidence'])*100))+'%', fontsize = 10,bbox = dict(facecolor = 'red', alpha = 0.5))
        # elif results.pandas().xyxy[0].iloc[i]['name']=='person': # cat detected in the image
        #   if results.pandas().xyxy[0].iloc[i]['confidence'] > 0.3: # setting confidence threshold
        #     # get the bounding box
        #     xmin,ymin,xmax,ymax = results.pandas().xyxy[0].iloc[i]['xmin'],results.pandas().xyxy[0].iloc[i]['ymin'],results.pandas().xyxy[0].iloc[i]['xmax'],results.pandas().xyxy[0].iloc[i]['ymax'] 
        #     # crop the image
        #     img_crop = img.crop((results.pandas().xyxy[0].iloc[i]['xmin'],results.pandas().xyxy[0].iloc[i]['ymin'],results.pandas().xyxy[0].iloc[i]['xmax'],results.pandas().xyxy[0].iloc[i]['ymax']))
        #     # plot rectangle
        #     rect = patches.Rectangle((xmin,ymin),xmax-xmin,ymax-ymin, linewidth=3, edgecolor='r', facecolor='none')
        #     ax.add_patch(rect)
        #     plt.text(xmin, ymin, 'human'+' '+str(int((results.pandas().xyxy[0].iloc[i]['confidence'])*100))+'%', fontsize = 10,bbox = dict(facecolor = 'red', alpha = 0.5)) 

    plt.savefig(path_out+name+'/'+'{}.jpg'.format(n),bbox_inches='tight',pad_inches = 0, dpi = 300)