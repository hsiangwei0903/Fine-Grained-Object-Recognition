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
import cv2
import copy
from scipy.optimize import linear_sum_assignment

def iou(bboxA,bboxB):
    left = max(bboxA[0],bboxB[0])
    right = min(bboxA[2],bboxB[2])
    top = max(bboxA[1],bboxB[1])
    bottom = min(bboxA[3],bboxB[3])

    if right<left or bottom < top:
        return 0.0
    iouArea = (right-left)*(bottom-top)
    bb1_Area = (bboxA[0]-bboxA[2])*(bboxA[1]-bboxA[3])
    bb2_Area = (bboxB[0]-bboxB[2])*(bboxB[1]-bboxB[3])

    return iouArea / float(bb1_Area+bb2_Area-iouArea)

def get_color(idx):
  if idx<=3:
    color_list = [(255,0,0),(0,255,0),(0,0,255)]
    return color_list[idx]
  else:
    idx = (idx+150) * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color

# load yolov5s
detector = torch.hub.load('ultralytics/yolov5', 'yolov5l')
print('yolov5 loaded')

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

seq = 'dogs'

imgs = sorted(glob.glob('/home/ubuntu/hsiangwei/Fine-Grained-Object-Recognition/wyze/{}/*'.format(seq)))
path_out = '/home/ubuntu/hsiangwei/Fine-Grained-Object-Recognition/wyze/result/'

if not os.path.exists(path_out+seq+'/'):
  os.mkdir(path_out+seq+'/')

tracks = []
tracks_name = []
initiate = []
for frame,img in enumerate(imgs):
  #print('processing :{}/{}'.format(i,len(imgs)))
  model.to(device)
  model.eval()
  image = Image.open(img).convert('RGB')
  results = detector(image)
  img = cv2.imread(img)
  im_out = copy.deepcopy(img)
  dets = []
  dets_name = []
  if len(results.xyxy[0])>0:
    for i in range(len(results.xyxy[0])):
        if results.pandas().xyxy[0].iloc[i]['name']=='dog': # dog detected in the image
          if results.pandas().xyxy[0].iloc[i]['confidence'] > 0: # setting confidence threshold
            xmin,ymin,xmax,ymax = results.pandas().xyxy[0].iloc[i]['xmin'],results.pandas().xyxy[0].iloc[i]['ymin'],results.pandas().xyxy[0].iloc[i]['xmax'],results.pandas().xyxy[0].iloc[i]['ymax'] 
            xmin,ymin,xmax,ymax = int(xmin),int(ymin),int(xmax),int(ymax)
            img_crop = image.crop((results.pandas().xyxy[0].iloc[i]['xmin'],results.pandas().xyxy[0].iloc[i]['ymin'],results.pandas().xyxy[0].iloc[i]['xmax'],results.pandas().xyxy[0].iloc[i]['ymax']))
            x = test_transform(img_crop)
            test_pred = model(x.unsqueeze(0).cuda())
            test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
            dets.append([xmin,ymin,xmax,ymax])
            dets_name.append(str(classes[test_label[-1]]))
  
  if len(tracks)==0:
    print('initiate tracking')
    for d_id,det in enumerate(dets):
      tracks.append(det)
      tracks_name.append(dets_name[d_id])
      initiate.append(True)

  else:
    cost_matrix = np.ones([len(tracks),len(dets)])
    for i in range(len(tracks)):
      for j in range(len(dets)):
        if iou(tracks[i],dets[j]) > 0.05:
          cost_matrix[i][j] = 1 - iou(tracks[i],dets[j])
        else:
          cost_matrix[i][j] = 1
      
    row_ind,col_ind = linear_sum_assignment(cost_matrix)

    for n in range(len(initiate)):
      initiate[n] = False

    for row in row_ind:
      for col in col_ind:
        if cost_matrix[row][col]<0.9:
          tracks[row] = dets[col]
          tracks_name[row] = dets_name[col]
          initiate[row] = True
          dets[col] = None
          dets_name[col] = None 
          

    for id,unmatch_det in enumerate(dets):
      if unmatch_det:
        tracks.append(unmatch_det)
        tracks_name.append(dets_name[id])
        initiate.append(True)
        print('initiate new track')
    
  for t_id,trk in enumerate(tracks):
    if initiate[t_id]:
      cv2.rectangle(im_out,(int(trk[0]),int(trk[1])),(int(trk[2]),int(trk[3])),get_color(t_id),4)
      cv2.putText(im_out,str(t_id)+'.'+tracks_name[t_id],(int(trk[0]),int(trk[1])),cv2.FONT_HERSHEY_PLAIN,max(1.0, img.shape[1]/1200),(0,255,255),thickness = 2)  
  
  print('writing img {}'.format(frame))
  cv2.imwrite(path_out+seq+'/{}.jpg'.format("%04d"%frame),im_out)



'''
        if results.pandas().xyxy[0].iloc[i]['name']=='cat': # cat detected in the image
          if results.pandas().xyxy[0].iloc[i]['confidence'] > 0: # setting confidence threshold
            # get the bounding box
            xmin,ymin,xmax,ymax = results.pandas().xyxy[0].iloc[i]['xmin'],results.pandas().xyxy[0].iloc[i]['ymin'],results.pandas().xyxy[0].iloc[i]['xmax'],results.pandas().xyxy[0].iloc[i]['ymax'] 
            xmin,ymin,xmax,ymax = int(xmin),int(ymin),int(xmax),int(ymax)
            img_crop = img[ymin:ymax,xmin:xmax]
            for t_id in tracks:
              if iou(tracks[t_id],[xmin,ymin,xmax,ymax])>0.6:
                tracks[t_id]=[xmin,ymin,xmax,ymax]
                cv2.rectangle(im_out,(int(xmin),int(ymin)),(int(xmax),int(ymax)),get_color[t_id],4)
                cv2.putText(im_out,'cat'+str(t_id),(int(xmin),int(ymin)),cv2.FONT_HERSHEY_PLAIN,max(1.0, img.shape[1]/1200),(0,255,255),thickness = 4)
                initiate = False
                break
            if initiate:
              t_id = len(tracks)
              tracks[t_id] = [xmin,ymin,xmax,ymax]
              cv2.rectangle(im_out,(int(xmin),int(ymin)),(int(xmax),int(ymax)),colors[len(tracks)+1],4)
              cv2.putText(im_out,'cat'+str(t_id),(int(xmin),int(ymin)),cv2.FONT_HERSHEY_PLAIN,max(1.0, img.shape[1]/1200),(0,255,255),thickness = 4)
'''
