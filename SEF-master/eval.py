import os, sys, time
import pickle as pk
import numpy as np
from pprint import pprint
from utils import mydataloader
from utils import myimagefolder

from undistort import undistort
import cv2

import torch, torchvision
import torch.nn as nn
from torchvision import transforms, datasets, models
import torch.optim as opt
from torch.optim import lr_scheduler
import torch.utils.model_zoo as model_zoo
import torch.multiprocessing as mlp
import torch.utils.tensorboard as tb
import torch.nn.functional as torchf


from utils import imdb
progpath = os.path.dirname(os.path.realpath(__file__))      
sys.path.append(progpath)
import modellearning
import sef



eps = torch.finfo().eps
# device = torch.device("cuda:0" if torch.cuda.is_available() > 0 else "cpu")
device = 'cpu'
# device_name = device.type+':'+str(device.index) if device.type=='cuda' else 'cpu'

###########################################################################################################################
#
#  model zoo and dataset path
#
###########################################################################################################################
datapath = '/content/drive/MyDrive/Capstone/'
modelzoopath = '/content/drive/MyDrive/Capstone/SEF-master/models/'
sys.path.append(os.path.realpath(modelzoopath))
modelpath = os.path.join(progpath, 'models')
resultpath = os.path.join(progpath, 'runs')




###########################################################################################################################
#
#  constructing loading models
#
###########################################################################################################################
# modelname = r'stdogs-net50-att1-lmgm0-entropy0-soft0-lr0.01-imgsz448-bsz32.model' # ResNet-50 base model
# modelname = r'stdogs-net50-att2-lmgm1-entropy1-soft0.05-lr0.01-imgsz448-bsz32.model' # ResNet-50 with SEF
modelname = r'stdogs25-net50-att2-lmgm1-entropy1-soft0.05-lr0.01-imgsz448-bsz32.model' # ResNet-50 with SEF trained on stdogs25
load_params = torch.load(os.path.join(modelpath, modelname), map_location=device)
networkname = modelname.split('-')[1]


################################## loading from models trained on single gpu to initialize params
model_state_dict, train_params = load_params['model_params'], load_params['train_params']
# pprint(train_params)

# datasetname = modelname.split('-',1)[0]
datasetname = 'stdogs25'
nparts = train_params['nparts']
lmgm = train_params['lmgm']
entropy = train_params['entropy_weights']
soft = train_params['soft_weights']
batchsize = train_params['batch_sz']
imgsz = train_params['imgsz']
lr = train_params['init_lr']
if datasetname == 'cubbirds': num_classes = 200
if datasetname == 'vggaircraft': num_classes = 100
if datasetname == 'stdogs': num_classes = 120
if datasetname == 'stcars': num_classes = 196
if datasetname == 'wyzedogs': num_classes = 120
if datasetname == 'wyze': num_classes = 120
if datasetname == 'yt': num_classes = 120
if datasetname == 'google': num_classes = 120
if datasetname == 'stdogs25': num_classes = 25
attention_flag = True if nparts > 1 else False
netframe = 'resnet50' if networkname.find('50') > -1 else 'resnet18'

# resnet with attention    
model = sef.__dict__[netframe](pretrained=False, model_dir=modelzoopath, nparts=nparts, num_classes=num_classes, attention=attention_flag)

# initializing model using pretrained params except the modified layers
model.load_state_dict(model_state_dict, strict=True)
    


###########################################################################################################################
#
#  generating pytorch dataset and dataloader
#
###########################################################################################################################
datasetpath = os.path.join(datapath, datasetname)
# assert imdb.creatDataset(datasetpath, datasetname=datasetname) == True, "Failing to creat train/val/test sets"
if datasetname in ['cubbirds', 'nabirds', 'vggaircraft']:
    data_transform = {
        'trainval': transforms.Compose([
            transforms.Resize((600,600)),
            transforms.RandomCrop((448, 448)),
            transforms.Resize((imgsz,imgsz)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((600,600)),
            transforms.CenterCrop((448, 448)),
            transforms.Resize((imgsz,imgsz)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
# For data captured by Wyze Cam, add undistort transformation to the test transforms
elif datasetname in ['wyzedogs', 'wyze']:
     data_transform = {
        'trainval': transforms.Compose([
            transforms.Resize((imgsz,imgsz)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((imgsz,imgsz)),
            transforms.ToTensor(),
            transforms.Normalize([0.46945122, 0.49808831, 0.51611604], [0.23592932, 0.24718173, 0.25559981])
        ])
    }
else:
    data_transform = {
        'trainval': transforms.Compose([
            transforms.Resize((imgsz,imgsz)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((imgsz,imgsz)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
test_transform = data_transform['test']



###########################################################################################################################
#
#  codes for model prediction and collecting correctly and wrongly predicted image names and labels.
#
###########################################################################################################################
testsplit = myimagefolder.ImageFolder(os.path.join(datasetpath, 'test'), data_transform['test'])
testloader = mydataloader.DataLoader(testsplit, batch_size=32, shuffle=False, num_workers=8)

testsplit_size = len(testsplit)
class_names = testsplit.classes
class_index = testsplit.class_to_idx
image_index = testsplit.imgs


# log_items = r'{}-net{}-att{}-lmgm{}-entropy{}-soft{}-lr{}-imgsz{}-bsz{}.pkl'.format(
#     datasetname, int(networkname[3:5]), nparts, lmgm, entropy, soft, lr, imgsz, batchsize)
    
log_items = r'stdogs-net{}-att{}-lmgm{}-entropy{}-soft{}-lr{}-imgsz{}-bsz{}.pkl'.format(
    int(networkname[3:5]), nparts, lmgm, entropy, soft, lr, imgsz, batchsize)

if not os.path.exists(os.path.join(resultpath, log_items)):
    model.cuda(device)
    test_rsltparams = modellearning.eval(model, testloader, datasetname=datasetname, device=device)
    with open(os.path.join(resultpath, log_items), 'wb') as f:
        pk.dump({'acc': test_rsltparams['acc'], 'good_data': test_rsltparams['good_data'], 'bad_data':test_rsltparams['bad_data'], 'avg_acc':test_rsltparams['avg_acc']}, f)
else:
    with open(os.path.join(resultpath, log_items), 'rb') as f:
        test_rsltparams = pk.load(f)
        
test_rsltparams = modellearning.eval(model, testloader, datasetname=datasetname, device=device)

print('General Acc: {}, Class Avg Acc: {}'.format(test_rsltparams['acc'], test_rsltparams['avg_acc']))

# Visualize activation maps for correct predictions
import numpy as np
import matplotlib.pyplot as plt

good = test_rsltparams['good_data']
img_paths = [good[i][0] for i in range(len(good))]
indxs = [good[i][1] for i in range(len(good))]
# print(img_paths)
# print(indxs)
# weight = model_state_dict['fc.weight']
# print(weight)
# print(type(weight))
# print(len(weight))

x = test_rsltparams['xmaps']
x = np.array(x)
np.save('xmap.npy', x)
paths = test_rsltparams['paths']
print(paths)
print(type(paths))

paths, labels = zip(*test_rsltparams['good_data'])
print(len(paths))
print(len(labels))
# x = test_rsltparams['xlocal']
# x = np.array(x)
# print(np.shape(x))
# print(x)
# print(np.sum(x))
# x = test_rsltparams['xcosin']
# x = np.array(x)
# print(np.shape(x))
# x = test_rsltparams['outputs']
# x = np.array(x)
# print(np.shape(x))


# print(f'Shape of xmaps: {np.shape(x)}')
# # x = np.exp(x) / (np.exp(x) + 1)
# print(np.shape(x[0]))
# y = np.reshape(x[0], (448, 448, 2))
# map1 = y[:,:,0]
# map2 = y[:,:,1]

# fig = plt.figure()
# plt.imshow(map1)
# fig.savefig('my_figure.png')


# # model_state_dict = weights
# weight = np.squeeze(model_state_dict['fc.weight'].data.numpy())


# def return_CAM(feature_conv, weight, class_idx):
#     # generate the class activation maps upsample to 256x256
#     size_upsample = (448, 448)
#     bz, nc, h, w = feature_conv.shape
#     output_cam = []
#     for idx in class_idx:
#         beforeDot =  feature_conv[idx]
#         cam = np.matmul(weight[idx], beforeDot)
#         cam = cam.reshape(h, w)
#         cam = cam - np.min(cam)
#         cam_img = cam / np.max(cam)
#         cam_img = np.uint8(255 * cam_img)
#         output_cam.append(cv2.resize(cam_img, size_upsample))
#     return output_cam

# for i in img_paths:
#     CAMs = return_CAM(test_rsltparams['xmaps'], weight, indxs)
#     img = cv2.imread(i)
#     height, width, _ = img.shape
#     heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
#     result = heatmap * 0.5 + img * 0.5
#     cv2.imwrite(f"CAM-{i}", result)




###################################################################
### Latency and throughput
###################################################################

# # model = model_state_dict
# # device = torch.device('cuda')
# # model.to(device)
# dummy_input = torch.randn(1, 3,448,448,dtype=torch.float)
# starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
# repetitions = 300
# timings=np.zeros((repetitions,1))
# #GPU-WARM-UP
# for _ in range(10):
#   _ = model(dummy_input)
# # MEASURE PERFORMANCE
# with torch.no_grad():
#   for rep in range(repetitions):
#      starter.record()
#      _ = model(dummy_input)
#      ender.record()
#      # WAIT FOR GPU SYNC
#      torch.cuda.synchronize()
#      curr_time = starter.elapsed_time(ender)
#      timings[rep] = curr_time
# mean_syn = np.sum(timings) / repetitions
# std_syn = np.std(timings)
# print(f'Latency: {mean_syn}')


# dummy_input = torch.randn(10, 3,448,448, dtype=torch.float)
# repetitions=100
# total_time = 0
# with torch.no_grad():
#   for rep in range(repetitions):
#      starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
#      starter.record()
#      _ = model(dummy_input)
#      ender.record()
#      torch.cuda.synchronize()
#      curr_time = starter.elapsed_time(ender)/1000
#      total_time += curr_time
# Throughput = (repetitions*10)/total_time
# print('Final Throughput: ', Throughput)