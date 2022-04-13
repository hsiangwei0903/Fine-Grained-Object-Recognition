import tent.tent as tent
import torchvision.models as models
import timm

import urllib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import torch
import os
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


TOTAL_IMAGES_ENTIRE = 0
TOTAL_CORRECT_ENTIRE = 0
print(torch.cuda.is_available())
# grab pretrained densenet model and set up config/transform to be used later on in processing input images
model = timm.create_model('densenet121', pretrained=True)
config = resolve_data_config({}, model=model)
transform = create_transform(**config)

# following github code for "tenting" model
model = tent.configure_model(model)
params, param_names = tent.collect_params(model)
optimizer = torch.optim.SGD(params, lr=1e-3)
print(tent.check_model(model))
tented_model = tent.Tent(model, optimizer)


# Get imagenet class mappings
url, filename = (
    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt", "imagenet_classes.txt")
urllib.request.urlretrieve(url, filename)
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

for folder in os.listdir("./TestImages/test_google"):
    dog_class = folder.split("-")[1]
    dog_class = dog_class.replace("_", " ").lower()
    categories = [category.lower() for category in categories]
    contains = (dog_class in categories)

    print(dog_class + " is a category in ImageNet?: " + str(contains))

    accurate_prediction_counter = 0
    total_number_of_images = 0

    for file in os.listdir("./TestImages/test_google/" + folder):
        if file.endswith("jpg"):
            total_number_of_images += 1
            image_path = "./TestImages/test_google/" + folder + "/" + file
            img = Image.open(image_path).convert('RGB')
            # transform and add batch dimension
            tensor = transform(img).unsqueeze(0)
            # with torch.no_grad():
            out = tented_model(tensor)
            probabilities = torch.nn.functional.softmax(out[0], dim=0)

            accurate_prediction = False
            top1_prob, top1_catid = torch.topk(probabilities, 1)
            print(categories[top1_catid[0]], top1_prob[0].item())
            if(categories[top1_catid[0]] == dog_class):
                accurate_prediction = True
            if (accurate_prediction):
                accurate_prediction_counter += 1
                
    accuracy = (accurate_prediction_counter / total_number_of_images) * 100
    print("Total images: " + str(total_number_of_images))
    print("DenseNet121 had a " + str(accuracy) +
          "% accuracy on images in " + dog_class)
    print("\n")
    TOTAL_IMAGES_ENTIRE += total_number_of_images
    TOTAL_CORRECT_ENTIRE += accurate_prediction_counter

print("\n")
print("accuracy is " + str(TOTAL_CORRECT_ENTIRE/TOTAL_IMAGES_ENTIRE))

