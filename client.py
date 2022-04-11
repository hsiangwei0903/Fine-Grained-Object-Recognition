import tent.tent as tent
import torch
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


# model = models.densenet121()
model = timm.create_model('densenet121', pretrained=True)
model = tent.configure_model(model)
params, param_names = tent.collect_params(model)
optimizer = torch.optim.SGD(params, lr=1e-3)
tented_model = tent.Tent(model, optimizer)
print(tent.check_model(tented_model))
config = resolve_data_config({}, model=model)
transform = create_transform(**config)
print(config)


# Get imagenet class mappings
url, filename = (
    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt", "imagenet_classes.txt")
urllib.request.urlretrieve(url, filename)
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

for folder in os.listdir("./TestImages/test"):
    # print(folder)
    dog_class = folder.split("-")[1]
    dog_class = dog_class.replace("_", " ").lower()
    categories = [category.lower() for category in categories]
    contains = (dog_class in categories)

    print(dog_class + " is a category in ImageNet?: " + str(contains))

    accurate_prediction_counter = 0
    total_number_of_images = 0

    for file in os.listdir("./TestImages/test/" + folder):
        if file.endswith("jpg"):
            total_number_of_images += 1
            image_path = "./TestImages/test/" + folder + "/" + file
            img = Image.open(image_path).convert('RGB')
            # transform and add batch dimension
            tensor = transform(img).unsqueeze(0)
            with torch.no_grad():
                out = tented_model(tensor)
                probabilities = torch.nn.functional.softmax(out[0], dim=0)
                accurate_prediction = False
                top1_prob, top1_catid = torch.topk(probabilities, 1)
                print(categories[top1_catid[0]], top1_prob[0].item())
                if(categories[top1_catid[0]] == dog_class):
                    accurate_prediction = True
                if (accurate_prediction):
                    # print(file + " had its actual category in the top 1 prediction")
                    # print("\n")
                    accurate_prediction_counter += 1
                # else:
                #     print(
                #         file + " did not have its actual category in the top 1 prediction")
                #     print("\n")
    accuracy = (accurate_prediction_counter / total_number_of_images) * 100
    print("Total images: " + str(total_number_of_images))
    print("DenseNet121 had a " + str(accuracy) + "% accuracy on images in " + dog_class)
    print("\n")
    TOTAL_IMAGES_ENTIRE += total_number_of_images
    TOTAL_CORRECT_ENTIRE += accurate_prediction_counter

print("\n")
print("accuracy is " + str(TOTAL_CORRECT_ENTIRE/TOTAL_IMAGES_ENTIRE))
# print("latency is: " + str(average_time))
# print("throughput per 60s is " + str(minute_throughput))
# outputs = tented_model(inputs)  # now it infers and adapts!
