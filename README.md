## Fine-Grained Object Recognition Project

This is the UW Capstone Project funded by [Wyze Labs](https://www.wyze.com/) and the [UW ECE department](https://www.ece.uw.edu/).
![](dog_classifier.png)


Installation
We are using python==3.8 torch>=1.7.0 torchvision>=0.8.1

Q1 Folder
This folder contains code used for benchmarking and evaluating ResNet50 and MobileNetV2. Download each Jupyter Notebook (.ipynb) and run the code cells to reproduce the results.

ClassFineTuningExperiment.ipynb
This file contains the experiment done in Q2. Download the Jupyter Notebook to perform the experiment. There are directions in the file tto help reproduce the results.

PythonScripts.ipynb
This file contains scripts (video-to-images, file renaming, google images scraper) used for data processing. Download the Jupyter Notebook to use the scripts. Code and example usages are within the file.



# Q1 - WSDAN: 
PyTorch implementation of WS-DAN (Weakly Supervised Data Augmentation Network) for FGVC (Fine-Grained Visual Classification) (Hu et al., "See Better Before Looking Closer: Weakly Supervised Data Augmentation Network for Fine-Grained Visual Classification", arXiv:1901.09891)
** More details inside the respective folder

<img width="654" alt="Screen Shot 2022-06-11 at 1 28 55 PM" src="https://user-images.githubusercontent.com/50242614/173204254-ebe07f5b-39d0-4e61-ac4b-0d7ac235dcbb.png">



# Q2 - Data augmentation: 
I wrote the entire pipeline from scratch keeping in account with our problem statement. Building a data augmentation pipeline with the ResNet background was crucial to understand and improve the flat classification baseline methods. ** More details inside the respective folder

The accuracy of deep learning models largely depends on the quality, quantity, and contextual meaning of training data. However, data scarcity is one of the most common challenges in building deep learning models. In production use cases, collecting such data can be costly and time-consuming. 
For our problem, incorporating the test dataset scenario into our training model 4 

Companies leverage a low-cost and effective method—data augmentation to reduce dependency on the collection and preparation of training examples and build high-precision AI models quicker.

<img width="820" alt="Screen Shot 2022-06-11 at 1 15 37 PM" src="https://user-images.githubusercontent.com/50242614/173203500-dde0cdcd-eb7f-4b2d-8456-6de0dcef714c.png">

