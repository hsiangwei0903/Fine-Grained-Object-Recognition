## Fine-Grained Object Recognition Project - Ravi's Work

This is the UW Capstone Project funded by Wyze Labs and the UW ECE department.


This branch contains all the work done by Ravi.


Notebooks folder contains the code for benchmarking pretrained (on ImageNet) models on our curated dataset, which includes Wyze-cam images, Google images, and YouTube-captured-by-Wyze-cam images.


TrainingAndTent notebook is my code for training a pretrained resnet on the stanford dogs dataset. It also includes the implementation of test-time-adaption on the trained model for our Wyze-cam images, and corresponding results. There are a total of three models that were benchmarked: Resnet34 with 10 epochs of fine tune training, Resnet34 with 65 epochs of fine tune training, and Resnet18 with 35 epochs of fine tune training.


The tent folder is the code from https://github.com/DequanWang/tent based on https://arxiv.org/pdf/2006.10726.pdf. This is code for parameter modulation at time of inference, which is also known as test-time adaption. 



## Installation
```
git clone https://github.com/hsiangwei0903/Fine-Grained-Object-Recognition

```
