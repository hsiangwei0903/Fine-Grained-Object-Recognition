## Fine-Grained Object Recognition Project - Ravi's Work

This is the UW Capstone Project funded by Wyze Labs and the UW ECE department.


This branch contains all the work done by Ravi.


Notebooks folder contains the code for benchmarking pretrained (on ImageNet) models on our curated dataset, which includes Wyze-cam images, Google images, and YouTube-captured-by-Wyze-cam images.


TrainingAndTent notebook is my code for training a pretrained resnet on the stanford dogs dataset.


The tent folder is the code from https://github.com/DequanWang/tent based on https://arxiv.org/pdf/2006.10726.pdf. This is code for parameter modulation at time of inference, which is also known as test-time adaption. Client.py is the implementation of this technique on my trained resnet.



## Installation
```
git clone https://github.com/hsiangwei0903/Fine-Grained-Object-Recognition

```
