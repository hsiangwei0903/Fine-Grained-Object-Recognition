# Fine-Grained Object Recognition Project

This is the UW Capstone Project funded by [Wyze Labs](https://www.wyze.com/) and the [UW ECE department](https://www.ece.uw.edu/).

This branch contains the work done by Conor Knox.

## Semantically Enhanced Features

`SEF-master` contains the code from https://github.com/cswluo/SEF which is based on https://arxiv.org/pdf/2006.13457.pdf.
The Jupyter Notebook file `SEF_final.ipynb` use parts of the original code and allow you to train and evaluate your own SEF model based on ResNet architecture. `SEF_aug_tent_final.ipynb` combines this work with data augmentation and Tent: Fully Test-Time Adaptation by Entropy Minimization from https://github.com/DequanWang/tent.

## Adjusting for Wyze Cam v3 distortion

`Checkerboard` contains images taken by a Wyze Cam v3. Run `calibrate.py` from within that folder to obtain the camera's parameters K and D.

Use these parameters in `undistort.py` to undistort images taken by the Wyze Cam v3.

## Image data

We used image data of the dog breeds in `stdogs25-classes.txt`for training and evaluating. The corresponding folders in the Stanford Dogs dataset were used for training. We curated 3 custom datasets for testing.

## Installation

`git clone https://github.com/hsiangwei0903/Fine-Grained-Object-Recognition`
