# Fine-Grained Object Recognition Project

This is the UW Capstone Project funded by [Wyze Labs](https://www.wyze.com/) and the [UW ECE department](https://www.ece.uw.edu/).
![](dog_classifier.png)

## Updates
TransFG Dog Tracker Released (2022.05.05)
```
python3 track_postprocessing.py
```

## Installation
We are using python==3.8 torch>=1.7.0 torchvision>=0.8.1
```
git clone https://github.com/hsiangwei0903/Fine-Grained-Object-Recognition
cd Fine-Grained-Object-Recognition
git clone https://github.com/ultralytics/yolov5
conda create -n capstone
conda activate capstone
pip install -r requirements.txt
pip install anvil-uplink
```

## Model preparation
1. Download [ViT-B_16 Model](https://drive.google.com/drive/folders/12iHLSfN_zYDwWt2BmR4wwBfV83GUFeAG) and put it in transfg/model
2. Download [transfg.bin](https://drive.google.com/drive/folders/1_fCMORZiUWMCpfdMzc-OLfFNaFYYwths) and put it in transfg/output

## Web app deployment
```
cd transfg
python3 dogclassifier.py
```

And then you can run the inference on the [dog classifier website](https://dog-classifier-capstone.anvil.app/)
