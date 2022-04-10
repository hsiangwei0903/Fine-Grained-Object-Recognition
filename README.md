## Fine-Grained Object Recognition Project

This is the UW Capstone Project funded by Wyze Labs and the UW ECE department.

## Installation
```
git clone https://github.com/hsiangwei0903/Fine-Grained-Object-Recognition
cd Fine-Grained-Object-Recognition
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
pip install anvil-uplink
```

## Model preparation
1. Download [[ViT-B_16 Model]](https://drive.google.com/drive/folders/12iHLSfN_zYDwWt2BmR4wwBfV83GUFeAG) and put it in transfg/model
2. Download the [[transFG pretrained model]](https://drive.google.com/drive/folders/1_fCMORZiUWMCpfdMzc-OLfFNaFYYwths) and put it in transfg/output
3. Download the [[YOLOv5s detection model]](https://drive.google.com/file/d/100EkA7zlxuQElRKkMBvCwjBdz1Qz1Hd4/view) and put it in yolov5 folder

## Web app deployment
```
cd ../transfg
python dogclassifier.py
```

And then you can run the inference on the [[dog classifier website]](https://dog-classifier-hsiangwei.anvil.app/)
