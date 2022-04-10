# Fine-Grained Object Recognition Project

This is the University of Washington Capstone Project funded by Wyze Lab and the UW ECE department.

To start the deployment of the project:
1. Download [[ViT-B_16 Model]](https://drive.google.com/drive/folders/12iHLSfN_zYDwWt2BmR4wwBfV83GUFeAG) and put it in transfg/model
2. Download the [[transFG pretrained model]](https://drive.google.com/drive/folders/1_fCMORZiUWMCpfdMzc-OLfFNaFYYwths) and put it in transfg/output

Environment Building
```
git clone https://github.com/hsiangwei0903/Fine-Grained-Object-Recognition.git
cd Fine-Grained-Object-Recognition
git clone https://github.com/ultralytics/yolov5
pip install -r requirements.txt
```

Web app deployment
```
cd transfg
python dogclassifier.py
```
