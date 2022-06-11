## Fine-Grained Object Recognition Project

This is the UW Capstone Project funded by [Wyze Labs](https://www.wyze.com/) and the [UW ECE department](https://www.ece.uw.edu/).


<img width="416" alt="Screen Shot 2022-06-11 at 1 48 16 PM" src="https://user-images.githubusercontent.com/50242614/173204388-9a35c66f-c26c-4402-bb6e-89ed356e4fc7.png">


# Installation :  
python==3.8 torch>=1.7.0 torchvision>=0.8.1

# Q1 - WSDAN: 
PyTorch implementation of WS-DAN (Weakly Supervised Data Augmentation Network) for FGVC (Fine-Grained Visual Classification) (Hu et al., "See Better Before Looking Closer: Weakly Supervised Data Augmentation Network for Fine-Grained Visual Classification", arXiv:1901.09891)
(** More details inside the respective folder**)

<img width="654" alt="Screen Shot 2022-06-11 at 1 28 55 PM" src="https://user-images.githubusercontent.com/50242614/173204254-ebe07f5b-39d0-4e61-ac4b-0d7ac235dcbb.png">


# Q2 - Data augmentation
I wrote the entire pipeline from scratch keeping in account with our problem statement. Building a data augmentation pipeline with the ResNet background was crucial to understand and improve the flat classification baseline methods.(** More details inside the respective folder**) 

The accuracy of deep learning models largely depends on the quality, quantity, and contextual meaning of training data. However, data scarcity is one of the most common challenges in building deep learning models. In production use cases, collecting such data can be costly and time-consuming. 
For our problem, incorporating the test dataset scenario( such as capturing noise, color and saturation hues) into our training model was important to help the model generalise better. 

Companies leverage a low-cost and effective method—data augmentation to reduce dependency on the collection and preparation of training examples and build high-precision AI models quicker.

<img width="820" alt="Screen Shot 2022-06-11 at 1 15 37 PM" src="https://user-images.githubusercontent.com/50242614/173203500-dde0cdcd-eb7f-4b2d-8456-6de0dcef714c.png">

