
## Data augmentation 
The accuracy of deep learning models largely depends on the quality, quantity, and contextual meaning of training data. However, data scarcity is one of the most common challenges in building deep learning models. In production use cases, collecting such data can be costly and time-consuming. For our problem, incorporating the test dataset scenario into our training model 4

Companies leverage a low-cost and effective method—data augmentation to reduce dependency on the collection and preparation of training examples and build high-precision AI models quicker.

I wrote the entire pipeline from scratch keeping in account with our problem statement. Building a data augmentation pipeline with the ResNet background was crucial.

<img width="822" alt="Screen Shot 2022-06-11 at 1 35 59 PM" src="https://user-images.githubusercontent.com/50242614/173204060-47e2204d-9485-45eb-9500-a0aaac66d944.png">


<img width="774" alt="Screen Shot 2022-06-11 at 1 36 26 PM" src="https://user-images.githubusercontent.com/50242614/173204076-fd9c662e-d702-4eab-83c8-29b6c8179dcd.png">


Folder contains :
> stanforddogs_train&test - renet50 architecture without Data Augmentation, trained and tested on Stanford dogs datset 
  > resnet50_noDataAug - renet50 architecture without Data Augmentation, trained on Stanford dogs datset, tested on Wyze Dataset
    > wyze_test - renet50 architecture with Data Augmentation, trained on Stanford dogs datset, tested on Wyze Dataset
      > youtube_googleImages - renet50 architecture with Data Augmentation, trained on Stanford dogs datset, tested on Google and Youtube Dataset
