# Cellular Image Classification

## Overview
This repository contains code to train an Efficientnet model using Fastai/PyTorch to classify the siRNA treatment from cellular images. 

## Dataset
This dataset was provided by Recursion Pharmaceuticals for their Kaggle competition and can be downloaded from the website or the Kaggle Client:
https://www.kaggle.com/c/recursion-cellular-image-classification/overview

The company used 1,108 different siRNA perturbations on four different cell types, and provided six channels of imaging per-sample in separate .png files. The goal of this competition is to predict the genetic perturbation used on each sample in an effort to disentangle noise from real biological signal from the siRNA knockdown.

## Training the model
I initialized the model weights with a pre-trained Efficientnet model, but did not utilize transfer learning as the first convolutional layer had to be changed to from three to six input channels. To initialize the weights on the additional input channels, I used an average of the existing weights from the first convolutional layer in the pretrained model. This approach was inspired by this notebook: https://www.kaggle.com/tanlikesmath/rcic-fastai-starter.
Each sample in this dataset had images taken from two different sites. I split the training data into a train and validation set, and first trained the model on site1 for all training samples for 18 epochs with a learning rate of 1e-3. This resulted in a final accuracy of 40%. I then loaded the state dict back in for another training run on site2 for 14 epochs. After the additional training round on the other half of the dataset, the model had an accuracy of 52%.

## How to use this code
The code to re-create this model is in ```ImageClassification.py```. The command line arguments are as follows:
```
-p, --path    working directory
-s, --site    imaging site to train on [1, 2, or both]
-t, --train   train the model
-ev, --eval   make predictions on test data and write to file
-e, --epochs  number of training epochs
-lr, --learnrate  learning rate to use in training
-ls, --loadstate  file name of state dict (.pth) to initialize weights, optional
-ex, --export     file name to export model pickle file and state dict after training, optional
```

The runtime is approximately 4-5 hours with a GPU accelerator on one site of the training data.
The dependencies can be found in ```requirements.txt```.

## Making predictions from API
I deployed a fastapi-based API on Heroku to make predictions from the trained model. More information about the API can be found here:
https://github.com/sara-keeble/rcic/

The application can be accessed using this URL:
https://rcic.herokuapp.com/docs
