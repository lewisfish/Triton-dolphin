# :dolphin: DolphinCounterML :dolphin:
Machine learning project to classify dolphins from drone footage using image and velocity information.

## Dataset 
Objects generated from https://github.com/lewisfish/dolphin-counter and labelled by human expert.

![Example of dolphin detection](https://raw.githubusercontent.com/lewisfish/dolphin-counter/master/example.png)


## Usage

classify-main.py [-h] [-e N] [-c] [-i] [-o] [-t] [--modelpath MODELPATH] [--model {resnet,densenet,vgg}] [--type {CNN,Triton}] [-lr LR] [--optim {Adam,AdamW,Adadelta,SGD}] [-w WEIGHT] [-bs BATCHSIZE] [-f FEATURES] [-test]

PyTorch pipeline for train and detection of 2 image classes.

optional arguments:\
  -h, --help            show this help message and exit\
  -e N, --epochs N      number of epochs to train for (default: 10)\
  -c, --continue_train  For continuing training.\
  -i, --evaluate        Infere on a bunch of images.\
  -o, --optimise        Tune the hyperparameters of the chosen model.\
  -t, --train           Train the chosen model.\
  --modelpath MODELPATH Path to saved model.\
  --model {resnet,densenet,vgg} Choice of CNN model for either CNN or Triton model.\
  --type {CNN,Triton}   Choice to run CNN or full Triton model.\
  -lr LR                Learning rate for optimiser\
  --optim {Adam,AdamW,Adadelta,SGD} Choice of optimiser.\
  -w WEIGHT, --weight WEIGHT Weight to bias loss function. Only biases the Dolphin class.\
  -bs BATCHSIZE, --batchsize BATCHSIZE Batchsize for training, evaluation etc.\
  -f FEATURES, --features FEATURES Number of features to use from CNN in Triton model.\
  -test                 If supplied then run on test data instead of validation data.\

## Installation

Install required packages

Using Anaconda:
 conda env create -f environment.yml


## Requirments

  - pytorch
  - torchvision
  - cudatoolkit=10.1
  - matplotlib
  - opencv
  - pandas
  - scikit-image
  - scikit-learn 
