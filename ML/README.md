# :dolphin: DolphinCounterML :dolphin:
Machine learning project to classify dolphins from drone footage using image and velocity information.

## Dataset 
Objects generated from https://github.com/lewisfish/Triton-dolphin/tree/main/label_creation and labelled by a human expert.


## Usage

usage: classify-main.py [-h] [-e N] [-c] [-i] [-o] [-t] [-mp MODELPATH]
                        [-a {resnet,densenet,vgg}] [--type {CNN,Triton}]
                        [-lr LR] [--optim {Adam,AdamW,Adadelta,SGD}]
                        [-w WEIGHT] [-bs BATCHSIZE] [-f FEATURES] [-test]
                        [-s SEED]

PyTorch pipeline for train and detection of 2 image classes, Dolphin and Not Dolphin.

Optional arguments:\
  -h, --help            show this help message and exit.\
  -e N, --epochs N      number of epochs to train for (default: 10).\
  -c, --continue_train  For continuing training.\
  -i, --evaluate        Infere on a bunch of images.\
  -o, --optimise        Tune the hyperparameters of the chosen model.\
  -t, --train           Train the chosen model.\
  -mp MODELPATH, --modelpath MODELPATH Path to saved model.\
  -a {resnet,densenet,vgg}, --arch {resnet,densenet,vgg} Choice of CNN model for either CNN or Triton model.\
  --type {CNN,Triton}   Choice to run CNN or full Triton model.\
  -lr LR                Learning rate for optimiser\
  --optim {Adam,AdamW,Adadelta,SGD} Choice of optimiser.\
  -w WEIGHT, --weight WEIGHT Weight to bias loss function. Only biases the Dolphin class.\
  -bs BATCHSIZE, --batchsize BATCHSIZE Batchsize for training, evaluation etc.\
  -f FEATURES, --features FEATURES Number of features to use from CNN in Triton model.\
  -test                 If supplied then run on test data instead of validation data.\
  -s SEED, --seed SEED  Set seed for the current run. Default is 1.

### Example

To train a CNN:
  - `python classify-main.py -t --epochs 30 --modelpath cnn-model.pth --arch resnet --type CNN --optim Adam -lr 1e-4 --batchsize 16`
  
To infer on the Triton model with DenseNet at triton-model.pth on the test set:
  - `python classify-main.py -i --modelpath triton-model.pth --arch densenet --modeltype Triton --batchsize 32 --features 6 -test`

To tune the hyperparmeters of the model:
  - `python classify-main.py -o` 

## Data

Unfortunately video and image data not currently publically available.
The folder data contains the unsupervised labels, velocities, and labels and bounding boxes of Dolphin and Not Dolphin objects.
Data folder also contains the class weights used in all DL models.
Finally, data/models contains several saved wieghts for the DL models. The best CNN and Triton models and a Triton model trained without any augmentations.

## Installation

Install required packages

Using Anaconda:
 conda env create -f environment.yml


## Requirments

  - pytorch
  - torchvision
  - cudatoolkit=10.1
  - imblearn
  - matplotlib
  - opencv
  - optuna
  - pandas
  - PIL
  - scikit-image
  - scikit-learn 
  - tqdm
