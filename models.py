from typing import Tuple

from imblearn.combine import SMOTETomek
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

__main__ = ["get_resnet50", "Frankenstein"]


def _get_resnet50_frank(features):

    # get pretrained model
    model = torchvision.models.resnet50(pretrained=True)
    # freeze layers
    for param in model.parameters():
        param.requires_grad = False

    # swap out final layer so has the correct number of classes.
    model.fc = nn.Sequential(nn.Linear(2048, 512),
                             nn.ReLU(),
                             nn.Dropout(0.2),
                             nn.Linear(512, features))

    return model


def get_resnet50(num_classes):

    # get pretrained model
    model = torchvision.models.resnet50(pretrained=True)
    # freeze layers
    for param in model.parameters():
        param.requires_grad = False

    # swap out final layer so has the correct number of classes.
    model.fc = nn.Sequential(nn.Linear(2048, 512),
                             nn.ReLU(),
                             nn.Dropout(0.2),
                             nn.Linear(512, num_classes),
                             nn.LogSoftmax(dim=1))

    return model


def trainKNN(traindata: Tuple[np.ndarray, np.ndarray]) -> KNeighborsClassifier:

    Xtrain, Ytrain = traindata

    sample = SMOTETomek(random_state=49, sampling_strategy='minority')
    Xtrain_sample, Ytrain_sample = sample.fit_sample(Xtrain, Ytrain)

    # maximum on 200 neighbours with 0.7326 b.accuracy for hdbscan
    # n_jobs=-1 to utilize all cores
    knn = KNeighborsClassifier(n_neighbors=200, n_jobs=-1)
    knn.fit(Xtrain_sample, Ytrain_sample.values.ravel())

    return knn


class Frankenstein(nn.Module):
    """docstring for Frankenstein"""
    def __init__(self, imageModel, dataModel, imageModelargs, dataModelargs):
        super(Frankenstein, self).__init__()
        self.imageModel = imageModel(imageModelargs)
        self.dataModel = dataModel(dataModelargs)

        self.fc1 = nn.Linear(imageModelargs.features + 1, 4)
        self.fc2 = nn.Linear(4, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, image, data):

        x1 = self.imageModel(image)
        x2 = self.dataModel(data)
        x = torch.cat((x1, x2), dim=1)

        x = F.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))

        return x
