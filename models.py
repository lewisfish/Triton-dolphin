from typing import Callable, Tuple, Dict, Union

from imblearn.combine import SMOTETomek
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

__main__ = ["get_densenet121", "get_vgg13_bn", "get_resnet50", "Triton", "trainKNN", "trainSVM", "get_resnet50new"]


def get_vgg13_bn(num_classes, classify=True):
    """Return instance of VGG13_bn with classifier replaced with FC layer

    Parameters
    ----------

    num_classes : int
        The number of classes/features to output

    Returns
    -------

    model : ?
        VGG13_bn instance

    """

    model = torchvision.models.vgg13_bn(pretrained=True)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    return model


def get_densenet121(num_classes, classify=True):
    """Return instance of Densenet121 with classifier replaced with FC layer

    Parameters
    ----------

    num_classes : int
        The number of classes/features to output

    Returns
    -------

    model : ?
        Densenet121 instance

    """

    model = torchvision.models.densenet121(pretrained=True)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)
    return model


def get_resnet50new(num_classes, classify=True):

    model = torchvision.models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model

def get_resnet50(num_classes: int, classify=True):
    """Function returns resnet50 with most layers frozen and changed final layer.

    Parameters
    ----------

    num_classes : int
        Number of classes to classify into.
        Or number of features to return.

    Classify: bool, optional
        Default = True. If true then return verions that can classify
        If False return version with LogSoftmax removed.

    Returns
    -------

    model : ?
        Resnet50 model with frozen layers, and a new FC layers
    """

    # get pretrained model
    model = torchvision.models.resnet50(pretrained=True)
    # freeze layers
    for param in model.parameters():
        param.requires_grad = False

    # swap out final layer so has the correct number of classes.
    if classify:
        model.fc = nn.Sequential(nn.Linear(2048, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, num_classes),
                                 nn.LogSoftmax(dim=1))
    else:
        model.fc = nn.Sequential(nn.Linear(2048, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, num_classes))

    return model


def trainKNN(traindata: Tuple[np.ndarray, np.ndarray]) -> KNeighborsClassifier:
    """Function returns a trained KNeighborsClassifier instance.

    Parameters
    ----------

    traindata : Tuple[np.ndarray, np.ndarray]
        Tuple of XTrain, and YTrain data to train the KNN with.

    Returns
    -------

    knn : KNeighborsClassifier
        Trained K nearest neighbors classifier.
    """

    Xtrain, Ytrain = traindata

    # Use SMotetomek sampling to balances the classes.
    sample = SMOTETomek(random_state=49, sampling_strategy='minority')
    Xtrain_sample, Ytrain_sample = sample.fit_sample(Xtrain, Ytrain)

    # Train KNN
    # maximum on 200 neighbors with 0.7326 b.accuracy for hdbscan
    # n_jobs=-1 to utilize all cores
    knn = KNeighborsClassifier(n_neighbors=200, n_jobs=-1)
    knn.fit(Xtrain_sample, Ytrain_sample.values.ravel())

    return knn


def trainSVM(traindata: Tuple[np.ndarray, np.ndarray]) -> svm.SVC:
    """Function returns a trained svm.SVC instance

    Parameters
    ----------
    traindata : Tuple[np.ndarray, np.ndarray]
        Tuple of XTrain, and YTrain data to train the KNN with.


    Returns
    -------
    svm_model : svm.SVC
        Trained SVM classifier.
    """

    Xtrain, Ytrain = traindata

    sample = SMOTETomek(random_state=49, sampling_strategy='minority')
    Xtrain_sample, Ytrain_sample = sample.fit_sample(Xtrain, Ytrain)

    # hyperparameters found using grid serach and 3 fold validation
    svm_model = svm.SVC(class_weight='balanced', C=1, gamma=0.001, kernel='linear')
    svm_model.fit(Xtrain_sample, Ytrain_sample.values.ravel())

    return svm_model


class Triton(nn.Module):
    """Triton Model. Takes two models; a CNN and a numerical model and welds them
       together to create one whole model.
    """

    def __init__(self, imageModel: Callable, dataModel: Callable, imageModelargs: Dict[str, Union[int, bool]], dataModelargs, num_classes: int):
        """Init Triton Model
        
        Parameters
        ----------
        imageModel : Callable
            CNN model which has been modified to remove its classification layer.
        dataModel : Callable
            Numerical classification model. Unmodified.
        imageModelargs : Dict[str, Union[int, bool]]
            Arguments to feed to the CNN model.
        dataModelargs : TYPE
            Arguments to
        num_classes : int
            Description
        """
        super(Triton, self).__init__()

        self.imageModel = imageModel(imageModelargs["features"], classify=imageModelargs["classify"])
        self.dataModel = dataModel(dataModelargs)

        # plus one for the feature from the motion data model
        self.fc1 = nn.Linear(imageModelargs["features"] + 1, 4)
        self.fc2 = nn.Linear(4, num_classes)

        self.out = nn.LogSoftmax(dim=1)

    def forward(self, image, data, device):

        x1 = self.imageModel(image)
        x2 = self.dataModel.predict(data)

        x2 = torch.as_tensor([x2], dtype=torch.float32).to(device)
        x2 = torch.transpose(x2, 0, 1)

        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)

        x = torch.cat((x1, x2), dim=1)

        x = F.relu(self.fc1(x))
        x = self.out(self.fc2(x))

        return x
