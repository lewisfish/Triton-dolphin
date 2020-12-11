import itertools

from matplotlib import colorbar, rcParams
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision import transforms as T

from models import Triton, get_resnet50, trainKNN, get_densenet121, get_vgg13_bn
from customdata import DolphinDatasetClass, getNumericalData


__all__ = ["plot_confusion_matrix", "imageTransfroms", "get_dataset", "select_model"]


def plot_confusion_matrix(cm,
                          labels,
                          path,
                          title='Confusion matrix',
                          cmap=None,
                          norm=False):
    """
    given a sklearn confusion matrix (cm), create a matplotlib figure

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    labels:       given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    norm:         If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm     = cm,                  # confusion matrix created by
                                                        # sklearn.metrics.confusion_matrix
                          norm   = True,                # show proportions
                          labels = y_labels_vals,       # list of names of the classes
                          title  = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    https://www.kaggle.com/grfiv4/plot-a-confusion-matrix
    https://github.com/pschrempf/plotml

    """

    # Set font params
    plt.rc("font", family="serif")
    plt.rc('text', usetex=True)
    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    plt.rc('axes', labelsize=8)

    # Calculate accuracy and max value
    accuracy = np.trace(cm) / float(np.sum(cm))
    maximum = 1 if norm else cm.max()

    # Set default colourmap (purple is nice)
    if cmap is None:
        cmap = plt.get_cmap('Purples')

    # Normalise values
    norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create figure
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=.25, bottom=.05, right=.75, top=.83)

    im = plt.imshow(norm_cm, interpolation='nearest', cmap=cmap, aspect='auto')
    plt.title(title, fontweight='bold')
    pos = ax.get_position()

    # Add values to figure
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = 'white' if cm[i, j] > cm[i].sum() / 2 else 'black'
        text = f"{norm_cm[i,j]:0.4f}" if norm else f"{cm[i,j]:0.0f}"
        plt.text(j, i, text, horizontalalignment='center', va='center', color=color, fontsize=8)
        ax.axhline(i-.5, color='black', linewidth=1.5)
        ax.axvline(j-.5, color='black', linewidth=1.5)

    # Add primary axes
    tick_marks = np.arange(len(labels))

    ax.tick_params(
        axis='both',
        which='both',
        labeltop=False,
        labelbottom=False,
        length=0)
    ax.set_ylabel('True')
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(labels)
    ax.set_xticks(tick_marks)
    ax.tick_params(axis='both', which='both', pad=5)
    ax.tick_params(axis='y', which='minor', labelrotation=90)

    # Add secondary axes displaying at top of figure
    ax2 = ax.twiny()
    ax2.tick_params(
        axis='both',
        which='both',
        labelbottom=False,
        length=0)
    ax2.tick_params(axis='both', which='both', pad=5)
    ax2.set_xticks(tick_marks)
    ax2.set_xlim(ax.get_xlim())

    ax.autoscale(False)
    ax2.autoscale(False)

    ax2.set_xlabel('Predicted')
    ax2.set_xticklabels(labels)

    # Add colourbar
    cbax = fig.add_axes([pos.x0+pos.width+.05, pos.y0, 0.08, pos.height])
    cb = colorbar.ColorbarBase(cbax, cmap=cmap, orientation='vertical')
    cb.set_label('Accuracy per label')
    figwidth = 3.503
    height = figwidth / 1.618

    fig.set_size_inches(figwidth, height)

    plt.savefig(f"{path}.png", dpi=300)


def imageTransfroms(train: bool):
    """Set of transforms to use on traing and test data.
       These trasforms include data augmentations

    Parameters
    ----------

    train : bool
        If true then augment the data. If false don't

    Returns
    -------

    Set of composed transforms

    """

    transforms = []
    transforms.append(T.Resize((224, 224)))
    if train:
        transforms.append(T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15))
        transforms.append(T.RandomRotation(180))
        transforms.append(T.RandomHorizontalFlip())

    transforms.append(T.ToTensor())
    transforms.append(T.Normalize([.485, .456, .406],
                                  [.229, .224, .225]))
    return T.Compose(transforms)


def select_model(modeltype: str, cnn_arch: str, features: int, device: torch.device) -> torchvision.models:
    """Setup model on given device.

    Parameters
    ----------
    modeltype : str
        String holds the model type to return
    cnn_arch : str
        String holds the CNN acrhitecture to use in model.
    features : int
        Number of features used in Triton model
    device : torch.device
        Device to run model on.

    Returns
    -------
    model : torchvision.models
        Machine learning model

    Raises
    ------
    NotImplementedError
        Description
    """

    # setup CNN or Triton
    if modeltype == "CNN":
        if cnn_arch == "vgg":
            model = get_vgg13_bn(2)
        elif cnn_arch == "resnet":
            model = get_resnet50(2)
        elif cnn_arch == "densenet":
            model = get_densenet121(2)
    elif modeltype == "Triton":
        classes = ["dolphin", "not dolphin"]
        num_classes = len(classes)

        # get preprocessing data
        trainfile = "data/train.csv"
        train = getNumericalData(trainfile)
        X_train, Y_train = train
        imageargs = {"features": 6, "classify": False}
        dataargs = (X_train, Y_train)

        # create Triton model woth correct CNN arch
        if cnn_arch == "vgg":
            model = Triton(get_vgg13_bn, trainKNN, imageargs, dataargs, num_classes)
        elif cnn_arch == "resnet":
            model = Triton(get_resnet50, trainKNN, imageargs, dataargs, num_classes)
        elif cnn_arch == "densenet":
            model = Triton(get_densenet121, trainKNN, imageargs, dataargs, num_classes)

    else:
        print("Model Type not implemented")
        raise NotImplementedError

    model.to(device)

    return model


def get_dataset(batch_size=64, test=False):
    """Function loads the trains, and either valid or test datasets.

    Parameters
    ----------
    batch_size : int, optional
        Set size of the batch for training
    test : bool, optional
        If True then load the test data, else load validation dataset

    Returns
    -------
    TYPE
        Description
    """

    root = "/data/lm959/imgs/"
    trainfile = "data/train.csv"
    validfile = "data/valid.csv"
    testfile = "data/test.csv"

    dataset = DolphinDatasetClass(root, imageTransfroms(train=True), trainfile)
    if test:
        dataset_test = DolphinDatasetClass(root, imageTransfroms(train=False), testfile)
    else:
        dataset_test = DolphinDatasetClass(root, imageTransfroms(train=False), validfile)

    class_weights = np.loadtxt("data/class_weights.csv", delimiter=",")

    sampler = torch.utils.data.sampler.WeightedRandomSampler(class_weights, len(dataset), replacement=True)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=6,
                                               drop_last=True, sampler=sampler)

    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=4,
                                              drop_last=True)

    return train_loader, test_loader
