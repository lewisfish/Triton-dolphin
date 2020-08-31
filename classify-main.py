import argparse

import numpy as np
import optuna
import torch
from torchvision import transforms as T
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch import optim

from customdata import DolphinDatasetClass, getNumericalData, windowDataset
from engine import train_CNN, evaluate_CNN, train_Triton, evaluate_Triton
from models import Triton, get_resnet50, trainKNN, trainSVM, get_densenet121, get_vgg13_bn, get_resnet50new


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


def get_dataset(batch_size=64, test=False):

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


def objective(trial):
    """Function that optuna will optimise. Basically a wrapper/loader for the
       whole train/evaluate loop.

    """

    torch.manual_seed(1)
    gpu = 1
    device = torch.device(gpu if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)

    classes = ["dolphin", "not dolphin"]
    num_classes = len(classes)

    model_name = trial.suggest_categorical("model_name", ["resnet", "vgg", "densenet"])

    trainfile = "data/train.csv"
    train = getNumericalData(trainfile)
    X_train, Y_train = train
    num_features = trial.suggest_int("nfeatures", 4, 16)

    imageargs = {"features": num_features, "classify": False}
    dataargs = (X_train, Y_train)

    # get model with  correct CNN model
    if model_name == "vgg":
        model = Triton(get_vgg13_bn, trainSVM, imageargs, dataargs, num_classes)
    elif model_name == "resnet":
        model = Triton(get_resnet50new, trainSVM, imageargs, dataargs, num_classes)
    elif model_name == "densenet":
        model = Triton(get_densenet121, trainSVM, imageargs, dataargs, num_classes)

    model.to(device)

    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD", "AdamW", "Adadelta"])
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    train_loader, test_loader = get_dataset(batch_size)

    weight = trial.suggest_uniform("weight", 1., 14.)
    weights = torch.FloatTensor([weight, 1.]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    experiment = f"{model_name}, batchsize={batch_size}, weights[{weight:.3f},{1.}], lr={lr:.3E}, {optimizer_name}"

    writer = SummaryWriter(f"dolphin/hyper/{experiment}")
    num_epochs = 30

    acc = train_Triton(trial, model, criterion, optimizer, train_loader, test_loader, device, num_epochs, writer, 0)

    return acc


def tune(studyname: str, dbpath: str, trials: int):
    """Function that inits optuna library to tune hyperparameters."""

    dbpath = "sqlite:///" + dbpath + ".db"
    study = optuna.create_study(direction="maximize", study_name=studyname, storage=dbpath)
    study.optimize(objective, n_trials=100)

    pruned_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


def train(args):
    torch.manual_seed(1)
    gpu = 1
    device = torch.device(gpu if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)

    classes = ["dolphin", "not dolphin"]
    num_classes = len(classes)

    model_name = "densenet"

    trainfile = "data/train.csv"
    train = getNumericalData(trainfile)
    X_train, Y_train = train
    imageargs = {"features": 6, "classify": False}
    dataargs = (X_train, Y_train)

    if model_name == "vgg":
        model = Triton(get_vgg13_bn, trainSVM, imageargs, dataargs, num_classes)
    elif model_name == "resnet":
        model = Triton(get_resnet50new, trainSVM, imageargs, dataargs, num_classes)
    elif model_name == "densenet":
        model = Triton(get_densenet121, trainSVM, imageargs, dataargs, num_classes)

    model.to(device)

    lr = 2.8637717478899047e-05
    optimizer = optim.Adam(model.parameters(), lr=lr)
    weight = 1.3501842034030416
    batch_size = 32

    train_loader, test_loader = get_dataset(batch_size)

    weights = torch.FloatTensor([weight, 1.]).to(device)
    criterion = nn.CrossEntropyLoss()

    experiment = f"{model_name}, batchsize={batch_size}, weights[{weight:.3f},{1.}], lr={lr:.3E}, Adam, features={imageargs['features']}, BEST_Triton_SVM_Final"

    writer = SummaryWriter(f"dolphin/final-SVM/{experiment}")
    num_epochs = 30

    # checkpoint = torch.load("checkpoint_state.pth")
    # model.load_state_dict(checkpoint["model_state_dict"])
    # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    # start_epoch = checkpoint["epoch"] + 1

    acc = train_Triton(None, model, criterion, optimizer, train_loader, test_loader, device, num_epochs, writer, 0)
    torch.save(model, "final-model-triton-SVM.pth")


def infer(modelpath: str, cnn_arch: str, modeltype: str, weight: float,
          batch_size: int, test: bool, features: int):
    """Summary

    Parameters
    ----------
    modelpath : str
        Path to the saved model
    cnn_arch : str
        CNN acrhitecture to use in model
    modeltype : str
        Which model to infer on, CNN or Triton.
    weight : float
        Description
    batch_size : int
        Batch size.
    test : bool
        If true then infer on test dataset. If False infer on validation set
    features : int
        Number of features to use in the Triton model.
    """

    # setup seed and device to use for infer
    torch.manual_seed(1)
    gpu = 1  # use only the 2nd GPU
    device = torch.device(gpu if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)

    # get data and setup criterion
    train_loader, test_loader = get_dataset(batch_size, test)
    weights = torch.FloatTensor([weight, 1.]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # setup CNN or Triton
    if modeltype == "CNN":
        if cnn_arch == "vgg":
            model = get_vgg13_bn(2)
        elif cnn_arch == "resnet":
            model = get_resnet50new(2)
        elif cnn_arch == "densenet":
            model = get_densenet121(2)
    elif modeltype == "Triton":
        classes = ["dolphin", "not dolphin"]
        num_classes = len(classes)

        # get preprocessing data
        trainfile = "data/train.csv"
        train = getNumericalData(trainfile)
        X_train, Y_train = train
        imageargs = {"features": features, "classify": False}
        dataargs = (X_train, Y_train)

        # create Triton model woth correct CNN arch
        if cnn_arch == "vgg":
            model = Triton(get_vgg13_bn, trainSVM, imageargs, dataargs, num_classes)
        elif cnn_arch == "resnet":
            model = Triton(get_resnet50new, trainSVM, imageargs, dataargs, num_classes)
        elif cnn_arch == "densenet":
            model = Triton(get_densenet121, trainSVM, imageargs, dataargs, num_classes)

    else:
        print("Model Type not implemented")
        raise NotImplementedError

    # load pretrained model for infer
    model = torch.load(modelpath)
    model.eval()
    val_losses = 0

    # infer
    if modeltype == "CNN":
        _, bacc = evaluate_CNN(model, test_loader, criterion, device, 0, writer=None, infer=True)
    elif modeltype == "Triton":
        _, bacc = evaluate_Triton(model, test_loader, criterion, device, 0, writer=None, infer=True)
    else:
        print("Model Type not implemented")
        raise NotImplementedError
    print(bacc)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch pipeline for train\
                                     and detection of 4 image classes.')

    parser.add_argument("-e", "--epochs", type=int, default=10, help="number of\
                        epochs to train for (default: 10)", metavar="N")
    parser.add_argument("-c", "--continue_train", action="store_true",
                        default=False, help="For continuing training.")

    parser.add_argument("-i", "--evaluate", action="store_true", help="Infere on a bunch of images.")
    parser.add_argument("-o", "--optimise", action="store_true", help="Tune the hyperparameters of the chosen model.")
    parser.add_argument("-t", "--train", action="store_true", help="Train the chosen model.")

    parser.add_argument("--modelpath", type=str, default="model.pth", help="Path to saved model.")
    parser.add_argument("--model", type=str, choices=["resnet", "densenet", "vgg"], help="Choice of CNN model for either CNN or Triton model.")
    parser.add_argument("--type", type=str, choices=["CNN", "Triton"], help="Choice to run CNN or full Triton model.")

    parser.add_argument("-lr", type=float, default=1e-3, help="Learning rate for optimiser")
    parser.add_argument("--optim", type=str, choices=["Adam", "AdamW", "Adadelta", "SGD"], help="Choice of optimiser.")
    parser.add_argument("-w", "--weight", type=float, default=2.0, help="Weight to bias loss function. Only biases the Dolphin class.")
    parser.add_argument("-bs", "--batchsize", type=int, default=32, help="Batchsize for training, evaluation etc.")
    parser.add_argument("-f", "--features", type=int, default=6, help="Number of features to use from CNN in Triton model.")

    parser.add_argument("-test", action="store_true", help="If supplied then run on test data instead of validation data.")

    args = parser.parse_args()

    if args.evaluate:
        infer(modelpath=args.modelpath, model_name=args.model,
              modeltype=args.type, weight=args.weight,
              batch_size=args.batchsize, test=args.test,
              features=args.features)
    elif args.continue_train or args.train:
        train(args)
    elif args.optimise:
        tune("triton-best-cnn", "triton-cnn", 35)
    else:
        print("Choose a mode!")
    # test_video()
