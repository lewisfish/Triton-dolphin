import argparse

import numpy as np
import optuna
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch import optim

from customdata import DolphinDatasetClass, getNumericalData
from engine import train_CNN, evaluate_CNN, train_Triton, evaluate_Triton
from utils import plot_confusion_matrix, select_model, get_dataset, imageTransfroms


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

    cnn_arch = trial.suggest_categorical("cnn_arch", ["resnet", "vgg", "densenet"])

    modeltype = "Triton"
    features = trial.suggest_int("features", 4, 16)

    model = select_model(modeltype, cnn_arch, features, device)

    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD", "AdamW", "Adadelta"])
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    train_loader, test_loader = get_dataset(batch_size)

    weight = trial.suggest_uniform("weight", 1., 5.)
    weights = torch.FloatTensor([weight, 1.]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    experiment = f"batchsize={batch_size}, weight={weight:.3f}, lr={lr:.3E}, {optimizer_name}, features={features}, arch={cnn_arch}"

    writer = SummaryWriter(f"dolphin/triton-resnet-knn/{experiment}")
    num_epochs = 30

    acc = train_Triton(trial, model, criterion, optimizer, train_loader, test_loader, device, num_epochs, writer, 0)

    return acc


def tune(studyname: str, dbpath: str, trials: int):
    """Function that inits optuna library to tune hyperparameters."""

    dbpath = "sqlite:///" + dbpath + ".db"
    study = optuna.create_study(direction="maximize", study_name=studyname, storage=dbpath)
    study.optimize(objective, n_trials=trials)

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


def train(args: argparse.Namespace):
    """Train a model

    Parameters
    ----------
    args : args.Namespace
        Collection of user defined cmdline variables.
    """

    # seeds used in finding best model
    seed = args.seed
    torch.manual_seed(seed)

    gpu = 1
    device = torch.device(gpu if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)

    classes = ["dolphin", "not dolphin"]
    num_classes = len(classes)

    cnn_arch = args.arch
    modeltype = args.type
    features = args.features

    model = select_model(modeltype, cnn_arch, features, device)

    lr = args.lr
    optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)
    weight = args.weight
    batch_size = args.batchsize

    train_loader, test_loader = get_dataset(batch_size)

    weights = torch.FloatTensor([weight, 1.]).to(device)
    criterion = nn.CrossEntropyLoss()

    experiment = f"{cnn_arch}, batchsize={batch_size}, weights[{weight:.3f},{1.}], lr={lr:.3E}, {optimizer}, features={features}, seed={seed}"

    writer = SummaryWriter(f"runs/{experiment}")#average-triton-results/{experiment}
    num_epochs = 30

    if modeltype == "Triton":
        acc = train_Triton(None, model, criterion, optimizer, train_loader, test_loader, device, num_epochs, writer, 0)
    elif modeltype == "CNN":
        acc = train_CNN(None, model, criterion, optimizer, train_loader, test_loader, device, num_epochs, writer, 0)

    torch.save(model, f"{args.modelpath}")


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

    model = select_model(modeltype, cnn_arch, features, device)

    # load pretrained model for infer
    model = torch.load(modelpath)
    model.eval()
    val_losses = 0

    # infer
    if modeltype == "CNN":
        _, bacc, cm = evaluate_CNN(model, test_loader, criterion, device, 0, writer=None, infer=True)
        cmname = "CNN-cm"
    elif modeltype == "Triton":
        _, bacc, cm = evaluate_Triton(model, test_loader, criterion, device, 0, writer=None, infer=True)
        cmname = "Triton-cm"
    else:
        print("Model Type not implemented")
        raise NotImplementedError

    print(bacc)
    plot_confusion_matrix(cm, ["Dolphin", "Not Dolphin"], "triton-cm", title='cm')


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

    parser.add_argument("-mp", "--modelpath", type=str, default="model.pth", help="Path to saved model.")
    parser.add_argument("-a", "--arch", type=str, choices=["resnet", "densenet", "vgg"], help="Choice of CNN model for either CNN or Triton model.")
    parser.add_argument("--type", type=str, choices=["CNN", "Triton"], help="Choice to run CNN or full Triton model.")

    parser.add_argument("-lr", type=float, default=1e-3, help="Learning rate for optimiser")
    parser.add_argument("--optim", type=str, choices=["Adam", "AdamW", "Adadelta", "SGD"], help="Choice of optimiser.")
    parser.add_argument("-w", "--weight", type=float, default=1.0, help="Weight to bias loss function. Only biases the Dolphin class.")
    parser.add_argument("-bs", "--batchsize", type=int, default=32, help="Batchsize for training, evaluation etc.")
    parser.add_argument("-f", "--features", type=int, default=6, help="Number of features to use from CNN in Triton model.")

    parser.add_argument("-test", action="store_true", help="If supplied then run on test data instead of validation data.")
    parser.add_argument("-s", "--seed", type=int, default=1, help="Set seed for the current run. Default is 1.")

    args = parser.parse_args()

    if args.evaluate:
        infer(modelpath=args.modelpath, cnn_arch=args.arch,
              modeltype=args.type, weight=args.weight,
              batch_size=args.batchsize, test=args.test,
              features=args.features)
    elif args.continue_train or args.train:
        train(args)
    elif args.optimise:
        tune(studyname="best-triton-resnet-knn", dbpath="triton-resnet-knn", trials=50)
    else:
        print("Choose a mode!")
    # test_video()
