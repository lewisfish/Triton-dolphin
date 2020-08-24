import argparse

import numpy as np
import optuna
import torch
from torchvision import transforms as T
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch import optim

from customdata import DolphinDatasetClass, getNumericalData, windowDataset
from engine import train_classify, class_evaluate, infer
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


def get_dataset(batch_size=64):

    root = "/data/lm959/imgs/"
    trainfile = "data/train.csv"
    validfile = "data/valid.csv"

    dataset = DolphinDatasetClass(root, imageTransfroms(train=True), trainfile)
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

    acc = train_classify(trial, model, criterion, optimizer, train_loader, test_loader, device, num_epochs, writer, 0)

    return acc


def tune():
    """Function that inits optuna library to tune hyperparameters."""

    study = optuna.create_study(direction="maximize", study_name="model-trial", storage="sqlite:///example.db")
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

    experiment = f"{model_name}, batchsize={batch_size}, weights[{weight:.3f},{1.}], lr={lr:.3E}, Adam, features={imageargs['features']}, nolossweights"

    writer = SummaryWriter(f"dolphin/best/{experiment}")
    num_epochs = 30

    # checkpoint = torch.load("checkpoint_state.pth")
    # model.load_state_dict(checkpoint["model_state_dict"])
    # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    # start_epoch = checkpoint["epoch"] + 1

    acc = train_classify(None, model, criterion, optimizer, train_loader, test_loader, device, num_epochs, writer, 0)
    torch.save(model, "final-model-triton-svm-nolossweights.pth")
    # model = torch.load("final-model-frankenstein.pth")
    # model.eval()
    # val_losses = 0
    # class_evaluate(model, test_loader, criterion, device, 0, writer=None, infer=True)


def infer(modelname: str, lr=2.864e-5, optimisername="Adam", weight=1.3502, batch_size=32):

    torch.manual_seed(1)
    gpu = 1  # use only the 2nd GPU
    device = torch.device(gpu if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)

    optimizer = getattr(optim, optimiserName)(model.parameters(), lr=lr)

    train_loader, test_loader = get_dataset(batch_size)

    weights = torch.FloatTensor([weight, 1.]).to(device)
    criterion = nn.CrossEntropyLoss()

    model = torch.load(modelname)
    model.eval()
    val_losses = 0
    class_evaluate(model, test_loader, criterion, device, 0, writer=None, infer=True)


def test_video():

    file = "data/clip.mp4"
    root = "/data/lm959/tmp/patches/"
    batch_size = 128

    transforms = []
    transforms.append(T.Resize((224, 224)))
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize([.485, .456, .406],
                                  [.229, .224, .225]))
    trans = T.Compose(transforms)

    # file, transforms, size, stride
    dataset = windowDataset(file, trans, 25, 25)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

    gpu = 1
    device = torch.device(gpu if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)

    classes = ["dolphin", "not dolphin"]
    num_classes = len(classes)

    model = get_densenet121(num_classes, classify=True)
    model.to(device)

    model = torch.load("final-model-densenet.pth")
    model.eval()
    weight = torch.FloatTensor([5.5690176113112395, 1.]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    lr = 0.0005528591422378424
    optimizer = optim.Adam(model.fc.parameters(), lr=lr)

    if args.continue_train:
        checkpoint = torch.load("checkpoint_state_DC4.pth")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
    else:
        start_epoch = 0

    num_epochs = args.epochs

    experiment = f"weights[{weight[0]},{weight[1]}],lr={lr},WeightedRandomSamplerler"

    if not args.evaluate:
        writer = SummaryWriter(f"dolphin/classify/{experiment}")
        trial = None
        bacc = train_classify(trial, model, criterion, optimizer, data_loader, data_loader_test, device, num_epochs, writer)
        torch.save(model, "final-model.pth")
    else:
        model = torch.load("final-model.pth")
        model.eval()
        val_losses = 0
        val_losses = class_evaluate(model, data_loader_test, criterion, device, 0, writer=None, infer=True)


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

    args = parser.parse_args()

    if args.evaluate:
        infer()
    elif args.continue_train:
        train(args)
    elif args.tune:
        tune()
    else:
        print("Choose a mode!")
    # test_video()
