import argparse

import numpy as np
import optuna
import torch
from torchvision import transforms as T
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch import optim
import tqdm

from customdata import DolphinDatasetClass, getNumericalData
from engine import train_classify, class_evaluate
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

    # model = get_resnet50(num_classes)

    model_name = trial.suggest_categorical("model_name", ["resnet", "vgg", "densenet"])

    trainfile = "data/train.csv"
    train = getNumericalData(trainfile)
    X_train, Y_train = train
    imageargs = {"features": 7, "classify": False}
    dataargs = (X_train, Y_train)

    if model_name == "vgg":
        model = Frankenstein(get_vgg13_bn, trainKNN, imageargs, dataargs, num_classes)
    elif model_name == "resnet":
        model = Frankenstein(get_resnet50, trainKNN, imageargs, dataargs, num_classes)
    elif model_name == "densenet":
        model = Frankenstein(get_densenet121, trainKNN, imageargs, dataargs, num_classes)

    model.to(device)

    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD", "AdamW", "Adadelta"])
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    weight = trial.suggest_uniform("weight", 1., 14.)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    train_loader, test_loader = get_dataset(batch_size)

    weights = torch.FloatTensor([weight, 1.]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    experiment = f"{model_name}, batchsize={batch_size}, weights[{weight:.3f},{1.}], lr={lr:.3E}, {optimizer_name}"

    writer = SummaryWriter(f"dolphin/hyper/{experiment}")
    num_epochs = 30
    acc = train_classify(trial, model, criterion, optimizer, train_loader, test_loader, device, num_epochs, writer)

    return acc


def main2():

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


def main(args):
    # import json
    torch.manual_seed(1)

    gpu = 1
    device = torch.device(gpu if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)
    print(device)

    classes = ["dolphin", "not dolphin"]
    num_classes = len(classes)

    root = "/data/lm959/imgs/"
    trainfile = "data/train.csv"
    validfile = "data/valid.csv"

    train = getNumericalData(trainfile)
    X_train, Y_train = train

    batch_size = 64

    dataset = DolphinDatasetClass(root, imageTransfroms(train=True), trainfile)
    dataset_test = DolphinDatasetClass(root, imageTransfroms(train=False), validfile)

    # create class weights.
    # easier to do once then read in saved weights as reading images takes
    # too long
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    try:
        class_weights = np.loadtxt("data/class_weights.csv", delimiter=",")
    except OSError:
        labels = []
        weights = 1000. / torch.tensor([1602, 11688.], dtype=torch.float)
        for i, (img, targ) in enumerate(tqdm.tqdm(data_loader)):
            labels.append(targ.item())
        class_weights = np.array(weights[labels])
        np.savetxt("data/class_weights.csv", class_weights)

    sampler = torch.utils.data.sampler.WeightedRandomSampler(class_weights, len(dataset), replacement=True)
    # now use dataloader with sampler
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4,
                                              drop_last=True, sampler=sampler)

    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=4,
                                                   drop_last=True)

    # model = get_resnet50(num_classes)
    imageargs = {"features": 7, "classify": False}
    dataargs = (X_train, Y_train)
    model = Frankenstein(get_resnet50, trainKNN, imageargs, dataargs, num_classes)
    model.to(device)

    # best values as found by optuna {'lr': 0.0005528591422378424, 'optimizer': 'Adam', 'weight': 5.5690176113112395}
    # penalize not dolphin class
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

    args = parser.parse_args()

    main(args)
    # main2()
