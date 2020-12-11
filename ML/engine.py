import optuna
from sklearn import metrics
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import torch
from tqdm import tqdm

__all__ = ["train_Triton", "train_CNN", "evaluate_Triton", "evaluate_CNN"]


def train_Triton(trial, model, criterion, optimizer, train_loader, test_loader, device, epochs, writer, start_epoch=0) -> float:
    """Train function for Triton model using given criterion, and optimizer.
       Writes out to tensorboard and supports optuna hyperparameter tuning

    Parameters
    ----------
    trial : ?
        trial object from optuna library. If none then were not optimising hyperparameters.
    model : ?
        The model to train

    criterion : nn.Module._Loss
        The criterion to use as a loss function to asses the performance of the
        model being trained.

    optimizer : torch.optim.Optimizer
        The learning rate optimizer.

    train_loader : torch.utils.data.dataloader.DataLoader
        The Pytorch data loader that loads the training data.

    test_loader : torch.utils.data.dataloader.DataLoader
        The Pytorch data loader that loads the evaluation data.

    device : torch.device
        Which device the model is being trained on.

    epochs : int
        The number of epochs to train the model for.

    writer : torch.utils.tensorboard.writer.SummaryWriter
        Tensorboard instance to write to.

    start_epoch : int, optional
        Epoch to start at

    Returns
    -------
    bacc : float
        The balanced accuracy of the model on the evaluation data.

    Raises
    ------
    optuna.exceptions.TrialPruned
    """

    model.train()
    losses = []
    batches = len(train_loader)
    val_batches = len(test_loader)

    for epoch in range(start_epoch, epochs):
        total_loss = 0
        progress = tqdm(enumerate(train_loader), desc="Loss: ", total=batches)
        model.train()
        for i, data in progress:

            inputs = data[0].to(device)
            labels = data[1].to(device)

            numericalData = data[2]

            optimizer.zero_grad()
            logps = model.forward(inputs, numericalData, device)

            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            current_loss = loss.item()
            total_loss += current_loss
            progress.set_description(f"Loss: {total_loss / (i + 1):.4f}")
            writer.add_scalar("Loss", total_loss / (i + 1), epoch * len(train_loader) + i)

        val_losses, bacc = evaluate_Triton(model, test_loader, criterion, device, epoch, writer)

        print(f"Epoch {epoch+1}/{epochs}, training loss: {total_loss/batches}, validation loss: {val_losses/val_batches}")

        losses.append(total_loss / batches)  # for plotting learning curve
        writer.add_scalar("Loss/train", total_loss / batches, epoch)
        writer.add_scalar("Loss/Test", val_losses / val_batches, epoch)
        writer.flush()
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                    }, "checkpoint_state.pth")
        if trial is not None:
            trial.report(bacc, epoch)

        # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
    return bacc


def train_CNN(trial, model, criterion, optimizer: torch.optim.Optimizer,
              train_loader: torch.utils.data.dataloader.DataLoader,
              test_loader: torch.utils.data.dataloader.DataLoader,
              device: torch.device, epochs: int,
              writer: torch.utils.tensorboard.writer.SummaryWriter, start_epoch=0) -> float:
    """Train function for CNN model using given criterion, and optimizer.
       Writes out to tensorboard and supports optuna hyperparameter tuning

    Parameters
    ----------
    trial : ?
        trial object from optuna library. If none then were not optimising hyperparameters.
    model : ?
        The model to train

    criterion : nn.Module._Loss
        The criterion to use as a loss function to asses the performance of the
        model being trained.

    optimizer : torch.optim.Optimizer
        The learning rate optimizer.

    train_loader : torch.utils.data.dataloader.DataLoader
        The Pytorch data loader that loads the training data.

    test_loader : torch.utils.data.dataloader.DataLoader
        The Pytorch data loader that loads the evaluation data.

    device : torch.device
        Which device the model is being trained on.

    epochs : int
        The number of epochs to train the model for.

    writer : torch.utils.tensorboard.writer.SummaryWriter
        Tensorboard instance to write to.

    start_epoch : int, optional
        Epoch to start at

    Returns
    -------
    bacc : float
        The balanced accuracy of the model on the evaluation data.

    Raises
    ------
    optuna.exceptions.TrialPruned
    """

    model.train()
    losses = []
    batches = len(train_loader)
    val_batches = len(test_loader)

    for epoch in range(start_epoch, epochs):
        total_loss = 0
        progress = tqdm(enumerate(train_loader), desc="Loss: ", total=batches)
        model.train()
        for i, data in progress:

            inputs = data[0].to(device)
            labels = data[1].to(device)

            optimizer.zero_grad()
            logps = model.forward(inputs)

            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            current_loss = loss.item()
            total_loss += current_loss
            progress.set_description(f"Loss: {total_loss / (i + 1):.4f}")
            writer.add_scalar("Loss", total_loss / (i + 1), epoch * len(train_loader) + i)

        val_losses, bacc = evaluate_CNN(model, test_loader, criterion, device, epoch, writer)

        print(f"Epoch {epoch+1}/{epochs}, training loss: {total_loss/batches}, validation loss: {val_losses/val_batches}")

        losses.append(total_loss / batches)  # for plotting learning curve
        writer.add_scalar("Loss/train", total_loss / batches, epoch)
        writer.add_scalar("Loss/Test", val_losses / val_batches, epoch)
        writer.flush()
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                    }, "checkpoint_state.pth")
        if trial is not None:
            trial.report(bacc, epoch)

        # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
    return bacc


def evaluate_Triton(model, test_loader, criterion, device, epoch, writer=None, infer=False):
    """Evaluate how well the model has learned.

    Parameters
    ----------
    model : ?
        The model to train
    test_loader : torch.utils.data.dataloader.DataLoader
        The Pytorch data loader that loads the evaluation data.
    criterion : nn.Module._Loss
        The criterion to use as a loss function to asses the performance of the
        model being trained.
    device : torch.device
        Which device the model is being trained on.
    epoch : TYPE
        Current epoch
    writer : torch.utils.tensorboard.writer.SummaryWriter
        Tensorboard instance to write to.
    infer : bool, optional
        If true then calculate performance on validation or test sets and return cm and bacc.

    Returns
    -------
    TYPE
        Description
    """

    val_losses = 0
    trues = []
    preds = []
    batches = len(test_loader)

    model.eval()
    with torch.no_grad():

        progressEval = tqdm(enumerate(test_loader), desc="Eval:", total=batches)

        for i, data in progressEval:
            X = data[0].to(device)
            y = data[1].to(device)
            z = data[2]

            outputs = model.forward(X, z, device)
            val_losses += criterion(outputs, y)
            predicted_classes = torch.max(outputs, 1)[1]  # get class from network's prediction
            trues.extend(y.cpu().detach().numpy())
            preds.extend(predicted_classes.cpu().detach().numpy())
        if infer:
            print(metrics.classification_report(trues, preds))
            cm = metrics.confusion_matrix(trues, preds)
            print(cm)

        results = metrics.precision_recall_fscore_support(trues, preds)
        acc = metrics.accuracy_score(trues, preds)
        bacc = balanced_accuracy_score(trues, preds)

        if writer:
            writer.add_scalar("Accuracy/accuracy", acc, epoch)
            writer.add_scalar("Accuracy/balanced_accuracy", bacc, epoch)

            writer.add_scalar("Pecision/Dolphin", results[0][0], epoch)
            writer.add_scalar("Recall/Dolphin", results[1][0], epoch)
            writer.add_scalar("F1/Dolphin", results[2][0], epoch)

            writer.add_scalar("Pecision/Not_dolphin", results[0][1], epoch)
            writer.add_scalar("Recall/Not_dolphin", results[1][1], epoch)
            writer.add_scalar("F1/Not_dolphin", results[2][1], epoch)
            writer.flush()

    if infer:
        return val_losses, bacc, cm
    else:
        return val_losses, bacc


def evaluate_CNN(model, test_loader, criterion, device, epoch, writer=None, infer=False):
    """Evaluate how well the model has learned.

    Parameters
    ----------
    model : ?
        The model to train
    test_loader : torch.utils.data.dataloader.DataLoader
        The Pytorch data loader that loads the evaluation data.
    criterion : nn.Module._Loss
        The criterion to use as a loss function to asses the performance of the
        model being trained.
    device : torch.device
        Which device the model is being trained on.
    epoch : TYPE
        Current epoch
    writer : torch.utils.tensorboard.writer.SummaryWriter
        Tensorboard instance to write to.
    infer : bool, optional
        If true then calculate performance on validation or test sets and return cm and bacc.

    Returns
    -------
    TYPE
        Description
    """

    val_losses = 0
    trues = []
    preds = []
    batches = len(test_loader)

    model.eval()
    with torch.no_grad():

        progressEval = tqdm(enumerate(test_loader), desc="Eval:", total=batches)

        for i, data in progressEval:
            X = data[0].to(device)
            y = data[1].to(device)

            outputs = model.forward(X)
            val_losses += criterion(outputs, y)
            predicted_classes = torch.max(outputs, 1)[1]  # get class from network's prediction
            trues.extend(y.cpu().detach().numpy())
            preds.extend(predicted_classes.cpu().detach().numpy())
        if infer:
            print(metrics.classification_report(trues, preds))
            cm = metrics.confusion_matrix(trues, preds)

        results = metrics.precision_recall_fscore_support(trues, preds)
        acc = metrics.accuracy_score(trues, preds)
        bacc = balanced_accuracy_score(trues, preds)

        if writer:
            writer.add_scalar("Accuracy/accuracy", acc, epoch)
            writer.add_scalar("Accuracy/balanced_accuracy", bacc, epoch)

            writer.add_scalar("Pecision/Dolphin", results[0][0], epoch)
            writer.add_scalar("Recall/Dolphin", results[1][0], epoch)
            writer.add_scalar("F1/Dolphin", results[2][0], epoch)

            writer.add_scalar("Pecision/Not_dolphin", results[0][1], epoch)
            writer.add_scalar("Recall/Not_dolphin", results[1][1], epoch)
            writer.add_scalar("F1/Not_dolphin", results[2][1], epoch)
            writer.flush()

    if infer:
        return val_losses, bacc, cm
    else:
        return val_losses, bacc
