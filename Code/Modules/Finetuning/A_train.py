import torch
import os
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from sklearn.metrics import f1_score


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    loss_fn: torch.nn.Module,
    epochs: int,
    device: torch.device,
    experiment,
) -> Dict[str, List]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").
    experiment: A CometML experiment where to log data

    Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for
    each epoch.
    In the form: {train_loss: [...],
              train_acc: [...],
              test_loss: [...],
              test_acc: [...]}
    For example if training for epochs=2:
             {train_loss: [2.0616, 1.0537],
              train_acc: [0.3945, 0.3945],
              test_loss: [1.2641, 1.5706],
              test_acc: [0.3400, 0.2973]}
    """

    # Create empty results dictionary
    results = {
        "train_loss": [],
        "valid_loss": [],
        "train_f1": [],
        "valid_f1": [],
    }

    # Make sure model on target device
    model.to(device)

    # Retrieve experiment name
    name = experiment.get_name()

    # Create directory where to store model
    saving_directory = os.path.join("Models", "Runs", name)
    os.makedirs(saving_directory, exist_ok=True)

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        # Get loss and F1 score after training step
        train_loss, train_f1 = train_step(
            model=model,
            epoch=epoch,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            experiment=experiment,
        )

        # Get validation loss and F1 score
        valid_loss, valid_f1 = valid_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device,
            experiment=experiment,
            epoch=epoch,
        )

        # Log all metrics to CometML
        experiment.log_metrics({"train_loss": train_loss}, epoch=epoch)
        experiment.log_metrics({"val_loss": valid_loss}, epoch=epoch)
        experiment.log_metrics({"train_f1": train_f1}, epoch=epoch)
        experiment.log_metrics({"val_f1": valid_f1}, epoch=epoch)

        # Save model if it has improved the validation loss
        if epoch > 0 and valid_loss < min(results["valid_loss"]):
            # Save model
            torch.save(model.state_dict(), f"{saving_directory}/best.pt")

        # Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"val_loss: {valid_loss:.4f} | "
            f"train_f1: {train_f1:.4f} | "
            f"val_f1: {valid_f1:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["valid_loss"].append(valid_loss)
        results["train_f1"].append(train_f1)
        results["valid_f1"].append(valid_f1)

    # Return the filled results at the end of the epochs
    return results


def train_step(
    model: torch.nn.Module,
    epoch: int,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    device: torch.device,
    experiment,
) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:

    (0.1112, 0.8743)
    """

    # Log value of learning rate
    experiment.log_metrics({"lr": scheduler.get_last_lr()}, epoch=epoch)

    # Put model in train mode
    model.train()

    # Initialise train loss accumulator
    train_loss = 0

    # Initialize empty list to store ground truth and predicted labels
    ground_truths = []
    predictions = []

    # Loop through data loader data batches
    for X, y in tqdm(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # Forward pass
        logits = model(X)

        # Calculate and accumulate loss
        batch_loss = loss_fn(logits, y)
        train_loss += batch_loss.item()

        # Optimizer zero grad
        optimizer.zero_grad()

        # Loss backward
        batch_loss.backward()

        # Optimizer step
        optimizer.step()

        # Take a scheduler step
        # scheduler.step()

        # Calculate and accumulate accuracy metric across all batches
        prediction = logits.argmax(dim=1)

        # Append ground truths and predictions
        ground_truths.append(y)
        predictions.append(prediction)

    # Take a scheduler step
    scheduler.step()

    # Concatenate to get the overall ground truths and predictions
    ground_truths = torch.cat(ground_truths, dim=0).cpu()
    predictions = torch.cat(predictions, dim=0).cpu()

    # Compute the f1 score for the training step
    train_f1 = f1_score(ground_truths, predictions, average="macro")

    # Return the average loss of the batch
    train_loss = train_loss / len(dataloader)

    return train_loss, train_f1


def valid_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    experiment,
    epoch: int,
) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). For example:

    (0.0223, 0.8985)
    """

    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    valid_loss = 0

    # Initialize empty list to store ground truth and predicted labels
    ground_truths = []
    predictions = []

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for X, y in tqdm(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # Forward pass
            logits = model(X)

            # Calculate and accumulate loss
            batch_loss = loss_fn(logits, y)
            valid_loss += batch_loss.item()

            # Calculate and accumulate accuracy
            prediction = logits.argmax(dim=1)

            # Append ground truths and predictions
            ground_truths.append(y)
            predictions.append(prediction)

    # Concatenate to get the overall ground truths and predictions
    ground_truths = torch.cat(ground_truths, dim=0).cpu()
    predictions = torch.cat(predictions, dim=0).cpu()

    # Compute the f1 score for the training step
    valid_f1 = f1_score(ground_truths, predictions, average="macro")

    # Adjust metrics to get average loss and accuracy per batch
    valid_loss = valid_loss / len(dataloader)

    # Create confusion matrix
    experiment.log_confusion_matrix(
        y_true=ground_truths,
        y_predicted=predictions,
        title=f"Confusion Matrix, F1 = {valid_f1:.2f}",
        row_label="Actual Category",
        column_label="Predicted Category",
        epoch=epoch,
    )

    return valid_loss, valid_f1
