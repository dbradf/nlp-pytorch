from typing import Dict, Any

from tqdm import tqdm
import torch
from torch.nn import Module

from nlp_pytorch.data.base_dataset import TRAIN, VAL, SplitDataset, generate_batches


def make_train_state():
    return {
        "epoch_index": 0,
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "test_loss": -1,
        "test_acc": -1,
    }


class RunningValue(object):
    def __init__(self) -> None:
        self.value = 0.0

    def add(self, value: float, index: int) -> None:
        self.value += (value - self.value) / (index + 1)


def train(
    args: Dict[str, Any],
    train_state: Dict,
    dataset: SplitDataset,
    classifier: Module,
    optimizer,
    loss_func,
    compute_accuracy,
):
    for epoch_index in tqdm(range(args["num_epochs"])):
        train_state["epoch_index"] = epoch_index

        dataset.set_split(TRAIN)
        batch_generator = generate_batches(
            dataset, batch_size=args["batch_size"], device=args["device"]
        )
        running_loss = RunningValue()
        running_acc = RunningValue()
        classifier.train()

        for batch_index, batch_dict in enumerate(batch_generator):
            optimizer.zero_grad()
            y_pred = classifier(x_in=batch_dict["x_data"])

            loss = loss_func(y_pred, batch_dict["y_target"])
            loss_batch = loss.item()
            running_loss.add(loss_batch, batch_index)

            loss.backward()

            optimizer.step()

            acc_batch = compute_accuracy(y_pred, batch_dict["y_target"])
            running_acc.add(acc_batch, batch_index)

        train_state["train_loss"].append(running_loss.value)
        train_state["train_acc"].append(running_acc.value)

        dataset.set_split(VAL)
        batch_generator = generate_batches(
            dataset, batch_size=args["batch_size"], device=args["device"]
        )
        running_loss = RunningValue()
        running_acc = RunningValue()
        classifier.eval()

        for batch_index, batch_dict in enumerate(batch_generator):
            y_pred = classifier(x_in=batch_dict["x_data"])
            loss = loss_func(y_pred, batch_dict["y_target"])
            loss_batch = loss.item()
            running_loss.add(loss_batch, batch_index)

            acc_batch = compute_accuracy(y_pred, batch_dict["y_target"])
            running_acc.add(acc_batch, batch_index)

        train_state["val_loss"].append(running_loss.value)
        train_state["val_acc"].append(running_acc.value)

        print(train_state)
