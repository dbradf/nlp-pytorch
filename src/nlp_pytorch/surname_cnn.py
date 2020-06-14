from __future__ import annotations

import numpy as np
import pandas as pd
from torch.nn import Module, Sequential, Conv1d, ELU, Linear, CrossEntropyLoss
from torch.nn.functional import softmax
from torch.optim import Adam

import torch

from nlp_pytorch.data.vocab import Vocabulary
from nlp_pytorch.surname import SurnameDataset, compute_accuracy
from nlp_pytorch.train import make_train_state, train


class SurnameVectorizer(object):
    def __init__(
        self, surname_vocab: Vocabulary, national_vocab: Vocabulary, max_surname_length: int
    ) -> None:
        self.surname_vocab = surname_vocab
        self.nationality_vocab = national_vocab
        self.max_surname_length = max_surname_length

    def vectorize(self, surname: str):
        one_hot_matrix_size = (len(self.surname_vocab), self.max_surname_length)
        one_hot_matrix = np.zeros(one_hot_matrix_size, dtype=np.float32)

        for pos_idx, character in enumerate(surname):
            ch_idx = self.surname_vocab.lookup_token(character)
            one_hot_matrix[ch_idx][pos_idx] = 1

        return one_hot_matrix

    @classmethod
    def from_dataframe(cls, surname_df: pd.DataFrame) -> SurnameVectorizer:
        surname_vocab = Vocabulary(unk_token="@")
        nationality_vocab = Vocabulary(add_unk=False)
        max_surname_length = 0

        for index, row in surname_df.iterrows():
            max_surname_length = max(max_surname_length, len(row.surname))
            for letter in row.surname:
                surname_vocab.add_token(letter)
            nationality_vocab.add_token(row.nationality)

        return cls(surname_vocab, nationality_vocab, max_surname_length)


class SurnameCnnClassifier(Module):
    def __init__(self, initial_num_channels: int, num_classes: int, num_channels: int) -> None:
        super().__init__()

        self.convnet = Sequential(
            Conv1d(in_channels=initial_num_channels, out_channels=num_channels, kernel_size=3),
            ELU(),
            Conv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=2),
            ELU(),
            Conv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=2),
            ELU(),
            Conv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=3),
            ELU(),
        )
        self.fc = Linear(num_channels, num_classes)

    def forward(self, x_in, apply_activator: bool = False):
        features = self.convnet(x_in).squeeze(dim=2)
        prediction_vector = self.fc(features)

        if apply_activator:
            prediction_vector = softmax(prediction_vector, dim=1)

        return prediction_vector


def predict_nationality(name, classifier, vectorizer):
    vectorized_name = vectorizer.vectorize(name)
    vectorized_name = torch.tensor(vectorized_name).unsqueeze(0)
    result = classifier(vectorized_name, apply_activator=True)

    probability_values, indices = result.max(dim=1)
    index = indices.item()

    predicted_nationality = vectorizer.nationality_vocab.lookup_index(index)
    probability_value = probability_values.item()

    return {"nationality": predicted_nationality, "probability": probability_value}


def main(batch_size: int = 128, num_epochs: int = 100, hidden_dim: int = 100):
    args = {
        "hidden_dim": hidden_dim,
        "num_channels": 256,
        "surname_csv": "data/surnames_with_splits.csv",
        "save_dir": "model_storage/yelp/",
        "model_state_file": "model.pth",
        "vectorizer_file": "vectorizer.json",
        "learning_rate": 0.001,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "early_stopping_criteria": 5,
        "frequency_cutoff": 25,
        "cuda": False,
    }
    train_state = make_train_state()

    if torch.cuda.is_available():
        args["cuda"] = True
    args["device"] = torch.device("cuda:0" if args["cuda"] else "cpu")
    print(args)

    dataset = SurnameDataset.load_dataset_and_make_vectorizer(
        args["surname_csv"], SurnameVectorizer.from_dataframe
    )
    vectorizer = dataset.vectorizer

    classifier = SurnameCnnClassifier(
        initial_num_channels=len(vectorizer.surname_vocab),
        num_classes=len(vectorizer.nationality_vocab),
        num_channels=args["num_channels"],
    )
    classifier = classifier.to(args["device"])

    loss_func = CrossEntropyLoss(dataset.class_weights)
    optimizer = Adam(classifier.parameters(), lr=args["learning_rate"])

    train(args, train_state, dataset, classifier, optimizer, loss_func, compute_accuracy)

    return {
        "train_state": train_state,
        "args": args,
        "dataset": dataset,
        "classifier": classifier,
        "loss_func": loss_func,
        "optimizer": optimizer,
    }
