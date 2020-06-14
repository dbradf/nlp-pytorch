from __future__ import annotations

import pandas as pd
import torch
from torch.nn import Module, Linear, CrossEntropyLoss
from torch.nn.functional import relu, softmax, dropout
from torch.optim import Adam

from nlp_pytorch.data.base_dataset import SplitDataset
from nlp_pytorch.data.vocab import Vocabulary
from nlp_pytorch.train import make_train_state, train


class SurnameVectorizer(object):
    def __init__(self, surname_vocab: Vocabulary, national_vocab: Vocabulary) -> None:
        self.surname_vocab = surname_vocab
        self.nationality_vocab = national_vocab

    def vectorize(self, surname: str):
        return self.surname_vocab.one_hot_encoding(surname)

    @classmethod
    def from_dataframe(cls, surname_df: pd.DataFrame) -> SurnameVectorizer:
        surname_vocab = Vocabulary(unk_token="@")
        nationality_vocab = Vocabulary(add_unk=False)

        for index, row in surname_df.iterrows():
            for letter in row.surname:
                surname_vocab.add_token(letter)
            nationality_vocab.add_token(row.nationality)

        return cls(surname_vocab, nationality_vocab)


class SurnameDataset(SplitDataset):
    def __init__(self, dataframe, vectorizer) -> None:
        super().__init__(dataframe, vectorizer)

        class_counts = dataframe.nationality.value_counts().to_dict()

        def sort_key(item):
            return vectorizer.nationality_vocab.lookup_token(item[0])

        sorted_counts = sorted(class_counts.items(), key=sort_key)
        frequencies = [count for _, count in sorted_counts]
        self.class_weights = 1.0 / torch.tensor(frequencies, dtype=torch.float32)

    @classmethod
    def load_dataset_and_make_vectorizer(cls, surname_csv: str) -> SurnameDataset:
        surname_df = pd.read_csv(surname_csv)
        return cls(surname_df, SurnameVectorizer.from_dataframe(surname_df))

    def __getitem__(self, index: int):
        row = self._target_df.iloc[index]
        surname_vector = self.vectorizer.vectorize(row.surname)
        nationality_index = self.vectorizer.nationality_vocab.lookup_token(row.nationality)

        return {"x_data": surname_vector, "y_target": nationality_index}

    def __len__(self) -> int:
        return self._target_size


class SurnameClassifier(Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()

        self.fc1 = Linear(input_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, output_dim)

    def forward(self, x_in, apply_activator: bool = False):
        intermediate_vector = relu(self.fc1(x_in))
        prediction_vector = self.fc2(dropout(intermediate_vector, p=0.5))

        if apply_activator:
            prediction_vector = softmax(prediction_vector, dim=1)

        return prediction_vector


def predict_nationality(name, classifier, vectorizer):
    vectorized_name = vectorizer.vectorize(name)
    vectorized_name = torch.tensor(vectorized_name).view(1, -1)
    result = classifier(vectorized_name, apply_activator=True)

    probability_values, indices = result.max(dim=1)
    index = indices.item()

    predicted_nationality = vectorizer.nationality_vocab.lookup_index(index)
    probability_value = probability_values.item()

    return {"nationality": predicted_nationality, "probability": probability_value}


def compute_accuracy(y_pred, y_target):
    _, y_pred_indices = y_pred.max(dim=1)
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100


def main(batch_size: int = 128, num_epochs: int = 100, hidden_dim: int = 300):
    args = {
        "hidden_dim": hidden_dim,
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

    dataset = SurnameDataset.load_dataset_and_make_vectorizer(args["surname_csv"])
    vectorizer = dataset.vectorizer

    classifier = SurnameClassifier(
        input_dim=len(vectorizer.surname_vocab),
        hidden_dim=args["hidden_dim"],
        output_dim=len(vectorizer.nationality_vocab),
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
