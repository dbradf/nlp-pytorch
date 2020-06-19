from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.nn import Module, Embedding, Linear, RNNCell, CrossEntropyLoss
from torch.nn.functional import softmax, relu, dropout
from torch.optim import Adam

from nlp_pytorch.data.base_dataset import SplitDataset
from nlp_pytorch.data.vocab import Vocabulary
from nlp_pytorch.train import make_train_state, train


class SurnameVectorizer(object):
    def __init__(
        self, surname_vocab: Vocabulary, national_vocab: Vocabulary, max_length: int
    ) -> None:
        self.surname_vocab = surname_vocab
        self.nationality_vocab = national_vocab
        self.max_length = max_length + 2

    def vectorize(self, surname: str):
        indices = [self.surname_vocab.begin_seq_index]
        indices.extend(self.surname_vocab.lookup_token(token) for token in surname)
        indices.append(self.surname_vocab.end_seq_index)

        out_vector = np.zeros(self.max_length, dtype=np.int64)
        out_vector[: len(indices)] = indices
        out_vector[len(indices) :] = self.surname_vocab.mask_index

        return out_vector, len(indices)

    @classmethod
    def from_dataframe(cls, surname_df: pd.DataFrame) -> SurnameVectorizer:
        surname_vocab = Vocabulary(unk_token="@")
        nationality_vocab = Vocabulary(add_unk=False)
        max_length = 0

        for index, row in surname_df.iterrows():
            max_length = max(len(row.surname), max_length)
            for letter in row.surname:
                surname_vocab.add_token(letter)
            nationality_vocab.add_token(row.nationality)

        return cls(surname_vocab, nationality_vocab, max_length)


class SurnameDataset(SplitDataset):
    def __init__(self, dataframe: pd.DataFrame, vectorizer) -> None:
        super().__init__(dataframe, vectorizer)

        class_counts = dataframe.nationality.value_counts().to_dict()

        def sort_key(item):
            return vectorizer.nationality_vocab.lookup_token(item[0])

        sorted_counts = sorted(class_counts.items(), key=sort_key)
        frequencies = [count for _, count in sorted_counts]
        self.class_weights = 1.0 / torch.tensor(frequencies, dtype=torch.float32)

    @classmethod
    def load_dataset_and_make_vectorizer(
        cls, surname_csv, create_vectorizer=SurnameVectorizer.from_dataframe
    ) -> SurnameDataset:
        surname_df = pd.read_csv(surname_csv)
        return cls(surname_df, create_vectorizer(surname_df))

    def __getitem__(self, index: int):
        row = self._target_df.iloc[index]
        surname_vector, vec_legnth = self.vectorizer.vectorize(row.surname)
        nationality_index = self.vectorizer.nationality_vocab.lookup_token(row.nationality)

        return {
            "x_data": surname_vector,
            "y_target": nationality_index,
            "x_length": vec_legnth,
        }

    def __len__(self) -> int:
        return self._target_size


class ElmmanRNN(Module):
    def __init__(self, input_size, rnn_hidden_size, batch_first=False) -> None:
        super().__init__()
        self.rnn_cell = RNNCell(input_size, rnn_hidden_size)

        self.batch_first = batch_first
        self.hidden_size = rnn_hidden_size

    def _initialize_hidden(self, batch_size):
        return torch.zeros((batch_size, self.hidden_size))

    def forward(self, x_in, initial_hidden=None):
        if self.batch_first:
            batch_size, seq_size, feat_size = x_in.size()
            x_in = x_in.permute(1, 0, 2)
        else:
            seq_size, batch_size, feat_size = x_in.size()

        hiddens = []

        if initial_hidden is None:
            initial_hidden = self._initialize_hidden(batch_size)
            initial_hidden = initial_hidden.to(x_in.device)

        hidden_t = initial_hidden

        for t in range(seq_size):
            hidden_t = self.rnn_cell(x_in[t], hidden_t)
            hiddens.append(hidden_t)

        hiddens = torch.stack(hiddens)

        if self.batch_first:
            hiddens = hiddens.permute(1, 0, 2)

        return hiddens


def column_gather(y_out, x_length):
    x_lengths = x_length.long().detach().cpu().numpy() - 1

    return torch.stack(
        [y_out[batch_index, column_index] for batch_index, column_index in enumerate(x_lengths)]
    )


class SurnameClassifier(Module):
    def __init__(
        self,
        embedding_size,
        num_embeddings,
        num_classes,
        rnn_hidden_size,
        batch_first=True,
        padding_idx=0,
    ) -> None:
        super().__init__()

        self.emb = Embedding(
            num_embeddings=num_embeddings, embedding_dim=embedding_size, padding_idx=padding_idx
        )
        self.rnn = ElmmanRNN(
            input_size=embedding_size, rnn_hidden_size=rnn_hidden_size, batch_first=batch_first
        )
        self.fc1 = Linear(in_features=rnn_hidden_size, out_features=rnn_hidden_size)
        self.fc2 = Linear(in_features=rnn_hidden_size, out_features=num_classes)

    def forward(self, x_in, x_lengths=None, apply_activator=False):
        x_embedded = self.emb(x_in)
        y_out = self.rnn(x_embedded)

        if x_lengths is not None:
            y_out = column_gather(y_out, x_lengths)
        else:
            y_out = y_out[:, -1, :]

        y_out = dropout(y_out, 0.5)
        y_out = relu(self.fc1(y_out))
        y_out = dropout(y_out, 0.5)
        y_out = self.fc2(y_out)

        if apply_activator:
            y_out = softmax(y_out, dim=1)

        return y_out


def compute_accuracy(y_pred, y_target):
    _, y_pred_indices = y_pred.max(dim=1)
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100


def main(batch_size: int = 128, num_epochs: int = 100):
    args = {
        "surname_csv": "data/surnames_with_splits.csv",
        "save_dir": "model_storage/yelp/",
        "model_state_file": "model.pth",
        "vectorizer_file": "vectorizer.json",
        "char_embedding_size": 100,
        "rnn_hidden_size": 64,
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
        embedding_size=args["char_embedding_size"],
        num_embeddings=len(vectorizer.surname_vocab),
        num_classes=len(vectorizer.nationality_vocab),
        rnn_hidden_size=args["rnn_hidden_size"],
    )
    classifier = classifier.to(args["device"])

    loss_func = CrossEntropyLoss()
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
