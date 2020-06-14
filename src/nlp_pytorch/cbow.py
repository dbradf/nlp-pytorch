from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.nn import Module, Embedding, Linear, CrossEntropyLoss
from torch.nn.functional import softmax
from torch.optim import Adam

from nlp_pytorch.data.base_dataset import SplitDataset
from nlp_pytorch.data.vocab import Vocabulary
from nlp_pytorch.train import make_train_state, train


class CbowVectorizer(object):
    def __init__(self, cbow_vocab: Vocabulary, max_vector_len: int) -> None:
        self.cbow_vocab = cbow_vocab
        self.max_vector_len = max_vector_len

    @classmethod
    def from_dataframe(cls, cbow_df: pd.DataFrame) -> CbowVectorizer:
        cbow_vocab = Vocabulary()

        max_len = 0
        for index, row in cbow_df.iterrows():
            max_len = max(max_len, len(row.context.split(" ")))
            for token in row.context.split(' '):
                cbow_vocab.add_token(token)
            cbow_vocab.add_token(row.target)
        return cls(cbow_vocab, max_len)

    def vectorize(self, context, vector_length=-1) -> np.array:
        indices = [self.cbow_vocab.lookup_token(token) for token in context.split(" ")]
        if vector_length < 0:
            vector_length = len(indices)

        out_vector = np.zeros(self.max_vector_len, dtype=np.int64)
        out_vector[:len(indices)] = indices
        out_vector[len(indices):] = self.cbow_vocab.mask_index

        return out_vector


class CbowDataset(SplitDataset):
    def __init__(self, dataframe: pd.DataFrame, vectorizer) -> None:
        super().__init__(dataframe, vectorizer)

    @classmethod
    def load_dataset_and_make_vectorizer(
        cls, csv_file: str, create_vectorizer=CbowVectorizer.from_dataframe
    ) -> CbowDataset:
        cbow_df = pd.read_csv(csv_file)
        return cls(cbow_df, create_vectorizer(cbow_df))

    def __getitem__(self, index: int):
        row = self._target_df.iloc[index]

        context_vector = self.vectorizer.vectorize(row.context)
        target_index = self.vectorizer.cbow_vocab.lookup_token(row.target)

        return {
            "x_data": context_vector,
            "y_target": target_index,
        }


def compute_accuracy(y_pred, y_target):
    _, y_pred_indices = y_pred.max(dim=1)
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100


class CbowClassifier(Module):
    def __init__(self, vocabulary_size: int, embedding_size, padding_idx: int = 0) -> None:
        super().__init__()

        self.embedding = Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_size,
            padding_idx=padding_idx
        )

        self.fc1 = Linear(in_features=embedding_size, out_features=vocabulary_size)

    def forward(self, x_in, apply_activator: bool = False):
        x_embedded_sum = self.embedding(x_in).sum(dim=1)
        y_out = self.fc1(x_embedded_sum)

        if apply_activator:
            y_out = softmax(y_out, dim=1)

        return y_out


def main(num_epochs: int = 100, batch_size: int = 128):
    args = {
        "cbow_csv": "data/frankenstein_with_splits.csv",
        "save_dir": "model_storage/yelp/",
        "model_state_file": "model.pth",
        "vectorizer_file": "vectorizer.json",
        "embedding_size": 300,
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

    dataset = CbowDataset.load_dataset_and_make_vectorizer(args["cbow_csv"])
    vectorizer = dataset.vectorizer

    classifier = CbowClassifier(
        vocabulary_size=len(vectorizer.cbow_vocab),
        embedding_size=args["embedding_size"],
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
