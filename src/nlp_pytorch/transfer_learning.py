from __future__ import annotations

from collections import Counter
import re
import string

import numpy as np
import pandas as pd
import torch
from torch.nn.init import xavier_uniform_
from torch.nn import Module, Embedding, Sequential, ELU, Conv1d, Linear, CrossEntropyLoss
from torch.nn.functional import avg_pool1d, dropout, relu, softmax
from torch.optim import Adam

from nlp_pytorch.data.base_dataset import SplitDataset
from nlp_pytorch.data.vocab import Vocabulary
from nlp_pytorch.train import train, make_train_state


def preprocess_text(text):
    text = " ".join(word.lower() for word in text.split(" "))
    text = re.sub(r"([.,!?])", r" \1 ", text)
    text = re.sub(r"[^a-zA-Z.,!?]+", r" ", text)
    return text


class NewsVectorizer(object):
    def __init__(self, title_vocab, category_vocab, max_title):
        self.title_vocab = title_vocab
        self.category_vocab = category_vocab
        self.max_vector_len = max_title + 2

    def vectorize(self, title, vector_length: int = -1) -> np.array:
        indices = [self.title_vocab.begin_seq_index]
        indices.extend(self.title_vocab.lookup_token(token) for token in title.split(" "))
        indices.append(self.title_vocab.end_seq_index)

        if vector_length < 0:
            vector_length = len(indices)

        out_vector = np.zeros(self.max_vector_len, dtype=np.int64)
        out_vector[: len(indices)] = indices
        out_vector[len(indices) :] = self.title_vocab.mask_index

        return out_vector

    @classmethod
    def from_dataframe(cls, news_df, cutoff=25):
        category_vocab = Vocabulary()
        for category in sorted(set(news_df.category)):
            category_vocab.add_token(category)

        max_title = 0
        word_counts = Counter()
        for title in news_df.title:
            title_tokens = title.split(" ")
            max_title = max(max_title, len(title_tokens))
            for token in title_tokens:
                if token not in string.punctuation:
                    word_counts[token] += 1

        title_vocab = Vocabulary()
        for word, word_count in word_counts.items():
            if word_count >= cutoff:
                title_vocab.add_token(word)

        return cls(title_vocab, category_vocab, max_title)


class NewsDataset(SplitDataset):
    def __init__(self, news_df: pd.DataFrame, vectorizer) -> None:
        super().__init__(news_df, vectorizer)

    @classmethod
    def load_dataset_and_make_vectorizer(cls, csv_file: str) -> NewsDataset:
        news_df = pd.read_csv(csv_file)
        return cls(news_df, NewsVectorizer.from_dataframe(news_df))

    def __getitem__(self, index: int):
        row = self._target_df.iloc[index]

        title_vector = self.vectorizer.vectorize(row.title)
        category_index = self.vectorizer.category_vocab.lookup_token(row.category)

        return {
            "x_data": title_vector,
            "y_target": category_index,
        }


def load_glove_from_file(glove_filepath):
    word_to_index = {}
    embeddings = []
    with open(glove_filepath, "r") as fp:
        for index, line in enumerate(fp):
            line = line.split(" ")
            word_to_index[line[0]] = index
            embedding_i = np.array([float(val) for val in line[1:]])
            embeddings.append(embedding_i)

    return word_to_index, np.stack(embeddings)


def make_embedding_matrix(glove_filepath, words):
    word_to_idx, glove_embeddings = load_glove_from_file(glove_filepath)
    embedding_size = glove_embeddings.shape[1]
    final_embeddings = np.zeros((len(words), embedding_size))

    for i, word in enumerate(words):
        if word in word_to_idx:
            final_embeddings[i, :] = glove_embeddings[word_to_idx[word]]
        else:
            embedding_i = torch.ones(1, embedding_size)
            xavier_uniform_(embedding_i)
            final_embeddings[i, :] = embedding_i

    return final_embeddings


class NewsClassifier(Module):
    def __init__(
        self,
        embedding_size,
        num_embeddings,
        num_channels,
        hidden_dim,
        num_classes,
        dropout_p,
        pretrained_embeddings=None,
        padding_idx=0,
    ):
        super().__init__()

        if pretrained_embeddings is None:
            self.emb = Embedding(
                embedding_dim=embedding_size, num_embeddings=num_embeddings, padding_idx=padding_idx
            )
        else:
            self.emb = Embedding(
                embedding_dim=embedding_size,
                num_embeddings=num_embeddings,
                padding_idx=padding_idx,
                _weight=pretrained_embeddings,
            )

        self.convnet = Sequential(
            Conv1d(in_channels=embedding_size, out_channels=num_channels, kernel_size=3),
            ELU(),
            Conv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=2),
            ELU(),
            Conv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=2),
            ELU(),
            Conv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=3),
            ELU(),
        )

        self._dropout_p = dropout_p
        self.fc1 = Linear(num_channels, hidden_dim)
        self.fc2 = Linear(hidden_dim, num_classes)

    def forward(self, x_in, apply_activator: bool = False):
        x_embedded = self.emb(x_in).permute(0, 2, 1)
        features = self.convnet(x_embedded)

        remaining_size = features.size(dim=2)
        features = avg_pool1d(features, remaining_size).squeeze(dim=2)
        features = dropout(features, p=self._dropout_p)

        intermediate_vector = relu(dropout(self.fc1(features), p=self._dropout_p))
        prediction_vector = self.fc2(intermediate_vector)

        if apply_activator:
            prediction_vector = softmax(prediction_vector, dim=1)

        return prediction_vector


def predict_category(title, classifer, vectorizer, max_length):
    title = preprocess_text(title)
    vectorized_title = torch.tensor(vectorizer.vectorize(title, vector_length=max_length))
    result = classifer(vectorized_title.unsqueeze(0), apply_activator=True)
    probability_values, indices = result.max(dim=1)
    predicated_category = vectorizer.category_vocab.lookup_index(indices.item())

    return {"category": predicated_category, "probability": probability_values.item()}


def compute_accuracy(y_pred, y_target):
    _, y_pred_indices = y_pred.max(dim=1)
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100


def main(num_epochs: int = 100, batch_size: int = 128):
    args = {
        "news_csv": "data/news_with_splits.csv",
        "save_dir": "model_storage/yelp/",
        "model_state_file": "model.pth",
        "glove_filepath": "data/glove.6B.100d.txt",
        "vectorizer_file": "vectorizer.json",
        "use_glove": False,
        "embedding_size": 100,
        "hidden_dim": 100,
        "num_channels": 100,
        "learning_rate": 0.001,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "early_stopping_criteria": 5,
        "frequency_cutoff": 25,
        "dropout_p": 0.1,
        "cuda": False,
    }
    train_state = make_train_state()

    if torch.cuda.is_available():
        args["cuda"] = True
    args["device"] = torch.device("cuda:0" if args["cuda"] else "cpu")
    print(args)

    dataset = NewsDataset.load_dataset_and_make_vectorizer(args["news_csv"])
    vectorizer = dataset.vectorizer

    words = vectorizer.title_vocab._token_to_idx.keys()
    embeddings = make_embedding_matrix(glove_filepath=args["glove_filepath"], words=words)

    classifier = NewsClassifier(
        embedding_size=args["embedding_size"],
        num_embeddings=len(vectorizer.title_vocab),
        num_channels=args["num_channels"],
        hidden_dim=args["hidden_dim"],
        num_classes=len(vectorizer.title_vocab),
        dropout_p=args["dropout_p"],
        pretrained_embeddings=torch.from_numpy(embeddings),
    )
    classifier = classifier.to(args["device"])
    dataset.class_weights.to(args["device"])
    classifier.double()

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
