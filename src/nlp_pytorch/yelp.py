from collections import defaultdict, Counter
import re
import string

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from nlp_pytorch.data.base_dataset import TRAIN, TEST, VAL, SplitDataset
from nlp_pytorch.data.vocab import Vocabulary
from nlp_pytorch.train import train, make_train_state


def preprocess_test(text):
    text = text.lower()
    text = re.sub(r"([.,!?])", r" \1 ", text)
    text = re.sub(r"[^a-zA-Z.,!?]+", r" ", text)
    return text


def split_data(review_subset, train_proportion, val_proportion, test_proportion):
    by_rating = defaultdict(list)
    for _, row in review_subset.iterrows():
        by_rating[row.rating].append(row.to_dict())

    final_list = []
    for _, item_list in sorted(by_rating.items()):
        np.random.shuffle(item_list)

        n_total = len(item_list)
        n_train = int(train_proportion * n_total)
        n_val = int(val_proportion * n_total)
        n_test = int(test_proportion * n_total)

        for item in item_list[:n_train]:
            item["split"] = TRAIN

        for item in item_list[n_train : n_train + n_val]:
            item["split"] = VAL

        for item in item_list[n_train + n_val : n_train + n_val + n_test]:
            item["split"] = TEST

        final_list.extend(item_list)

    return pd.DataFrame(final_list)


class ReviewVectorizer(object):
    def __init__(self, review_vocab, rating_vocab):
        self.review_vocab = review_vocab
        self.rating_vocab = rating_vocab

    def vectorize(self, review):
        return self.review_vocab.one_hot_encoding(review.split(" "))

    @classmethod
    def from_dataframe(cls, review_df, cutoff=25):
        review_vocab = Vocabulary(add_unk=True)
        rating_vocab = Vocabulary(add_unk=False)

        for rating in sorted(set(review_df.rating)):
            rating_vocab.add_token(rating)

        word_counts = Counter()
        for review in review_df.review:
            for word in review.split(" "):
                if word not in string.punctuation:
                    word_counts[word] += 1

        for word, count in word_counts.items():
            if count > cutoff:
                review_vocab.add_token(word)

        return cls(review_vocab, rating_vocab)

    @classmethod
    def from_serializable(cls, contents):
        review_vocab = Vocabulary.from_serializable(contents["review_vocab"])
        rating_vocab = Vocabulary.from_serializable(contents["rating_vocab"])
        return cls(review_vocab=review_vocab, rating_vocab=rating_vocab)

    def to_serializable(self):
        return {
            "review_vocab": self.review_vocab.to_serializeable(),
            "rating_vocab": self.rating_vocab.to_serializeable(),
        }


class ReviewDataset(SplitDataset):
    def __init__(self, review_df, vectorizer):
        super().__init__(review_df, vectorizer)

    @classmethod
    def load_dataset_and_make_vectorizer(cls, review_csv):
        review_df = pd.read_json(review_csv, lines=True)
        review_df["rating"] = review_df["stars"] <= 3
        review_df["review"] = review_df["text"]
        review_df = split_data(review_df, 0.70, 0.15, 0.15)
        return cls(review_df, ReviewVectorizer.from_dataframe(review_df))

    def __getitem__(self, index):
        row = self._target_df.iloc[index]
        review_vector = self.vectorizer.vectorize(row.review)
        rating_index = self.vectorizer.rating_vocab.lookup_token(row.rating)
        return {"x_data": review_vector, "y_target": float(rating_index)}

    def __len__(self) -> int:
        return self._target_size


class ReviewClassifier(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features=num_features, out_features=1)

    def forward(self, x_in, apply_activator=False):
        y_out = self.fc1(x_in).squeeze()
        if apply_activator:
            y_out = F.sigmoid(y_out)
        return y_out


def compute_accuracy(y_pred, y_target):
    y_target = y_target.cpu()
    y_pred_indices = (torch.sigmoid(y_pred) > 0.5).cpu().long()
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100


def main(batch_size: int = 128, num_epochs: int = 100):
    args = {
        "review_csv": "data/yelp_reviews_lite.json",
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

    dataset = ReviewDataset.load_dataset_and_make_vectorizer(args["review_csv"])
    vectorizer = dataset.vectorizer

    classifier = ReviewClassifier(num_features=len(vectorizer.review_vocab))
    classifier = classifier.to(args["device"])

    loss_func = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=args["learning_rate"])

    train(args, train_state, dataset, classifier, optimizer, loss_func, compute_accuracy)

    return {
        "train_state": train_state,
        "args": args,
        "dataset": dataset,
        "classifier": classifier,
        "loss_func": loss_func,
        "optimizer": optimizer,
    }
