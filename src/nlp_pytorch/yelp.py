from collections import defaultdict, Counter
import re
import string

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

TRAIN = 'train'
TEST = 'test'
VAL = 'val'


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
            item['split'] = TRAIN

        for item in item_list[n_train:n_train+n_val]:
            item['split'] = VAL

        for item in item_list[n_train+n_val:n_train+n_val+n_test]:
            item['split'] = TEST

        final_list.extend(item_list)

    return pd.DataFrame(final_list)


class Vocabulary(object):
    def __init__(self, token_to_idx=None, add_unk=True, unk_token="<UNK>"):
        if token_to_idx is None:
            token_to_idx = {}
        self._token_to_idx = token_to_idx

        self._idx_to_token = {idx: token for token, idx in self._token_to_idx.items()}

        self._add_unk = add_unk
        self._unk_token = unk_token

        self.unk_index = -1
        if add_unk:
            self.unk_index = self.add_token(unk_token)

    def to_serializable(self):
        return {
            'token_to_idx': self._token_to_idx,
            'add_unk': self._add_unk,
            'unk_token': self._unk_token,
        }

    @classmethod
    def from_serializable(cls, contents):
        return cls(**contents)

    def add_token(self, token):
        if token in self._token_to_idx:
            index = self._token_to_idx[token]
        else:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token

        return index

    def lookup_token(self, token):
        if self._add_unk:
            return self._token_to_idx.get(token, self.unk_index)
        else:
            return self._token_to_idx[token]

    def lookup_index(self, index):
        if index not in self._idx_to_token:
            raise KeyError(f"the index ({index}) is not in the vocabulary")
        return self._idx_to_token[index]

    def __str__(self):
        return f"<Vocabulary(size={len(self)}"

    def __len__(self):
        return len(self._token_to_idx)


class ReviewVectorizer(object):
    def __init__(self, review_vocab, rating_vocab):
        self.review_vocab = review_vocab
        self.rating_vocab = rating_vocab

    def vectorize(self, review):
        one_hot = np.zeros(len(self.review_vocab), dtype=np.float32)

        for token in review.split(' '):
            if token not in string.punctuation:
                one_hot[self.review_vocab.lookup_token(token)] = 1

        return one_hot

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
            'review_vocab': self.review_vocab.to_serializeable(),
            'rating_vocab': self.rating_vocab.to_serializeable(),
        }


class ReviewDataset(Dataset):
    def __init__(self, review_df, vectorizer):
        self.review_df = review_df
        self._vectorizer = vectorizer

        self.train_df = self.review_df[self.review_df.split == TRAIN]
        self.train_size = len(self.train_df)

        self.val_df = self.review_df[self.review_df.split == VAL]
        self.val_size = len(self.val_df)

        self.test_df = self.review_df[self.review_df.split == TEST]
        self.test_size = len(self.test_df)

        self._lookup_dict = {
            TRAIN: (self.train_df, self.train_size),
            VAL: (self.val_df, self.val_size),
            TEST: (self.test_df, self.test_size),
        }

        self.set_split(TRAIN)

    @classmethod
    def load_dataset_and_make_vectorizer(cls, review_csv):
        review_df = pd.read_json(review_csv, lines=True)
        review_df["rating"] = review_df["stars"] <= 3
        review_df["review"] = review_df["text"]
        review_df = split_data(review_df, 0.70, 0.15, 0.15)
        return cls(review_df, ReviewVectorizer.from_dataframe(review_df))

    def get_vectorizer(self):
        return self._vectorizer

    def set_split(self, split=TRAIN):
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]

    def __len__(self):
        return self._target_size

    def __getitem__(self, index):
        row = self._target_df.iloc[index]
        review_vector = self._vectorizer.vectorize(row.review)
        rating_index = self._vectorizer.rating_vocab.lookup_token(row.rating)
        return {'x_data': review_vector, 'y_target': rating_index}

    def get_num_batches(self, batch_size):
        return len(self) // batch_size


def generate_batches(dataset, batch_size, shuffle=True, drop_last=True, device="cpu"):
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

    for data_dict in dataloader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name].to(device)
        yield out_data_dict


class ReviewClassifier(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features=num_features, out_features=1)

    def forward(self, x_in, apply_sigmoid=False):
        y_out = self.fc1(x_in).squeeze()
        if apply_sigmoid:
            y_out = F.sigmoid(y_out)
        return y_out


def make_train_state():
    return {
        'epoch_index': 0,
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'test_loss': -1,
        'test_acc': -1
    }


def compute_accuracy(y_pred, y_target):
    y_target = y_target.cpu()
    y_pred_indices = (torch.sigmoid(y_pred) > 0.5).cpu().long()
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100


def train(args, train_state, dataset, classifier, optimizer, loss_func):
    for epoch_index in range(args['num_epochs']):
        train_state["epoch_index"] = epoch_index

        dataset.set_split(TRAIN)
        batch_generator = generate_batches(dataset, batch_size=args["batch_size"], device=args["device"])
        running_loss = 0.0
        running_acc = 0.0
        classifier.train()

        for batch_index, batch_dict in enumerate(batch_generator):
            optimizer.zero_grad()
            y_pred = classifier(x_in=batch_dict['x_data'].float())

            loss = loss_func(y_pred, batch_dict['y_target'].float())
            loss_batch = loss.item()
            running_loss += (loss_batch - running_loss) / (batch_index + 1)

            loss.backward()

            optimizer.step()

            acc_batch = compute_accuracy(y_pred, batch_dict['y_target'])
            running_acc += (acc_batch - running_acc) / (batch_index + 1)

        train_state["train_loss"].append(running_loss)
        train_state["train_acc"].append(running_acc)

        print(f"Epoch {epoch_index} / {args['num_epochs']}")
        print(train_state)

        dataset.set_split(VAL)
        batch_generator = generate_batches(dataset, batch_size=args["batch_size"], device=args["device"])
        running_loss = 0.
        running_acc = 0.
        classifier.eval()

        for batch_index, batch_dict in enumerate(batch_generator):
            y_pred = classifier(x_in=batch_dict['x_data'].float())
            loss = loss_func(y_pred, batch_dict['y_target'].float())
            loss_batch = loss.item()
            running_loss += (loss_batch - running_acc) / (batch_index + 1)

            acc_batch = compute_accuracy(y_pred, batch_dict["y_target"])
            running_acc += (acc_batch - running_acc) / (batch_index + 1)

        print(f"Val Epoch {epoch_index}")
        train_state['val_loss'].append(running_loss)
        train_state['val_acc'].append(running_acc)

        print(train_state)


def main():
    args = {
        "review_csv": "data/yelp_reviews_lite.json",
        "save_dir": "model_storage/yelp/",
        "model_state_file": "model.pth",
        "vectorizer_file": "vectorizer.json",
        "learning_rate": 0.001,
        "num_epochs": 100,
        "batch_size": 128,
        "early_stopping_criteria": 5,
        "frequency_cutoff": 25,
        "cuda": False,
    }
    train_state = make_train_state()

    if torch.cuda.is_available():
        args['cuda'] = True
    args['device'] = torch.device("cuda:0" if args["cuda"] else "cpu")
    print(args)

    dataset = ReviewDataset.load_dataset_and_make_vectorizer(args["review_csv"])
    vectorizer = dataset.get_vectorizer()

    classifier = ReviewClassifier(num_features=len(vectorizer.review_vocab))
    classifier = classifier.to(args["device"])

    loss_func = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=args["learning_rate"])

    train(args, train_state, dataset, classifier, optimizer, loss_func)

    return {
        "train_state": train_state,
        "args": args,
        "dataset": dataset,
        "classifier": classifier,
        "loss_func": loss_func,
        "optimizer": optimizer,
    }
