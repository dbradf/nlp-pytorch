import pandas as pd
from torch.utils.data import Dataset, DataLoader

TRAIN = "train"
TEST = "test"
VAL = "val"


class SplitDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, vectorizer) -> None:
        self.dataframe = dataframe
        self.vectorizer = vectorizer

        self.train_df = self.dataframe[self.dataframe.split == TRAIN]
        self.train_size = len(self.train_df)

        self.val_df = self.dataframe[self.dataframe.split == VAL]
        self.val_size = len(self.val_df)

        self.test_df = self.dataframe[self.dataframe.split == TEST]
        self.test_size = len(self.test_df)

        self._lookup_dict = {
            TRAIN: (self.train_df, self.train_size),
            VAL: (self.val_df, self.val_size),
            TEST: (self.test_df, self.test_size),
        }

        self.set_split(TRAIN)

    def set_split(self, split: str = TRAIN):
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]

    def get_num_batches(self, batch_size):
        return len(self) // batch_size

    def __len__(self):
        return self._target_size


def generate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = True,
    device: str = "cpu",
):
    dataloader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
    )

    for data_dict in dataloader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name].to(device)
        yield out_data_dict
