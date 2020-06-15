from __future__ import annotations

from typing import Dict, Optional, Any, List

import numpy as np


class Vocabulary(object):
    def __init__(
        self,
        token_to_idx: Optional[Dict[str, int]] = None,
        add_unk: bool = True,
        unk_token: str = "<UNK>",
        mask_token: str = "<MASK>",
        begin_seq_token: str = "<BEGIN>",
        end_seq_token: str = "<END>",
    ) -> None:
        if token_to_idx is None:
            token_to_idx = {}
        self._token_to_idx = token_to_idx

        self._idx_to_token = {idx: token for token, idx in self._token_to_idx.items()}

        self._add_unk = add_unk
        self._unk_token = unk_token
        self._mask_token = mask_token
        self.begin_seq_token = begin_seq_token
        self.end_seq_token = end_seq_token

        self.mask_index = self.add_token(mask_token)
        self.begin_seq_index = self.add_token(begin_seq_token)
        self.end_seq_index = self.add_token(end_seq_token)
        self.unk_index = -1
        if add_unk:
            self.unk_index = self.add_token(unk_token)

    def to_serializable(self) -> Dict[str, Any]:
        return {
            "token_to_idx": self._token_to_idx,
            "add_unk": self._add_unk,
            "unk_token": self._unk_token,
            "mask_token": self._mask_token,
            "begin_seq_token": self.begin_seq_token,
            "end_seq_token": self.end_seq_token,
        }

    @classmethod
    def from_serializable(cls, contents: Dict[str, Any]) -> Vocabulary:
        return cls(**contents)

    def add_token(self, token: str) -> int:
        if token not in self._token_to_idx:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token

        return self._token_to_idx[token]

    def lookup_token(self, token: str) -> int:
        if self._add_unk:
            return self._token_to_idx.get(token, self.unk_index)
        else:
            return self._token_to_idx[token]

    def lookup_index(self, index: int) -> str:
        if index not in self._idx_to_token:
            raise KeyError(f"the index ({index}) is not in the vocabulary")
        return self._idx_to_token[index]

    def __str__(self) -> str:
        return f"<Vocabulary(size={len(self)}"

    def __len__(self) -> int:
        return len(self._token_to_idx)

    def one_hot_encoding(self, tokens: List[str]) -> np.array:
        one_hot = np.zeros(len(self), dtype=np.float32)
        for token in tokens:
            one_hot[self.lookup_token(token)] = 1.0
        return one_hot
