import os
import re
import numpy as np
import pandas as pd
import dask.dataframe as dd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


PAD, UNK = 0, 1

class CreateDataLoaders:
    def __init__(self, dataset, emotions, batch_size=32, word_embedder=None,
                 npartitions=8, cache_path="goemotions_ids.pt",
                 seed=42, num_workers=0):
        self.dataset = dataset
        self.emotions = list(emotions)
        self.batch_size = batch_size
        self.word_embedder = word_embedder
        self.npartitions = npartitions
        self.cache_path = cache_path
        self.seed = seed
        self.num_workers = num_workers

    @staticmethod
    def _tokenize(text):
        # must match the tokenization your embedder expects
        return re.findall(r"\w+|[^\w\s]", text.lower())

    def _build_vocab(self):
        ddf = dd.from_pandas(self.dataset[["text"]], npartitions=self.npartitions)
        # parallel: set of unique words per partition
        sets = ddf.map_partitions(
            lambda p: pd.Series([{w for t in p["text"] for w in self._tokenize(t)}]),
            meta=("w", object),
        ).compute()
        words = sorted(set().union(*sets))

        vocab = {w: i + 2 for i, w in enumerate(words)}      # 0=PAD, 1=UNK
        dim = np.asarray(self.word_embedder.embed(words[0])).reshape(-1, ).shape[0]
        weights = np.zeros((len(vocab) + 2, dim), dtype=np.float32)
        for w, i in vocab.items():                           # embed each word once
            weights[i] = np.asarray(self.word_embedder.embed(w), dtype=np.float32).reshape(-1)
        return vocab, torch.from_numpy(weights)

    def _build_tensors(self):
        vocab, weights = self._build_vocab()

        def to_ids(part):
            return pd.Series(
                [[vocab.get(w, UNK) for w in self._tokenize(t)] for t in part["text"]],
                index=part.index, dtype=object)

        ddf = dd.from_pandas(self.dataset[["text"]], npartitions=self.npartitions)
        seqs = ddf.map_partitions(to_ids, meta=("ids", object)).compute().tolist()

        max_len = max(len(s) for s in seqs)
        X = torch.zeros(len(seqs), max_len, dtype=torch.long)
        for i, s in enumerate(seqs):
            X[i, :len(s)] = torch.tensor(s)

        e2i = {e: i for i, e in enumerate(self.emotions)}
        y_idx = torch.tensor([e2i[e] for e in self.dataset["emotion"]])
        y = F.one_hot(y_idx, num_classes=len(self.emotions)).float()
        return X, y, weights

    def _split(self, X, y):
        g = torch.Generator().manual_seed(self.seed)
        perm = torch.randperm(len(X), generator=g)
        X, y = X[perm], y[perm]
        n_tr, n_val = int(0.8 * len(X)), int(0.1 * len(X))
        return {"train": (X[:n_tr], y[:n_tr]),
                "val": (X[n_tr:n_tr + n_val], y[n_tr:n_tr + n_val]),
                "test": (X[n_tr + n_val:], y[n_tr + n_val:])}

    def create_dataloaders(self):
        """Returns train, val, test loaders and the embedding weights."""
        if os.path.exists(self.cache_path):
            cache = torch.load(self.cache_path)
        else:
            X, y, weights = self._build_tensors()
            cache = {"splits": self._split(X, y), "weights": weights}
            torch.save(cache, self.cache_path)

        def make(name, shuffle):
            X, y = cache["splits"][name]
            return DataLoader(TensorDataset(X, y), batch_size=self.batch_size,
                              shuffle=shuffle, num_workers=self.num_workers)

        return make("train", True), make("val", False), make("test", False), cache["weights"]