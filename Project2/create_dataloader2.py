import os

import numpy as np
import spacy
import torch
import torch.nn.functional as F
from dataset import my_load_dataset
from embedding_models import FastTextEmbedder, GloveEmbedder
from torch.utils.data import DataLoader, TensorDataset

PAD, UNK = 0, 1
EMBEDDING_DIM = 300
CACHE_FORMAT_VERSION = 2


class CreateDataLoaders:
    def __init__(
        self,
        dataset,
        emotions,
        batch_size=32,
        word_embedder=None,
        npartitions=8,
        cache_path="goemotions_ids.pt",
        seed=42,
        num_workers=0,
    ):
        if word_embedder is None:
            raise ValueError("A FastTextEmbedder or GloveEmbedder is required")
        if len(dataset) == 0:
            raise ValueError("The dataset is empty")

        self.dataset = dataset
        self.emotions = list(emotions)
        self.batch_size = batch_size
        self.word_embedder = word_embedder
        self.npartitions = npartitions
        self.cache_path = cache_path
        self.seed = seed
        self.num_workers = num_workers
        self._nlp = spacy.blank("en")

    def _tokenize(self, text):
        """Tokenize without normalizing content that may carry emotional meaning."""
        if not isinstance(text, str):
            text = str(text)
        return [token.text for token in self._nlp.make_doc(text)]

    def _emotion_index(self, emotion):
        if (
            isinstance(emotion, (int, np.integer))
            or isinstance(emotion, str)
            and emotion.isdigit()
        ):
            emotion_index = int(emotion)
        else:
            if emotion not in self.emotions:
                raise ValueError(f"Unknown emotion label: {emotion!r}")
            return self.emotions.index(emotion)

        if 0 <= emotion_index < len(self.emotions):
            return emotion_index
        raise ValueError(f"Emotion index out of range: {emotion_index}")

    def _build_vocab(self):
        words = sorted(
            {token for text in self.dataset["text"] for token in self._tokenize(text)}
        )
        vocab = {w: i + 2 for i, w in enumerate(words)}  # 0=PAD, 1=UNK

        probe = np.asarray(self.word_embedder.embed(words[0] if words else "test"))
        dim = probe.reshape(-1).shape[0]
        if dim != EMBEDDING_DIM:
            raise ValueError(
                f"Expected {EMBEDDING_DIM}-dimensional embeddings, got {dim}"
            )

        weights = np.zeros((len(vocab) + 2, dim), dtype=np.float32)
        for w, i in vocab.items():  # embed each word once
            weights[i] = np.asarray(
                self.word_embedder.embed(w), dtype=np.float32
            ).reshape(-1)
        return vocab, torch.from_numpy(weights)

    def _build_tensors(self):
        vocab, weights = self._build_vocab()

        seqs = [
            [vocab.get(token, UNK) for token in self._tokenize(text)]
            for text in self.dataset["text"]
        ]

        max_len = max((len(sequence) for sequence in seqs), default=1)
        max_len = max(max_len, 1)
        X = torch.zeros(len(seqs), max_len, dtype=torch.long)
        for i, s in enumerate(seqs):
            X[i, : len(s)] = torch.tensor(s)

        y_idx = torch.tensor(
            [self._emotion_index(emotion) for emotion in self.dataset["emotion"]],
            dtype=torch.long,
        )
        y = F.one_hot(y_idx, num_classes=len(self.emotions)).float()
        return X, y, weights

    def _split(self, X, y):
        g = torch.Generator().manual_seed(self.seed)
        perm = torch.randperm(len(X), generator=g)
        X, y = X[perm], y[perm]
        n_tr, n_val = int(0.8 * len(X)), int(0.1 * len(X))
        return {
            "train": (X[:n_tr], y[:n_tr]),
            "val": (X[n_tr : n_tr + n_val], y[n_tr : n_tr + n_val]),
            "test": (X[n_tr + n_val :], y[n_tr + n_val :]),
        }

    def create_dataloaders(self):
        """Returns train, val, test loaders and the embedding weights."""
        metadata = {
            "cache_format_version": CACHE_FORMAT_VERSION,
            "embedder": type(self.word_embedder).__name__,
            "emotions": self.emotions,
            "seed": self.seed,
            "dataset_size": len(self.dataset),
        }
        cache = None

        if os.path.exists(self.cache_path):
            try:
                loaded = torch.load(self.cache_path, map_location="cpu")
                if all(loaded.get(key) == value for key, value in metadata.items()):
                    cache = loaded
            except OSError, RuntimeError, EOFError, KeyError, TypeError:
                cache = None

        if cache is None:
            X, y, weights = self._build_tensors()
            cache = {"splits": self._split(X, y), "weights": weights}
            cache.update(metadata)
            torch.save(cache, self.cache_path)

        def make(name, shuffle):
            X, y = cache["splits"][name]
            return DataLoader(
                TensorDataset(X, y),
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=self.num_workers,
            )

        return (
            make("train", True),
            make("val", False),
            make("test", False),
            cache["weights"],
        )


if __name__ == "__main__":
    my_glove_embedder = GloveEmbedder()
    my_fasttext_embedder = FastTextEmbedder()
    dataset, emotions = my_load_dataset()

    my_dataloader_creator = CreateDataLoaders(
        dataset, emotions, word_embedder=my_glove_embedder
    )
    train_loader, val_loader, test_loader, weights = (
        my_dataloader_creator.create_dataloaders()
    )
    # Test the dataloaders by iterating through one batch from each
    for loader, name in zip(
        [train_loader, val_loader, test_loader], ["train", "val", "test"]
    ):
        print(f"{name} loader:")
        for batch in loader:
            X, y = batch
            print(f"X shape: {X.shape}, y shape: {y.shape}")
            break
