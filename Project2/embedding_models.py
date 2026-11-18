import gzip
import os
import shutil
import urllib.request
import zipfile

import fasttext_new as fasttext
import numpy as np


def load_fasttext_model():
    url = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz"

    compressed_file = "cc.en.300.bin.gz"
    model_file = "cc.en.300.bin"

    # Download model if necessary
    if not os.path.exists(compressed_file):
        print("Downloading FastText model...")
        urllib.request.urlretrieve(url, compressed_file)
        print("Download complete.")

    # Decompress model if necessary
    if not os.path.exists(model_file):
        print("Extracting FastText model...")

        with gzip.open(compressed_file, "rb") as f_in, open(model_file, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

        print("Extraction complete.")

    # Load pretrained model
    model = fasttext.load_model(model_file)

    return model


def load_glove_model():
    url = "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.840B.300d.zip"

    compressed_file = "glove.840B.300d.zip"
    model_file = "glove.840B.300d.txt"

    # Download model if necessary
    if not os.path.exists(compressed_file):
        print("Downloading GloVe model...")
        urllib.request.urlretrieve(url, compressed_file)
        print("Download complete.")

    # Extract model if necessary
    if not os.path.exists(model_file):
        print("Extracting GloVe model...")

        with zipfile.ZipFile(compressed_file, "r") as zip_ref:
            zip_ref.extractall(".")

        print("Extraction complete.")

    # Load pretrained embeddings
    print("Loading GloVe embeddings...")

    embeddings = {}

    with open(model_file, "r", encoding="utf-8") as f:
        for line in f:
            values = line.rstrip().split(" ")

            word = values[0]
            vector = np.asarray(values[1:], dtype=np.float32)

            embeddings[word] = vector

    print(f"Loaded {len(embeddings):,} word embeddings.")

    return embeddings
