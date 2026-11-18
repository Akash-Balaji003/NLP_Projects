import gzip
import os
import shutil
import urllib.request
import zipfile

import fasttext_new as fasttext
import numpy as np


def load_fasttext_model():
    url = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz"

    os.makedirs("models", exist_ok=True)

    compressed_file = "models/cc.en.300.bin.gz"
    model_file = "models/cc.en.300.bin"

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

    os.makedirs("models", exist_ok=True)

    compressed_file = "models/glove.840B.300d.zip"
    model_file = "models/glove.840B.300d.txt"

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


class GloveEmbedder:
    def __init__(self):                 
        self.embedding = load_glove_model()

    def embed(self, word):
        return self.embedding.get(word, np.zeros(300, dtype=np.float32))  # Return zero vector if word not found
    


class FastTextEmbedder:
    def __init__(self):
        self.fasttext_model = load_fasttext_model()

    def embed(self, word):
        return self.fasttext_model.get_word_vector(word)

if __name__ == "__main__":
    emotions = [
        "admiration",
        "amusement",
        "anger",
        "annoyance",
        "approval",
        "caring",
        "confusion",
        "curiosity",
        "desire",
        "disappointment",
        "disapproval",
        "disgust",
        "embarrassment",
        "excitement",
        "fear",
        "gratitude",
        "grief",
        "joy",
        "love",
        "nervousness",
        "optimism",
        "pride",
        "realization",
        "relief",
        "remorse",
        "sadness",
        "surprise",
    ]
    fasttext_model = load_fasttext_model()
    glove_embeddings = load_glove_model()

    # Example usage
    for emotion in emotions:
        fasttext_vector = fasttext_model.get_word_vector(emotion)
        glove_vector = glove_embeddings.get(emotion)

        # save the vectors to a file or use them for further processing
        print(f"Emotion: {emotion}")
        print(f"    FastText Vector: {fasttext_vector[:5]}...")
        with open("fasttext_vectors.txt", "a") as f:
            f.write(f"{fasttext_vector.tolist()}\n")
        if glove_vector is not None:
            print(f"    GloVe Vector: {glove_vector[:5]}...")
            with open("glove_vectors.txt", "a") as f:
                f.write(f"{glove_vector.tolist()}\n")
        else:
            print("GloVe Vector: Not found")
            with open("glove_vectors.txt", "a") as f:
                f.write("Not found\n")
