import ast

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

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

# Each line is a vector as list


def read_vectors(path):
    with open(path, "r", encoding="utf-8") as f:
        return np.array(
            [ast.literal_eval(line) for line in f],
            dtype=np.float32,
        )


glove_vectors = read_vectors("glove_vectors.txt")
fasttext_vectors = read_vectors("fasttext_vectors.txt")

print(glove_vectors)
print(fasttext_vectors)

fig, axs = plt.subplots(1, 2, figsize=(16, 6))
# Do PCA on GloVe vectors and plot the first two principal components with word as label
pca = PCA(n_components=2)
glove_pca = pca.fit_transform(glove_vectors)
axs[0].scatter(glove_pca[:, 0], glove_pca[:, 1], marker="x", s=10, color="black")
# Add labels for each point
for i, emotion in enumerate(emotions):
    axs[0].annotate(emotion, (glove_pca[i, 0], glove_pca[i, 1]), fontsize=8)
axs[0].set_title("PCA of GloVe Vectors")
axs[0].set_xlabel("Principal Component 1")
axs[0].set_ylabel("Principal Component 2")

# Do PCA on FastText vectors and plot the first two principal components with word as label
pca = PCA(n_components=2)
fasttext_pca = pca.fit_transform(fasttext_vectors)
axs[1].scatter(fasttext_pca[:, 0], fasttext_pca[:, 1], marker="x", s=10, color="black")
# Add labels for each point
for i, emotion in enumerate(emotions):
    axs[1].annotate(emotion, (fasttext_pca[i, 0], fasttext_pca[i, 1]), fontsize=8)
axs[1].set_title("PCA of FastText Vectors")
axs[1].set_xlabel("Principal Component 1")
axs[1].set_ylabel("Principal Component 2")

plt.tight_layout()
plt.savefig("pca_plot.png")
