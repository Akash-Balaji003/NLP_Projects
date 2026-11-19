import argparse
import os
from pathlib import Path
import sys

import pandas as pd
import torch

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from create_dataloader2 import CreateDataLoaders
from dataset import my_load_dataset
from embedding_models import FastTextEmbedder, GloveEmbedder
from logistic_regression2 import EmotionClassifier


def create_embedder(name):
    if name == "glove":
        return GloveEmbedder()
    if name == "fasttext":
        return FastTextEmbedder()
    raise ValueError(f"Unknown embedding model: {name}")


def load_model(checkpoint_path, weights, n_classes, pooling, device):
    try:
        model = EmotionClassifier.load_from_checkpoint(
            checkpoint_path,
            weights=weights,
            n_classes=n_classes,
            pooling=pooling,
            map_location=device,
        )
    except RuntimeError, KeyError, TypeError:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        state_dict = checkpoint.get("state_dict", checkpoint)
        model = EmotionClassifier(weights, n_classes, pooling=pooling)
        model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()
    return model


def save_test_predictions(checkpoint_path, embedding, pooling, output_path):
    dataset, emotions = my_load_dataset()
    train_loader, val_loader, test_loader, weights = CreateDataLoaders(
        dataset,
        emotions,
        word_embedder=create_embedder(embedding),
        cache_path=f"goemotions_{embedding}_ids.pt",
    ).create_dataloaders()

    del train_loader, val_loader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path, weights, len(emotions), pooling, device)

    predicted_indices = []
    true_indices = []

    with torch.no_grad():
        for ids, labels in test_loader:
            ids = ids.to(device)
            logits = model(ids)
            predicted_indices.extend(logits.argmax(dim=1).cpu().tolist())
            true_indices.extend(labels.argmax(dim=1).cpu().tolist())

    predictions = pd.DataFrame(
        {
            "predicted_index": predicted_indices,
            "predicted_emotion": [emotions[index] for index in predicted_indices],
            "true_index": true_indices,
            "true_emotion": [emotions[index] for index in true_indices],
        }
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    print(f"Saved {len(predictions)} test predictions to " f"{output_path.resolve()}")


if __name__ == "__main__":
    os.chdir(PROJECT_DIR)

    parser = argparse.ArgumentParser(
        description="Load a checkpoint and save test-loader predictions."
    )
    parser.add_argument("checkpoint", type=Path, help="Path to a Lightning checkpoint")
    parser.add_argument("--embedding", choices=["glove", "fasttext"], default="glove")
    parser.add_argument("--pooling", choices=["sum", "mean", "lstm"], default="mean")
    parser.add_argument("--output", type=Path, default=Path("test_predictions.csv"))
    args = parser.parse_args()

    save_test_predictions(
        args.checkpoint,
        embedding=args.embedding,
        pooling=args.pooling,
        output_path=args.output,
    )
