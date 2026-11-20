import matplotlib.pyplot as plt
import pandas as pd


def per_class_f1(file):
    df = pd.read_csv(file)
    # Columns are input_text, class_prediction, true_label
    # Calculate F1 score for each class
    f1_scores = {}
    for label in df["true_label"].unique():
        tp = len(df[(df["class_prediction"] == label) & (df["true_label"] == label)])
        fp = len(df[(df["class_prediction"] == label) & (df["true_label"] != label)])
        fn = len(df[(df["class_prediction"] != label) & (df["true_label"] == label)])

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        f1_scores[label] = f1_score

    return f1_scores


def plot_per_class_f1(f1_scores, titles, filename):
    labels = [
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
    plt.figure(figsize=(12, 6))
    plt.xlabel("Classes")
    plt.ylabel("F1 Score")
    plt.xticks(rotation=90)
    x = range(len(labels))
    bar_width = 0.8 / len(f1_scores)

    macro_f1s = []
    for index, (f1, title) in enumerate(zip(f1_scores, titles)):
        scores = list(f1.values())
        macro_f1 = sum(scores) / len(scores)
        macro_f1s.append(macro_f1)
        offsets = [position + index * bar_width for position in x]
        plt.bar(
            offsets,
            scores,
            width=bar_width,
            label=f"{title} (Macro F1: {macro_f1:.2f})",
        )

    plt.xticks(
        [position + (len(f1_scores) - 1) * bar_width / 2 for position in x],
        labels,
        rotation=90,
    )
    plt.title("Per-Class F1 Scores")
    plt.legend()
    plt.ylim(0, 1)
    plt.savefig(filename, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # hardcode baseline for
    baseline = {
        "admiration": 0.65,
        "amusement": 0.80,
        "anger": 0.47,
        "annoyance": 0.34,
        "approval": 0.36,
        "caring": 0.39,
        "confusion": 0.37,
        "curiosity": 0.54,
        "desire": 0.49,
        "disappointment": 0.28,
        "disapproval": 0.39,
        "disgust": 0.45,
        "embarrassment": 0.43,
        "excitement": 0.34,
        "fear": 0.60,
        "gratitude": 0.86,
        "grief": 0.00,
        "joy": 0.51,
        "love": 0.78,
        "nervousness": 0.35,
        "optimism": 0.51,
        "pride": 0.36,
        "realization": 0.21,
        "relief": 0.15,
        "remorse": 0.66,
        "sadness": 0.49,
        "surprise": 0.50,
    }
    # Read the CSV files and calculate F1 scores
    glove_sum = per_class_f1("glove_sum.csv")

    plot_per_class_f1(
        [baseline, glove_sum],
        ["Baseline", "GloVe Sum"],
        "f1.png",
    )
