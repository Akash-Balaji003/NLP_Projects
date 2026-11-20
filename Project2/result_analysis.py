import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _f1_value(score):
    return score[0] if isinstance(score, tuple) else score


def _counts(scores):
    return {
        label: score[1] for label, score in scores.items() if isinstance(score, tuple)
    }


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
        f1_scores[label] = (f1_score, tp + fn)
    return f1_scores


def per_length_f1(file):
    df = pd.read_csv(file)
    # Columns are input_text, class_prediction, true_label
    # Calculate F1 score for each input length
    f1_scores = {}
    # Group lengths inclusively: 0-10, 11-20, ..., 91-100.
    text_lengths = df["input_text"].str.len()
    for group_end in range(10, 181, 10):
        group_start = 0 if group_end == 10 else group_end - 9
        subset = df[(text_lengths >= group_start) & (text_lengths <= group_end)]
        tp = len(
            subset[
                (subset["class_prediction"] == subset["true_label"])
                & (subset["true_label"] != "")
            ]
        )
        fp = len(
            subset[
                (subset["class_prediction"] != subset["true_label"])
                & (subset["true_label"] != "")
            ]
        )
        fn = len(
            subset[
                (subset["class_prediction"] != subset["true_label"])
                & (subset["true_label"] != "")
            ]
        )

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        f1_scores[group_end] = (f1_score, tp + fn)

    return f1_scores


def predict_correctness_by_class(file):
    df = pd.read_csv(file)
    # make classes ints
    classes = pd.get_dummies(df["true_label"]).to_numpy()
    correctness = (df["class_prediction"] == df["true_label"]).to_numpy()

    model = LogisticRegression()
    model.fit(classes, correctness)
    # evalute the model's accuracy
    accuracy = model.score(classes, correctness)

    return accuracy


def predict_correctness_by_length(file):
    df = pd.read_csv(file)
    lengths = df["input_text"].str.len().to_numpy().reshape(-1, 1)
    correctness = (df["class_prediction"] == df["true_label"]).to_numpy()

    model = LogisticRegression()
    model.fit(lengths, correctness)
    # evalute the model's accuracy
    accuracy = model.score(lengths, correctness)

    return accuracy


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

    plt.figure(figsize=(8, 8))
    count_scores = next(
        (scores for scores in f1_scores if _counts(scores)),
        {},
    )
    class_counts = _counts(count_scores)
    heatmap = [[_f1_value(f1.get(label, 0)) for f1 in f1_scores] for label in labels]
    image = plt.imshow(heatmap, vmin=0, vmax=1, aspect="auto")
    plt.colorbar(image, label="F1 Score")
    plt.ylabel("Classes")
    titles = [
        f"{title}\n(macro F1: {sum(_f1_value(score) for score in f1.values()) / len(f1):.2f})"
        for f1, title in zip(f1_scores, titles)
    ]
    plt.xticks(range(len(titles)), titles)
    plt.yticks(
        range(len(labels)),
        [f"{label} (n={class_counts.get(label, 0)})" for label in labels],
    )

    for row_index, row in enumerate(heatmap):
        for column_index, score in enumerate(row):
            plt.text(
                column_index,
                row_index,
                f"{score:.2f}",
                ha="center",
                va="center",
            )

    plt.title("Per-Class F1 Scores")
    plt.savefig(filename, bbox_inches="tight")
    plt.show()


def plot_per_length_f1(f1_scores, titles, macros, filename):
    plt.figure(figsize=(8, 6))
    group_ends = list(range(10, 181, 10))
    bar_width = 0.8 / len(f1_scores)
    positions = range(len(group_ends))
    for index, (scores, title, macro) in enumerate(zip(f1_scores, titles, macros)):
        offsets = [position + index * bar_width for position in positions]
        plt.bar(
            offsets,
            [_f1_value(scores.get(group_end, 0)) for group_end in group_ends],
            width=bar_width,
            label=f"{title} (macro F1: {macro:.2f})",
        )

    plt.xlabel("Input length")
    plt.ylabel("F1 Score")
    length_counts = _counts(f1_scores[0])
    plt.xticks(
        [position + bar_width * (len(f1_scores) - 1) / 2 for position in positions],
        [
            f"{0 if group_end == 10 else group_end - 9}-{group_end} "
            f"\n(n={length_counts.get(group_end, 0)})"
            for group_end in group_ends
        ],
        size=8,
        rotation=90,
    )
    plt.ylim(0, 1)
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.title("F1 Scores by Input Length")
    plt.savefig(filename)
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
    glove_sum_length = per_length_f1("glove_sum.csv")
    glove_lstm = per_class_f1("glove_lstm.csv")
    glove_lstm_length = per_length_f1("glove_lstm.csv")
    macro_f1_baseline = sum(baseline.values()) / len(baseline)
    macro_f1_glove_sum = sum(_f1_value(score) for score in glove_sum.values()) / len(
        glove_sum
    )
    macro_f1_glove_lstm = sum(_f1_value(score) for score in glove_lstm.values()) / len(
        glove_lstm
    )

    print(f"Macro F1 (Baseline): {macro_f1_baseline:.2f}")
    print(f"Macro F1 (GloVe Sum): {macro_f1_glove_sum:.2f}")
    print(f"Macro F1 (GloVe LSTM): {macro_f1_glove_lstm:.2f}")

    glove_sum_class_accuracy = predict_correctness_by_class("glove_sum.csv")
    glove_lstm_class_accuracy = predict_correctness_by_class("glove_lstm.csv")

    print("Prediction accuracy based on class:")
    print(f"Accuracy (GloVe Sum): {glove_sum_class_accuracy:.2f}")
    print(f"Accuracy (GloVe LSTM): {glove_lstm_class_accuracy:.2f}")

    glove_sum_length_accuracy = predict_correctness_by_length("glove_sum.csv")
    glove_lstm_length_accuracy = predict_correctness_by_length("glove_lstm.csv")

    print("Prediction accuracy based on input length:")
    print(f"Accuracy (GloVe Sum): {glove_sum_length_accuracy:.2f}")
    print(f"Accuracy (GloVe LSTM): {glove_lstm_length_accuracy:.2f}")

    plot_per_class_f1(
        [baseline, glove_sum, glove_lstm],
        ["Baseline", "GloVe Sum", "GloVe LSTM"],
        "f1_per_class.png",
    )

    plot_per_length_f1(
        [glove_sum_length, glove_lstm_length],
        ["GloVe Sum", "GloVe LSTM"],
        [macro_f1_glove_sum, macro_f1_glove_lstm],
        "f1_per_length.png",
    )
