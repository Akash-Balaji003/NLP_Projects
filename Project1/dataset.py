import kagglehub
import pandas as pd
import os

def download_dataset():
    # Download latest version
    for end in ["train", "dev", "test"]:
        kagglehub.dataset_download("debarshichanda/goemotions", path=f"data/{end}.tsv", output_dir="./")


def my_load_dataset():
    """
    Load the three files from the data directory and combine them.

    Returns:
        pd.DataFrame: The dataset as a pandas DataFrame.
    """
    if not os.path.exists("data"):
        download_dataset()
    # Load the three files
    # there are no headers in the files, so we need to specify them
    headers = ["text", "emotion", "unknown"]
    for end in ["train", "dev", "test"]:
        file_path = f"data/{end}.tsv"
        df = pd.read_csv(file_path, sep='\t', names=headers)
        if end == "train":
            dataset = df
        else:
            dataset = pd.concat([dataset, df], ignore_index=True)

    # remove unnecessary columns
    dataset = dataset.drop(columns=["unknown"])

    # we only want one emotion per text, 
    # so we filter out all rows where emotion has more than one label (there is a comma in the string)
    dataset = dataset[dataset["emotion"].str.contains(",") == False]

    # drop duplicates
    dataset = dataset.drop_duplicates(subset=["text"])

    # Remove all neutral emotions (emotion == 27)
    dataset = dataset[dataset["emotion"] != "27"]

    # List of emotions in the dataset
    emotions = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise"]

    return dataset, emotions



if __name__ == "__main__":
    dataset, emotions = my_load_dataset()
    print(f"Dataset shape: {dataset.shape}")
    print("Occurence of each emotion in the dataset:")
    print(dataset["emotion"].value_counts().sort_index(key=lambda x: x.astype(int)))
    print(f"Emotions: {emotions}")
