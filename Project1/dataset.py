import kagglehub
import pandas as pd

def download_dataset():
    # Download latest version
    kagglehub.dataset_download("debarshichanda/goemotions", path="data/full_dataset", output_dir="data")

def load_dataset():
    """
    Load the three files from the data directory and combine them.

    Returns:
        pd.DataFrame: The dataset as a pandas DataFrame.
    """

    # Load the three files
    for end in range(1, 4):
        file_path = f"data/data/full_dataset/goemotions_{end}.csv"
        df = pd.read_csv(file_path)
        if end == 1:
            dataset = df
        else:
            dataset = pd.concat([dataset, df], ignore_index=True)

    # remove unnecessary columns
    unneded = ["id", "author", "subreddit", "link_id", "parent_id", "created_utc", "rater_id", "example_very_unclear"]
    dataset = dataset.drop(columns=unneded)
    return dataset

if __name__ == "__main__":
    dataset = load_dataset()
    # we only want one emotion per text, 
    # so we filter out all rows where the sum of the one-hot encoded labels is not equal to 1
    dataset = dataset[dataset.drop(columns=["text"]).sum(axis=1) == 1]
    # drop duplicates
    dataset = dataset.drop_duplicates(subset=["text"])

    print(dataset.shape)
    print(dataset.head())