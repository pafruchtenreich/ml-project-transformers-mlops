import os.path

import pandas as pd
import requests

from src.features.functions_preprocessing import (
    preprocess_articles,
    preprocess_summaries,
    remove_outlier,
)
from src.load_data.load_dataset_kaggle import load_dataset_kaggle
from src.setup_logger import setup_logger


def load_data(reload_data, n_process, batch_size, filename):
    """
    Load the dataset and preprocess the articles and summaries. Do it from scratch if reload_data is True.

    Parameters:
    - reload_data: bool, whether to reload the data from scratch
    - n_process: int, number of processes to use for preprocessing
    - batch_size: int, batch size for preprocessing
    - filename: str, filename to save/find the preprocessed data

    Returns:
    - news_data: pd.DataFrame, preprocessed articles and summaries
    """
    logger = setup_logger()
    if reload_data:
        news_data = load_dataset_kaggle()

        news_data = remove_outlier(
            news_data, col="Content", lower_percent=10, upper_percent=90
        )
        news_data = remove_outlier(
            news_data, col="Summary", lower_percent=10, upper_percent=90
        )

        news_data = news_data.sample(n=500, random_state=42)  # To make training faster

        news_data.loc[:, "Content"] = preprocess_articles(
            news_data["Content"].tolist(), n_process=n_process, batch_size=batch_size
        )
        news_data.loc[:, "Summary"] = preprocess_summaries(
            news_data["Summary"].tolist(), n_process=n_process, batch_size=batch_size
        )

        logger.info("Articles and summaries have been preprocessed")

        news_data.to_parquet(filename, index=False)
        logger.info(
            f"Preprocessed articles and summaries have been saved in {filename}"
        )
        news_data = pd.read_parquet(filename)
    else:
        if not os.path.isfile(filename):
            logger.error(
                f"{filename} not found. Change DATA_FILENAME or set reload_data=True"
            )
            raise FileNotFoundError(
                f"{filename} not found. Change DATA_FILENAME or set reload_data=True"
            )
        else:
            try:
                news_data = pd.read_parquet(filename)
            except Exception as e:
                logger.error(f"Error while loading {filename}")
                logger.error(e)
                raise e
    logger.info(f"Preprocessed articles and summaries have been loaded from {filename}")
    return news_data


def download_model_weights(output_dir):
    """
    Downloads a the models weights and saves them to the specified output directory.

    Parameters:
    - output_dir : The path to the directory where the downloaded files will be saved.
        The directory will be created if it does not already exist.
    """
    urls = [
        f"https://minio.lab.sspcloud.fr/gamer35/public/transformer_article_weights/transformer_weights_{i+1}_epochs.pth"
        for i in range(3)
    ]

    os.makedirs(output_dir, exist_ok=True)

    for url in urls:
        filename = os.path.basename(url)
        dest_path = os.path.join(output_dir, filename)

        # Only downloads if the weight file is not already in the folder
        if os.path.exists(dest_path):
            print(f"Skipping {filename} (already exists)")
            continue

        print(f"Downloading {filename}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise error if the download fails

        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print(f"Saved to {dest_path}")
