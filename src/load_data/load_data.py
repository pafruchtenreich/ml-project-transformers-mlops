import os.path

import pandas as pd

from src.features.functions_preprocessing import (
    preprocess_articles,
    preprocess_summaries,
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
