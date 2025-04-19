import kagglehub
import pandas as pd

from src.utils.setup_logger import setup_logger


def load_dataset_kaggle():
    """

    Download and extract the news summarization dataset from Kaggle, then load it into a pandas DataFrame.

    Parameters:
    - None

    Returns:
    - news_data (pd.DataFrame): A pandas DataFrame containing the news summarization dataset.
    """
    logger = setup_logger()
    path = kagglehub.dataset_download("sbhatti/news-summarization")
    logger.info(f"Dataset loaded and saved at {path}/data.csv")
    news_data = pd.read_csv(path + "/data.csv")

    news_data.loc[:, "Content"] = news_data["Content"].astype(str)
    news_data.loc[:, "Summary"] = news_data["Summary"].astype(str)
    return news_data
