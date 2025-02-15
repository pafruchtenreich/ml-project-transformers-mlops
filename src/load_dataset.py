import kagglehub
import pandas as pd


def load_dataset():
    """

    Download and extract the news summarization dataset from Kaggle, then load it into a pandas DataFrame.

    Parameters:
    - None

    Returns:
    - news_data (pd.DataFrame): A pandas DataFrame containing the news summarization dataset.
    """
    path = kagglehub.dataset_download("sbhatti/news-summarization")
    news_data = pd.read_csv(path + "/data.csv")
    return news_data
