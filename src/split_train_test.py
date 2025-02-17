import pandas as pd


def split_train_test(
    data: pd.DataFrame, ratio: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_copy = data[:]
    data_copy = data.sample(frac=1, random_state=42)

    train_size = int(ratio * len(data_copy))

    train_data = data_copy[:train_size]
    test_data = data_copy[train_size:]
    return train_data, test_data
