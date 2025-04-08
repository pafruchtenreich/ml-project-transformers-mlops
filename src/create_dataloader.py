import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset


def create_dataloader(
    tokenized_articles, tokenized_summaries, batch_size, n_process, pad_value=0
):
    """
    Creates a DataLoader object from tokenized articles and summaries.
    Supports both padded tensors and raw token ID lists.

    Parameters:
    - tokenized_articles (list or torch.Tensor): Tokenized input sequences.
    - tokenized_summaries (list or torch.Tensor): Tokenized output sequences.
    - batch_size (int): The batch size.
    - n_process (int): Number of workers.
    - pad_value (int): Padding value to use if input is a list.

    Returns:
    - DataLoader: Batched and shuffled dataset loader.
    """

    def ensure_tensor(x):
        return (
            x.clone().detach()
            if isinstance(x, torch.Tensor)
            else torch.tensor(x, dtype=torch.long)
        )

    if isinstance(tokenized_articles, list):
        tokenized_articles = pad_sequence(
            [ensure_tensor(x) for x in tokenized_articles],
            batch_first=True,
            padding_value=pad_value,
        )

    if isinstance(tokenized_summaries, list):
        tokenized_summaries = pad_sequence(
            [ensure_tensor(x) for x in tokenized_summaries],
            batch_first=True,
            padding_value=pad_value,
        )

    dataset = TensorDataset(tokenized_articles, tokenized_summaries)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=n_process, shuffle=True
    )
    return dataloader
