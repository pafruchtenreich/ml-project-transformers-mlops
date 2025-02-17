from torch.utils.data import DataLoader, TensorDataset


def create_dataloader(tokenized_articles, tokenized_summaries, batch_size, n_process):
    """
    Create a DataLoader object from the tokenized articles and summaries.

    Parameters:
    - tokenized_articles (torch.Tensor): A tensor containing the tokenized articles.
    - tokenized_summaries (torch.Tensor): A tensor containing the tokenized summaries.
    - batch_size (int): The batch size.
    - n_process (int): The number of processes to use.

    Returns:
    - dataloader (DataLoader): A DataLoader object containing the tokenized articles and summaries.
    """
    dataset = TensorDataset(tokenized_articles, tokenized_summaries)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=n_process, shuffle=True
    )
    return dataloader
