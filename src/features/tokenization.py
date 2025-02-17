from concurrent.futures import ProcessPoolExecutor
from functools import partial

import pandas as pd
import torch
from transformers import BertTokenizer

from src.setup_logger import setup_logger


def tokenize_chunk(texts_chunk, tokenizer_name: str, max_length: int = 512):
    """
    Tokenize a chunk of texts using the provided tokenizer name.
    Instantiates the tokenizer in each process to avoid pickle issues.
    """
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    out = tokenizer(
        texts_chunk,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    return out["input_ids"]


def parallel_tokenize(
    texts,
    tokenizer_name="bert-base-uncased",
    max_workers=1,
    chunk_size=2000,
    max_length=512,
):
    """
    Splits `texts` into chunks, then uses multiple processes to tokenize them in parallel.

    Args:
        texts (List[str]): The texts to tokenize.
        tokenizer_name (str): Name/path of the BERT tokenizer to use.
        chunk_size (int): Number of texts in each chunk.
        max_workers (int): Number of processes to spawn.
        max_length (int): Max sequence length for tokenization.

    Returns:
        torch.Tensor: Concatenated tensor of input_ids from all chunks.
    """
    # Split the data into chunks
    chunks = [texts[i : i + chunk_size] for i in range(0, len(texts), chunk_size)]

    # Prepare a partial function with tokenizer args
    tokenize_fn = partial(
        tokenize_chunk, tokenizer_name=tokenizer_name, max_length=max_length
    )

    # List for storing tokenized results from each process
    results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for tokenized_ids in executor.map(tokenize_fn, chunks):
            results.append(tokenized_ids)

    # Concatenate all chunk results
    tokenized_tensor = torch.cat(results, dim=0)
    return tokenized_tensor


def tokenize_and_save(
    data: pd.DataFrame,
    column: str,
    n_process: int,
    filename: str,
):
    """
    Tokenize the data and save it to a file.

    Parameters:
    - data: The dataframe that contains the data to tokenize.
    - column: The column name of the data to tokenize (either Summary or Content).
    - n_process: Number of processes to use.
    - filename: Name of the file to save the tokenized data to (without file extension).

    Returns:
    - None
    """
    logger = setup_logger()
    texts = list(data[column])
    max_length = 512 if column == "Content" else 129
    tokenized_data = parallel_tokenize(
        texts,
        tokenizer_name="bert-base-uncased",
        max_workers=n_process,
        chunk_size=2000,
        max_length=max_length,
    )
    # print("tokenized_summaries.shape =", tokenized_summaries.shape)
    logger.info(f"{filename}.shape = {tokenized_data.shape}")
    torch.save(tokenized_data, f"{filename}.pt")
