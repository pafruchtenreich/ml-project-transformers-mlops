# Standard library
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial

# Third-party libraries
import pandas as pd
import torch
from transformers import BartTokenizer

# Internal modules
from src.utils.setup_logger import setup_logger


def tokenize_chunk_bart(texts_chunk, tokenizer_name: str, max_length: int = 512):
    """
    Tokenize a chunk of texts using a BART tokenizer.
    Instantiates inside each process to avoid pickle issues.
    """
    tokenizer = BartTokenizer.from_pretrained(tokenizer_name)
    out = tokenizer(
        texts_chunk,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    return out["input_ids"]


def parallel_tokenize_bart(
    texts,
    tokenizer_name="facebook/bart-large-cnn",
    max_workers=1,
    chunk_size=2000,
    max_length=512,
):
    # Split data into chunks
    chunks = [texts[i : i + chunk_size] for i in range(0, len(texts), chunk_size)]

    # Partial function for parallel processing
    tokenize_fn = partial(
        tokenize_chunk_bart, tokenizer_name=tokenizer_name, max_length=max_length
    )

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for tokenized_ids in executor.map(tokenize_fn, chunks):
            results.append(tokenized_ids)

    tokenized_tensor = torch.cat(results, dim=0)
    return tokenized_tensor


def tokenize_and_save_bart(
    data: pd.DataFrame,
    column: str,
    n_process: int,
    filename: str,
):
    logger = setup_logger()

    output_dir = "output/token"
    os.makedirs(output_dir, exist_ok=True)

    filepath = os.path.join(output_dir, f"{filename}.pt")

    texts = list(data[column])
    max_length = 512 if column == "Content" else 128  # for shorter summaries

    tokenized_data = parallel_tokenize_bart(
        texts,
        tokenizer_name="facebook/bart-large-cnn",
        max_workers=n_process,
        chunk_size=2000,
        max_length=max_length,
    )
    logger.info(f"{filepath}.shape = {tokenized_data.shape}")
    torch.save(tokenized_data, filepath)
