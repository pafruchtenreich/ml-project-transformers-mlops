import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_processing.create_dataloader import create_dataloader


def test_create_dataloader_shapes():
    # Create dummy tokenized inputs and outputs (as lists of lists of token IDs)
    dummy_articles = [list(range(5)), list(range(7)), list(range(6))]  # Varying lengths
    dummy_summaries = [list(range(4)), list(range(6)), list(range(5))]

    batch_size = 2
    n_process = 0

    # Create the DataLoader
    dataloader = create_dataloader(
        tokenized_articles=dummy_articles,
        tokenized_summaries=dummy_summaries,
        batch_size=batch_size,
        n_process=n_process,
        pad_value=0
    )

    # Fetch one batch
    batch = next(iter(dataloader))
    inputs, summaries = batch

    # Check shapes
    assert inputs.shape[0] == batch_size, "Batch size mismatch in inputs"
    assert summaries.shape[0] == batch_size, "Batch size mismatch in summaries"
    assert inputs.ndim == 2, "Inputs should be 2D (batch_size, seq_len)"
    assert summaries.ndim == 2, "Summaries should be 2D (batch_size, seq_len)"

    # Check that padding worked: all sequences in a batch should be the same length
    assert inputs.shape[1] == max(len(seq) for seq in dummy_articles), "Input sequence padding failed"
    assert summaries.shape[1] == max(len(seq) for seq in dummy_summaries), "Summary sequence padding failed"
