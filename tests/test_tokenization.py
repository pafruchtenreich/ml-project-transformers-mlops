import torch
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_processing.tokenization import tokenize_and_save_bart

def test_tokenize_and_save_bart_basic(tmp_path):
    df = pd.DataFrame({
        "Content": ["This is a short sentence.", "Another sentence goes here."]
    })

    filename = "test_tokenized"
    output_path = tmp_path / "token"
    os.makedirs(output_path, exist_ok=True)

    # Run function
    tokenize_and_save_bart(
        data=df,
        column="Content",
        n_process=1,
        filename=filename,
    )

    saved_file = os.path.join("output/token", f"{filename}.pt")
    assert os.path.exists(saved_file)

    tokenized = torch.load(saved_file)
    assert tokenized.shape[0] == 2
    assert tokenized.shape[1] == 512  # max_length for 'Content'

def test_tokenize_and_save_bart_empty_input(tmp_path):
    df = pd.DataFrame({"Content": []})

    filename = "test_empty_tokenized"

    tokenize_and_save_bart(
        data=df,
        column="Content",
        n_process=1,
        filename=filename,
    )

    saved_file = os.path.join("output/token", f"{filename}.pt")
    tokenized = torch.load(saved_file)
    assert tokenized.shape[0] == 0


def test_tokenize_and_save_bart_multi_process(tmp_path):
    df = pd.DataFrame({
        "Content": ["Sentence number " + str(i) for i in range(100)]
    })

    filename = "test_multi_proc"

    tokenize_and_save_bart(
        data=df,
        column="Content",
        n_process=4,
        filename=filename,
    )

    saved_file = os.path.join("output/token", f"{filename}.pt")
    tokenized = torch.load(saved_file)
    assert tokenized.shape[0] == 100
