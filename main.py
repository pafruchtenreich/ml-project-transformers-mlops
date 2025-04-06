"""
Main python file
"""

# pip install -r requirements.txt
# python -m spacy download en_core_web_sm


### IMPORTS ###

import argparse
import os
import warnings

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from transformers import BartTokenizer

from src.create_dataloader import create_dataloader
from src.create_scheduler import create_scheduler
from src.features.functions_preprocessing import (
    descriptive_statistics,
    plot_text_length_distribution,
)
from src.features.tokenization import tokenize_and_save_bart
from src.load_data.load_data import load_data
from src.models.train_models import train_model
from src.models.transformer import Transformer
from src.set_up_config_device import (
    get_allowed_cpu_count,
    set_up_config_device,
    set_up_device,
)
from src.setup_logger import setup_logger

### GLOBAL VARIABLES ###

DATA_FILENAME = "news_data_cleaned.parquet"
BATCH_SIZE = 32
TEST_RATIO = 0.2
VAL_RATIO = 0.5
N_EPOCHS = 3
LEARNING_RATE = 5e-6
PARAMS_MODEL = {
    "pad_idx": 0,
    "hidden_size": 512,
    "n_head": 8,
    "max_len": 512,
    "dec_max_len": 150,
    "ffn_hidden": 2048,
    "n_layers": 6,
}

if __name__ == "__main__":
    # Retrieve arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reload_data", default=False, type=bool, help="Reload data from scracth"
    )
    parser.add_argument(
        "--retrain_model",
        default=False,
        type=bool,
        help="Retrain model from default pretrained",
    )
    args = parser.parse_args()
    reload_data = args.reload_data
    retrain_model = args.retrain_model

    # Initialize logger
    logger = setup_logger()

    device = set_up_device()

    cpu_count = get_allowed_cpu_count()

    n_process = set_up_config_device(cpu_count)

    # Load dataset
    news_data = load_data(
        reload_data=reload_data,
        n_process=n_process,
        batch_size=BATCH_SIZE,
        filename=DATA_FILENAME,
    )

    # Descriptive statistics
    descriptive_statistics(data=news_data, column_name="Content")
    descriptive_statistics(data=news_data, column_name="Summary")

    plot_text_length_distribution(data=news_data, column_name="Content")
    plot_text_length_distribution(data=news_data, column_name="Summary")

    """
    Tokenization

    We shuffle the dataset, split it into training and testing sets with an 80-20 ratio,
    and print the sizes of both subsets.
    """

    train_data, temp_data = train_test_split(
        news_data, test_size=TEST_RATIO, random_state=42, shuffle=True
    )
    val_data, test_data = train_test_split(
        temp_data, test_size=VAL_RATIO, random_state=42
    )

    logger.info(f"Train size dataset length: {len(train_data)}")
    logger.info(f"Validation size dataset length: {len(val_data)}")
    logger.info(f"Test size dataset length: {len(test_data)}")

    if retrain_model:
        files = {
            "tokenized_articles_train": (train_data, "Content"),
            "tokenized_summaries_train": (train_data, "Summary"),
            "tokenized_articles_val": (val_data, "Content"),
            "tokenized_summaries_val": (val_data, "Summary"),
        }

        for filename, (df, column) in files.items():
            path = os.path.join("output", "token", f"{filename}.pt")
            if not os.path.exists(path):
                tokenize_and_save_bart(
                    data=df,
                    column=column,
                    n_process=n_process,
                    filename=filename,
                )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tokenized_articles_train = torch.load(
                "output/token/tokenized_articles_train.pt"
            )
            tokenized_summaries_train = torch.load(
                "output/token/tokenized_summaries_train.pt"
            )
            tokenized_articles_val = torch.load(
                "output/token/tokenized_articles_val.pt"
            )
            tokenized_summaries_val = torch.load(
                "output/token/tokenized_summaries_val.pt"
            )

        """
        Transformer
        """

        tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

        PARAMS_MODEL["voc_size"] = len(tokenizer)
        PARAMS_MODEL["pad_idx"] = tokenizer.pad_token_id

        modelTransformer = Transformer(**PARAMS_MODEL)

        dataloader_train = create_dataloader(
            tokenized_articles=tokenized_articles_train,
            tokenized_summaries=tokenized_summaries_train,
            batch_size=BATCH_SIZE,
            n_process=n_process,
        )

        dataloader_val = create_dataloader(
            tokenized_articles=tokenized_articles_val,
            tokenized_summaries=tokenized_summaries_val,
            batch_size=BATCH_SIZE,
            n_process=n_process,
        )

        optimizer = torch.optim.AdamW(
            modelTransformer.parameters(),
            lr=LEARNING_RATE,
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=1e-2,
        )

        scheduler = create_scheduler(
            dataloader=dataloader_train,
            optimizer=optimizer,
            n_epochs=N_EPOCHS,
        )

        params_training = {
            "model": modelTransformer,
            "train_dataloader": dataloader_train,
            "val_dataloader": dataloader_val,
            "num_epochs": N_EPOCHS,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "loss_fn": nn.CrossEntropyLoss(
                ignore_index=tokenizer.pad_token_id,
                label_smoothing=0.0,
            ),
            "model_name": "Transformer",
            "device": device,
            "save_weights": True,
            "grad_accum_steps": 1,
            "use_amp": True,
            "early_stopping_patience": None,
        }

        train_model(**params_training)

    modelTransformer = Transformer(**PARAMS_MODEL)

"""         modelTransformer.load_state_dict(
        torch.load("output/model_weights/transformer_weights_25_epochs.pth")
        )
        modelTransformer.eval()

        """
"""         Prediction and evaluation
 """ """

        tokenize_and_save_bart(
        data=test_data,
        column="Content",
        n_process=n_process,
        filename="tokenized_articles_test",
        )

        tokenized_articles_test = torch.load("tokenized_articles_test.pt")

        predictions_transformer = generate_summaries_transformer(
        model=modelTransformer,
        batch_size=BATCH_SIZE,
        tokenized_input=tokenized_articles_test,
        limit=None,
        )

        evaluate_model(data=test_data, predictions=predictions_transformer) """
