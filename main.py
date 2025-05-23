### IMPORTS ###

# Standard library
import argparse
import os
import warnings

# Third-party libraries
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from transformers import BartTokenizer

# Feature engineering
from src.descriptive_statistics.descriptive_statistics import (
    descriptive_statistics,
    plot_text_length_distribution,
)
from src.data_processing.tokenization import tokenize_and_save_bart
from src.data_processing.create_dataloader import create_dataloader

# Evaluation and prediction
from src.evaluation.model_evaluation import evaluate_model

# Data loading
from src.load_data.load_data import download_model_weights, load_data

# Modeling and training
from src.models.transformer import Transformer
from src.prediction.generate_summaries_transformer import generate_summaries_transformer
from src.training.create_scheduler import create_scheduler
from src.training.train_models import (
    finetune_model_with_gridsearch_cv,
    train_model,
)


# Utils and setup
from src.utils.set_up_config_device import (
    get_allowed_cpu_count,
    set_up_config_device,
    set_up_device,
)
from src.utils.setup_logger import setup_logger

#MlFlow
import mlflow
import pandas as pd

tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")


### GLOBAL VARIABLES ###

DATA_FILENAME = "news_data_cleaned.parquet"
WEIGHTS_OUTPUT_DIR = "output/model_weights/"

BATCH_SIZE = 32
TEST_RATIO = 0.2
VAL_RATIO = 0.5
N_EPOCHS = 3
PARAMS_MODEL = {
    "pad_idx": tokenizer.pad_token_id,
    "hidden_size": 512,
    "n_head": 8,
    "max_len": 512,
    "dec_max_len": 150,
    "ffn_hidden": 2048,
    "n_layers": 6,
    "voc_size": len(tokenizer),
}


if __name__ == "__main__":
    ### ARGUMENT PARSING ###
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reload_data", default=False, type=bool, help="Reload data from scratch"
    )
    parser.add_argument(
        "--retrain_model",
        default=False,
        type=bool,
        help="Retrain model from default pretrained",
    )
    
    parser.add_argument(
    "--experiment_name", type=str, default="transformer_news_summarization", help="MLflow experiment name"
)
    args = parser.parse_args()
    reload_data = args.reload_data
    retrain_model = args.retrain_model

    ### LOGGER AND DEVICE SETUP ###
    logger = setup_logger()
    device = set_up_device()
    cpu_count = get_allowed_cpu_count()
    n_process = set_up_config_device(cpu_count)

    ### LOAD DATA ###
    logger.info("Loading data")
    news_data = load_data(
        reload_data=reload_data,
        n_process=n_process,
        batch_size=BATCH_SIZE,
        filename=DATA_FILENAME,
    )

    ### SETUP MLFLOW ###
    mlflow_server = os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(mlflow_server)
    mlflow.set_experiment(args.experiment_name)
    logger.info(f"Logging experiment to MLflow server: {mlflow_server}")

    ### DESCRIPTIVE STATISTICS ###
    descriptive_statistics(data=news_data, column_name="Content")
    descriptive_statistics(data=news_data, column_name="Summary")

    plot_text_length_distribution(data=news_data, column_name="Content")
    plot_text_length_distribution(data=news_data, column_name="Summary")

    ### TOKENIZATION AND DATA SPLIT ###
    logger.info("Tokenization and data split")
    train_data, temp_data = train_test_split(
        news_data, test_size=TEST_RATIO, random_state=42, shuffle=True
    )
    val_data, test_data = train_test_split(
        temp_data, test_size=VAL_RATIO, random_state=42
    )

    logger.info("Train size dataset length: %d", len(train_data))
    logger.info("Validation size dataset length: %d", len(val_data))
    logger.info("Test size dataset length: %d", len(test_data))

    if not os.path.exists(WEIGHTS_OUTPUT_DIR):
        os.mkdir(WEIGHTS_OUTPUT_DIR)

    if retrain_model:
        with mlflow.start_run(run_name="train_final_model"):
            logger.info("Retraining model")
            ### TOKENIZE AND SAVE TRAIN/VAL DATA ###
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
    
            ### TRAINING ###
            BEST_PARAMS = finetune_model_with_gridsearch_cv(
                Transformer,
                PARAMS_MODEL,
                tokenized_articles_train,
                tokenized_summaries_train,
                tokenizer,
                device,
                k_folds=3,
                num_epochs=3,
                batch_size=32,
                n_process=n_process,
                seed=42,
                grad_accum_steps=1,
                use_amp=True,
                early_stopping_patience=None,
            )
    
            modelTransformer = Transformer(**PARAMS_MODEL)
    
            dataloader_train = create_dataloader(
                tokenized_articles=tokenized_articles_train,
                tokenized_summaries=tokenized_summaries_train,
                batch_size=BATCH_SIZE,
                n_process=n_process,
                pad_value=tokenizer.pad_token_id,
            )
    
            dataloader_val = create_dataloader(
                tokenized_articles=tokenized_articles_val,
                tokenized_summaries=tokenized_summaries_val,
                batch_size=BATCH_SIZE,
                n_process=n_process,
                pad_value=tokenizer.pad_token_id,
            )
    
            optimizer = torch.optim.AdamW(
                modelTransformer.parameters(),
                lr=BEST_PARAMS["learning_rate"],
                betas=(0.9, 0.98),
                eps=1e-9,
                weight_decay=BEST_PARAMS["weight_decay"],
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
                    label_smoothing=BEST_PARAMS["label_smoothing"],
                ),
                "model_name": "Transformer",
                "device": device,
                "save_weights": True,
                "grad_accum_steps": 1,
                "use_amp": True,
                "early_stopping_patience": None,
            }
    
            train_model(**params_training)
            
            # Log final model
            mlflow.pytorch.log_model(modelTransformer, "final_transformer_model")

            # Log final training parameters
            mlflow.log_params(PARAMS_MODEL)
            mlflow.log_params(BEST_PARAMS)
    else:
        download_model_weights(WEIGHTS_OUTPUT_DIR)

    ### LOAD TRAINED MODEL ###
    modelTransformer = Transformer(**PARAMS_MODEL)

    modelTransformer.load_state_dict(
        torch.load(f"output/model_weights/transformer_weights_{N_EPOCHS}_epochs.pth")
    )
    modelTransformer.eval()

    ### PREDICTION AND EVALUATION ###
    tokenize_and_save_bart(
        data=test_data,
        column="Content",
        n_process=n_process,
        filename="tokenized_articles_test",
    )

    tokenized_articles_test = torch.load("output/token/tokenized_articles_test.pt")

    logger.info("Starting prediction")
    predictions_transformer = generate_summaries_transformer(
        model=modelTransformer,
        batch_size=BATCH_SIZE,
        tokenized_input=tokenized_articles_test,
        limit=None,
    )

    logger.info("Starting evaluation")
    metrics = evaluate_model(data=test_data, predictions=predictions_transformer)

    # Log evaluation metrics
    for metric, value in metrics.items():
        mlflow.log_metric(metric, value)




