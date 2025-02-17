"""
Main python file
"""

# pip install -r requirements.txt
# python -m spacy download en_core_web_sm

import warnings

import evaluate
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer

from src.evaluation.model_evaluation import (
    generate_summaries_transformer,
)
from src.features.functions_preprocessing import (
    drop_short_long_articles,
    drop_short_long_summaries,
    preprocess_articles,
    preprocess_summaries,
)
from src.features.tokenization import tokenize_and_save
from src.load_dataset import load_dataset
from src.models.train_models import train_model
from src.models.transformer import Transformer
from src.set_up_config_device import (
    get_allowed_cpu_count,
    set_up_config_device,
    set_up_device,
)
from src.setup_logger import setup_logger
from src.split_train_test import split_train_test

BATCH_SIZE = 32
SPLIT_RATIO = 0.8

# Initialize logger
logger = setup_logger()

device = set_up_device()

cpu_count = get_allowed_cpu_count()

n_process = set_up_config_device(cpu_count)

# Load dataset
news_data = load_dataset()

# Drop the rows with too long/short (based on 10th and 90th percentiles) articles or summaries
news_data = drop_short_long_articles(news_data)
news_data = drop_short_long_summaries(news_data)

# Preprocess the articles and summaries (lowercasing + tokens)
news_data.loc[:, "Content"] = preprocess_articles(
    news_data["Content"].tolist(), n_process=n_process, batch_size=BATCH_SIZE
)
news_data.loc[:, "Summary"] = preprocess_summaries(
    news_data["Summary"].tolist(), n_process=n_process, batch_size=BATCH_SIZE
)

logger.info("Articles and summaries have been preprocessed")

news_data.to_parquet("news_data_cleaned.parquet", index=False)
logger.info(
    "Preprocessed articles and summaries have been saved in news_data_cleaned.parquet"
)

news_data = pd.read_parquet("news_data_cleaned.parquet")

train_data, test_data = split_train_test(news_data, ratio=SPLIT_RATIO)
logger.info(f"Train size dataset length: {len(train_data)}")
logger.info(f"Test size dataset length: {len(test_data)}")

if __name__ == "__main__":
    tokenize_and_save(
        data=train_data,
        column="Content",
        n_process=n_process,
        filename="tokenized_articles",
    )
    tokenize_and_save(
        data=train_data,
        column="Summary",
        n_process=n_process,
        filename="tokenized_summaries",
    )
    tokenize_and_save(
        data=test_data,
        column="Content",
        n_process=n_process,
        filename="tokenized_articles_test",
    )

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    tokenized_articles = torch.load("tokenized_articles.pt")
    tokenized_summaries = torch.load("tokenized_summaries.pt")
    tokenized_articles_test = torch.load("tokenized_articles_test.pt")

article_ids = tokenized_articles.long()
summary_ids = tokenized_summaries.long()

"""
Transformer
"""

dataset = TensorDataset(tokenized_articles, tokenized_summaries)
dataloader = DataLoader(
    dataset, batch_size=BATCH_SIZE, num_workers=n_process, shuffle=True
)

modelTransformer = Transformer(
    pad_idx=0,
    voc_size=BertTokenizer.from_pretrained("bert-base-uncased").vocab_size,
    hidden_size=128,
    n_head=8,
    max_len=512,
    dec_max_len=512,
    ffn_hidden=128,
    n_layers=3,
)

train_model(
    model=modelTransformer,
    dataloader=dataloader,
    num_epochs=25,
    optimizer=torch.optim.Adam(modelTransformer.parameters(), lr=2e-4),
    loss_fn=nn.CrossEntropyLoss(
        ignore_index=BertTokenizer.from_pretrained("bert-base-uncased").pad_token_id
    ),
    model_name="Transformer",
    device=device,
)

modelTransformer = Transformer(
    pad_idx=0,
    voc_size=BertTokenizer.from_pretrained("bert-base-uncased").vocab_size,
    hidden_size=128,
    n_head=8,
    max_len=512,
    dec_max_len=128,
    ffn_hidden=128,
    n_layers=3,
)
modelTransformer.load_state_dict(
    torch.load("output/model_weights/transformer_weights_25_epochs.pth")
)
modelTransformer.eval()

"""
Evaluation
"""

rouge = evaluate.load("rouge")

predictions_transformer = generate_summaries_transformer(
    model=modelTransformer,
    batch_size=32,
    tokenized_input=tokenized_articles_test,
    limit=None,
)

test_data.loc[:, "predictions_transformer"] = predictions_transformer

reference_summaries = list(test_data["Summary"])
results = rouge.compute(
    predictions=predictions_transformer, references=reference_summaries
)
logger.info(f"ROUGE metrics: {results}")
