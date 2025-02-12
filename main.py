"""
Main python file
"""

# pip install -r requirements.txt
# python -m spacy download en_core_web_sm

import random
import warnings
import zipfile

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
    plot_text_length_distribution,
    preprocess_articles,
    preprocess_summaries,
)
from src.features.tokenization import parallel_tokenize
from src.models.train_models import train_model
from src.models.transformer import Transformer
from src.set_up_config_device import (
    get_allowed_cpu_count,
    set_up_config_device,
    set_up_device,
)
from src.setup_logger import setup_logger

# Initialize logger
logger = setup_logger()

device = set_up_device()

cpu_count = get_allowed_cpu_count()

n_process = set_up_config_device(cpu_count)

"""
Kaggle dataset

Download and extract the news summarization dataset from Kaggle, then load it into a pandas DataFrame.
"""

# !kaggle datasets download -d sbhatti/news-summarization

with zipfile.ZipFile("news-summarization.zip", "r") as zip_ref:
    zip_ref.extractall("news-summarization")

news_data = pd.read_csv("news-summarization/data.csv")

# print(news_data.head())

N = random.randint(1, len(news_data))

# print(news_data["Content"][N])
# print()
# print(news_data["Summary"][N])


lengths_article = news_data["Content"].str.len()
lengths_article.describe()

news_data = news_data[
    (lengths_article >= lengths_article.quantile(0.10))
    & (lengths_article <= lengths_article.quantile(0.90))
]

plot_text_length_distribution(news_data, "Content")

lengths_summary = news_data["Summary"].str.len()
lengths_summary.describe()

news_data = news_data[
    (lengths_summary >= lengths_summary.quantile(0.10))
    & (lengths_summary <= lengths_summary.quantile(0.90))
]

news_data["Summary"].str.len().describe()

plot_text_length_distribution(news_data, "Summary")

# print(len(news_data))

news_data.loc[:, "Content"] = preprocess_articles(
    news_data["Content"].tolist(), n_process=n_process, batch_size=32
)
news_data.loc[:, "Summary"] = preprocess_summaries(
    news_data["Summary"].tolist(), n_process=n_process, batch_size=32
)

news_data.to_parquet("news_data_cleaned.parquet", index=False)

news_data = pd.read_parquet("news_data_cleaned.parquet")

"""
Tokenization

We shuffle the dataset, split it into training and testing sets with an 80-20 ratio, and print the sizes of both subsets.
"""

data_copy = news_data[:]
data_copy = news_data.sample(frac=1, random_state=42)

train_ratio = 0.8
train_size = int(train_ratio * len(data_copy))

# Slice the dataset
train_data = data_copy[:train_size]
test_data = data_copy[train_size:]

logger.info(f"Train size dataset length: {len(train_data)}")
logger.info(f"Test size dataset length: {len(test_data)}")

if __name__ == "__main__":
    texts_content = list(train_data["Content"])
    # print("Tokenizing Content...")
    tokenized_articles = parallel_tokenize(
        texts_content,
        tokenizer_name="bert-base-uncased",
        max_workers=n_process,
        chunk_size=2000,
        max_length=512,
    )
    # print("tokenized_articles.shape =", tokenized_articles.shape)
    torch.save(tokenized_articles, "tokenized_articles.pt")

if __name__ == "__main__":
    texts_summary = list(train_data["Summary"])
    # print("Tokenizing Summaries...")
    tokenized_summaries = parallel_tokenize(
        texts_summary,
        tokenizer_name="bert-base-uncased",
        max_workers=n_process,
        chunk_size=2000,
        max_length=129,
    )
    # print("tokenized_summaries.shape =", tokenized_summaries.shape)
    torch.save(tokenized_summaries, "tokenized_summaries.pt")

if __name__ == "__main__":
    texts_content = list(test_data["Content"])
    # print("Tokenizing Content...")
    tokenized_articles_test = parallel_tokenize(
        texts_content,
        tokenizer_name="bert-base-uncased",
        max_workers=n_process,
        chunk_size=2000,
        max_length=512,
    )
    # print("tokenized_articles.shape =", tokenized_articles_test.shape)
    torch.save(tokenized_articles_test, "tokenized_articles_test.pt")

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

batch_size = 32

dataset = TensorDataset(tokenized_articles, tokenized_summaries)
dataloader = DataLoader(
    dataset, batch_size=batch_size, num_workers=n_process, shuffle=True
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
