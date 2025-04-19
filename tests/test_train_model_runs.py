# tests/test_train_model.py
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

import sys 
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.training.train_models import train_model
from src.models.transformer import Transformer

def get_dummy_dataloader(num_samples=8, seq_len=10, vocab_size=1000, pad_val=1, batch_size=4):
    X = torch.randint(2, vocab_size, (num_samples, seq_len))  # input_ids
    Y = torch.randint(2, vocab_size, (num_samples, seq_len))  # target_ids
    dataset = TensorDataset(X, Y)
    return DataLoader(dataset, batch_size=batch_size)

def test_train_model_runs():
    # Model params (small model)
    PARAMS_MODEL = {
        "pad_idx": 1,
        "hidden_size": 64,
        "n_head": 2,
        "max_len": 10,
        "dec_max_len": 10,
        "ffn_hidden": 128,
        "n_layers": 2,
        "voc_size": 1000,
    }

    model = Transformer(**PARAMS_MODEL)

    dataloader = get_dummy_dataloader()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    params_training = {
        "model": model,
        "train_dataloader": dataloader,
        "val_dataloader": dataloader,  # use same dummy data
        "num_epochs": 1,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "loss_fn": nn.CrossEntropyLoss(ignore_index=1),
        "model_name": "Transformer",
        "device": "cpu",  # run on CPU for test
        "save_weights": False,
        "grad_accum_steps": 1,
        "use_amp": False,
        "early_stopping_patience": None,
    }

    train_model(**params_training)
