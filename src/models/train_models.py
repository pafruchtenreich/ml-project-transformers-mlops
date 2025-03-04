import copy
import time

import numpy as np
import torch
from sklearn.model_selection import KFold, ParameterGrid
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from src.create_dataloader import create_dataloader
from src.create_scheduler import create_scheduler
from src.setup_logger import setup_logger


def train_model(
    model,
    train_dataloader,
    val_dataloader,
    num_epochs,
    optimizer,
    scheduler,
    loss_fn,
    model_name,
    device,
    grad_accum_steps=1,
    use_amp=False,
    early_stopping_patience=None,
):
    """
    Generalized function to train a transformer model with validation.

    Parameters:
    - model (torch.nn.Module): The model to train.
    - train_dataloader (DataLoader): DataLoader for training data.
    - val_dataloader (DataLoader): DataLoader for validation data.
    - num_epochs (int): Number of epochs to train.
    - optimizer (torch.optim.Optimizer): Optimizer (e.g., AdamW).
    - scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
    - loss_fn (torch.nn.Module): Loss function (e.g., cross entropy or label smoothing).
    - model_name (str): Name to use when saving model checkpoints.
    - device (torch.device): CPU or GPU.
    - grad_accum_steps (int): Number of steps to accumulate gradients before update.
    - use_amp (bool): If True, train with automatic mixed precision.
    - early_stopping_patience (int or None): If set, stop if val loss fails to improve for this many epochs.

    Returns:
    - None
    """
    logger = setup_logger()
    model.to(device)

    scaler = GradScaler() if use_amp else None

    total_start_time = time.time()
    best_val_loss = float("inf")
    no_improvement_epochs = 0  # for early stopping

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        # ----------------
        # Training Phase
        # ----------------
        model.train()
        total_train_loss = 0.0

        # Zero gradients
        optimizer.zero_grad()

        for step, batch in enumerate(
            tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} [TRAIN]")
        ):
            input_batch, summary_batch = batch
            input_batch = input_batch.to(device)
            summary_batch = summary_batch.to(device)

            # ---------------------
            # Forward + Backward
            # ---------------------
            if use_amp:
                with autocast():
                    outputs = model(input_batch.long(), summary_batch[:, :-1])
                    shifted_target = summary_batch[:, 1:]
                    loss = loss_fn(
                        outputs.reshape(-1, outputs.shape[-1]),
                        shifted_target.reshape(-1),
                    )
            else:
                outputs = model(input_batch.long(), summary_batch[:, :-1])
                shifted_target = summary_batch[:, 1:]
                loss = loss_fn(
                    outputs.reshape(-1, outputs.shape[-1]), shifted_target.reshape(-1)
                )

            # Accumulate loss for reporting
            total_train_loss += loss.item()

            # Divide loss for gradient accumulation
            loss = loss / grad_accum_steps

            # Backward pass
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # ---------------------
            # Gradient Accumulation
            # ---------------------
            if (step + 1) % grad_accum_steps == 0:
                # Gradient clipping
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Update weights
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                # Update scheduler
                scheduler.step()

                # Reset gradients
                optimizer.zero_grad()

            # Periodic logging
            if step % 1000 == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    f"[TRAIN] Epoch: {epoch+1}, Step: {step}, "
                    f"Loss: {loss.item() * grad_accum_steps:.4f}, LR: {current_lr}"
                )

        # Calculate average training loss
        avg_train_loss = total_train_loss / len(train_dataloader)

        # -------------------
        # Validation Phase
        # -------------------
        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for step, batch in enumerate(
                tqdm(val_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} [VAL]")
            ):
                input_batch, summary_batch = batch
                input_batch = input_batch.to(device)
                summary_batch = summary_batch.to(device)

                if use_amp:
                    with autocast():
                        outputs = model(input_batch.long(), summary_batch[:, :-1])
                        shifted_target = summary_batch[:, 1:]
                        val_loss = loss_fn(
                            outputs.reshape(-1, outputs.shape[-1]),
                            shifted_target.reshape(-1),
                        )
                else:
                    outputs = model(input_batch.long(), summary_batch[:, :-1])
                    shifted_target = summary_batch[:, 1:]
                    val_loss = loss_fn(
                        outputs.reshape(-1, outputs.shape[-1]),
                        shifted_target.reshape(-1),
                    )

                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_dataloader)

        # Measure epoch time
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        logger.info(
            f"Epoch {epoch + 1}/{num_epochs} - "
            f"Train Loss: {avg_train_loss:.4f} - "
            f"Val Loss: {avg_val_loss:.4f} - "
            f"Time: {epoch_duration:.2f}s"
        )

        # -----------------
        # Checkpointing
        # -----------------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improvement_epochs = 0
            torch.save(
                model.state_dict(),
                f"model_weights/{model_name.lower()}_weights_{epoch + 1}_epochs.pth",
            )
            logger.info(
                f"Best model saved at epoch {epoch+1} with val loss {avg_val_loss:.4f}"
            )
        else:
            no_improvement_epochs += 1

        # -----------------
        # Early Stopping
        # -----------------
        if (
            early_stopping_patience is not None
            and no_improvement_epochs >= early_stopping_patience
        ):
            logger.info(
                f"No improvement for {no_improvement_epochs} consecutive epochs. "
                f"Early stopping triggered!"
            )
            break

    # Measure total training time
    total_end_time = time.time()
    total_training_time = total_end_time - total_start_time
    logger.info(f"Total training time: {total_training_time:.2f}s")

    return best_val_loss


def finetune_model_with_gridsearch_cv(
    model_class,
    base_params_model,
    train_articles,
    train_summaries,
    tokenizer,
    device,
    k_folds=3,
    num_epochs=10,
    batch_size=8,
    n_process=1,
    seed=42,
    grad_accum_steps=1,
    use_amp=False,
    early_stopping_patience=None,
):
    """
    Generalized function to perform grid search over a set of hyperparameters
    using k-fold cross-validation to find the best configuration for a Transformer model.

    Parameters:
    - model_class (callable): Callable that returns a Transformer model when called as model_class(**model_params).
    - base_params_model (dict): Base model parameters passed to `model_class`.
    - train_articles (list): List of tokenized input sequences (articles).
    - train_summaries (list): List of tokenized output sequences (summaries).
    - tokenizer: Tokenizer used to obtain pad_token_id and (optionally) for label_smoothing.
    - device (torch.device): CPU or GPU.
    - create_dataloader (callable): Function to build a PyTorch DataLoader given articles, summaries, etc.
    - create_scheduler (callable): Function to build the learning rate scheduler (e.g., linear warmup).
    - k_folds (int): Number of cross-validation folds.
    - num_epochs (int): Number of epochs to train in each fold.
    - batch_size (int): Training/validation batch size.
    - n_process (int): Number of workers for DataLoader.
    - seed (int): Random seed for reproducibility.
    - grad_accum_steps (int): Number of steps for gradient accumulation.
    - use_amp (bool): If True, train with automatic mixed precision.
    - early_stopping_patience (int or None): Stop if val loss fails to improve for this many epochs.

    Returns:
    - best_params (dict): The best hyperparameter combination found (lowest average validation loss).
    """

    # ----------------
    # Setup and config
    # ----------------
    logger = setup_logger()
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    # Define a grid of hyperparameters we want to search
    param_grid = {
        "learning_rate": [5e-5, 1e-4, 3e-4],
        "weight_decay": [0.0, 1e-2],
        "label_smoothing": [0.0, 0.1],
    }

    # Initialize variables for tracking
    dataset_indices = np.arange(len(train_articles))
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    all_combos = list(ParameterGrid(param_grid))
    best_params = None
    best_loss = float("inf")

    # ----------------
    # Grid Search Loop
    # ----------------
    for combo_idx, combo in enumerate(all_combos):
        logger.info(f"=== Combination {combo_idx + 1}/{len(all_combos)}: {combo} ===")
        fold_losses = []

        # -----------------------------
        # Cross-validation (k-fold) Loop
        # -----------------------------
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(dataset_indices)):
            logger.info(f"  -- Fold {fold_idx + 1}/{k_folds} --")

            # Subset the data for this fold
            train_articles_fold = [train_articles[i] for i in train_idx]
            train_summaries_fold = [train_summaries[i] for i in train_idx]
            val_articles_fold = [train_articles[i] for i in val_idx]
            val_summaries_fold = [train_summaries[i] for i in val_idx]

            # Create DataLoaders
            train_dataloader_fold = create_dataloader(
                tokenized_articles=train_articles_fold,
                tokenized_summaries=train_summaries_fold,
                batch_size=batch_size,
                n_process=n_process,
            )
            val_dataloader_fold = create_dataloader(
                tokenized_articles=val_articles_fold,
                tokenized_summaries=val_summaries_fold,
                batch_size=batch_size,
                n_process=n_process,
            )

            # Build a fresh model
            model_params = copy.deepcopy(base_params_model)
            model = model_class(**model_params)

            # Extract hyperparams from combo
            learning_rate = combo["learning_rate"]
            weight_decay = combo["weight_decay"]
            label_smoothing = combo["label_smoothing"]

            # Optimizer
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                betas=(0.9, 0.98),
                eps=1e-9,
                weight_decay=weight_decay,
            )

            # Scheduler
            scheduler = create_scheduler(train_dataloader_fold, optimizer, num_epochs)

            # Loss function
            loss_fn = torch.nn.CrossEntropyLoss(
                ignore_index=tokenizer.pad_token_id, label_smoothing=label_smoothing
            )

            # Train model for this fold
            best_val_loss_fold = train_model(
                model=model,
                train_dataloader=train_dataloader_fold,
                val_dataloader=val_dataloader_fold,
                num_epochs=num_epochs,
                optimizer=optimizer,
                scheduler=scheduler,
                loss_fn=loss_fn,
                model_name="Transformer",
                device=device,
                grad_accum_steps=grad_accum_steps,
                use_amp=use_amp,
                early_stopping_patience=early_stopping_patience,
            )

            # Keep track of final/best val loss for this fold
            fold_losses.append(best_val_loss_fold)

        # ----------------
        # Evaluate results
        # ----------------
        avg_val_loss = np.mean(fold_losses)
        logger.info(
            f"  >> Average val loss for combo {combo_idx + 1}: {avg_val_loss:.4f}"
        )

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_params = combo

    # ------------------
    # Final best results
    # ------------------
    logger.info("===== GRID SEARCH RESULTS =====")
    logger.info(f"Best hyperparameters: {best_params}")
    logger.info(f"Best average validation loss: {best_loss:.4f}")

    return best_params
