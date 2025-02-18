import time

import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

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
