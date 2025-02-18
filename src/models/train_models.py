import time

import torch
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
):
    """
    Generalized function to train a transformer model with validation.

    Parameters:
    - model (torch.nn.Module): The model to train.
    - train_dataloader (DataLoader): DataLoader for the training data.
    - val_dataloader (DataLoader): DataLoader for the validation data.
    - num_epochs (int): Number of epochs to train for.
    - optimizer (torch.optim.Optimizer): Optimizer for training.
    - loss_fn (torch.nn.Module): Loss function.
    - model_name (str): Name used to save model checkpoints.
    - device (torch.device): Device to train on (CPU or GPU).

    Returns:
    - None
    """
    logger = setup_logger()
    model.to(device)
    total_start_time = time.time()
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        # ----------------
        # Training Phase
        # ----------------
        model.train()
        total_train_loss = 0

        for step, batch in enumerate(
            tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} [TRAIN]")
        ):
            input_batch, summary_batch = batch
            input_batch = input_batch.to(device)
            summary_batch = summary_batch.to(device)

            # Clear gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(input_batch.long(), summary_batch[:, :-1])
            shifted_target = summary_batch[:, 1:]  # Shift target for loss
            loss = loss_fn(
                outputs.reshape(-1, outputs.shape[-1]), shifted_target.reshape(-1)
            )

            total_train_loss += loss.item()

            # Backward pass
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update weights
            optimizer.step()

            # Update learning rate
            scheduler.step()

            if step % 1000 == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    f"[TRAIN] Epoch: {epoch+1}, Step: {step}, Loss: {loss.item():.4f}, LR: {current_lr}"
                )

        # Calculate average training loss for the epoch
        avg_train_loss = total_train_loss / len(train_dataloader)

        # -------------------
        # Validation Phase
        # -------------------
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for step, batch in enumerate(
                tqdm(val_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} [VAL]")
            ):
                input_batch, summary_batch = batch
                input_batch = input_batch.to(device)
                summary_batch = summary_batch.to(device)

                outputs = model(input_batch.long(), summary_batch[:, :-1])
                shifted_target = summary_batch[:, 1:]
                val_loss = loss_fn(
                    outputs.reshape(-1, outputs.shape[-1]), shifted_target.reshape(-1)
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

        # Save model after each epoch
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                model.state_dict(),
                f"model_weights/{model_name.lower()}_weights_{epoch + 1}_epochs.pth",
            )
            logger.info(
                f"Best model saved at epoch {epoch+1} with val loss {avg_val_loss:.4f}"
            )

    # Measure total training time
    total_end_time = time.time()
    total_training_time = total_end_time - total_start_time
    logger.info(f"Total training time: {total_training_time:.2f}s")
