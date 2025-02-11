import time
import torch
from tqdm import tqdm


def train_model(
    model,
    dataloader,
    num_epochs,
    optimizer,
    loss_fn,
    # obsolete
    model_name,
    device,
    teacher_forcing_ratio=None,
):
    """
    Generalized function to train a model (Seq2Seq, Transformer, or BERT).

    Parameters:
    - model (torch.nn.Module): The model to train.
    - dataloader (DataLoader): DataLoader for the training data.
    - num_epochs (int): Number of epochs to train for.
    - optimizer (torch.optim.Optimizer): Optimizer for training.
    - loss_fn (torch.nn.Module): Loss function.
    - device (torch.device): Device to train on (CPU or GPU).
    - teacher_forcing_ratio (float or None): Ratio for teacher forcing (only for Seq2Seq).

    Returns:
    - None
    """
    model.to(device)
    total_start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        total_loss = 0

        for step, batch in enumerate(
            tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
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

            total_loss += loss.item()

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()

            if step % 1000 == 0:
                print(f"Epoch: {epoch+1}, Step: {step}, Loss: {loss.item():.4f}")

        # Calculate average loss for the epoch
        avg_loss = total_loss / len(dataloader)

        # Measure epoch time
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        print(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Average Loss: {avg_loss:.4f} - "
            f"Time: {epoch_duration:.2f}s"
        )
        torch.save(
            model.state_dict(),
            f"model_weights/{model_name.lower()}_weights_{epoch+1}_epochs.pth",
        )

    # Measure total training time
    total_end_time = time.time()
    total_training_time = total_end_time - total_start_time
    print(f"Total training time: {total_training_time:.2f}s")
