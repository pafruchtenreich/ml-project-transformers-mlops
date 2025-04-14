from transformers import get_linear_schedule_with_warmup


def create_scheduler(dataloader, optimizer, n_epochs):
    """
    Creates a scheduler for the optimizer.

    Parameters:
    - dataloader: DataLoader that contains training data
    - optimizer: Optimizer used to update the model's weights
    - n_epochs: Number of epochs to train the model

    Returns:
    - scheduler: Scheduler for the optimizer
    """
    num_train_steps = len(dataloader) * n_epochs
    warmup_steps = int(0.1 * num_train_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_steps
    )
    return scheduler
