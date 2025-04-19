# Standard library
import time

# Third-party libraries
import numpy as np
import torch
import mlflow
from sklearn.model_selection import KFold, ParameterGrid
from tqdm import tqdm

# Internal modules
from src.data_processing.create_dataloader import create_dataloader
from src.training.create_scheduler import create_scheduler
from src.utils.setup_logger import setup_logger


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
    save_weights=True,
    grad_accum_steps=1,
    use_amp=False,
    early_stopping_patience=None,
):
    logger = setup_logger()
    model.to(device)
    scaler = torch.amp.GradScaler(device) if use_amp else None

    total_start_time = time.time()
    best_val_loss = float("inf")
    no_improvement_epochs = 0

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        # Training phase
        model.train()
        total_train_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(
            tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} [TRAIN]")
        ):
            input_batch, summary_batch = batch
            input_batch = input_batch.to(device)
            summary_batch = summary_batch.to(device)

            if use_amp:
                with torch.amp.autocast(device):
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

            total_train_loss += loss.item()
            loss = loss / grad_accum_steps

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % grad_accum_steps == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()

            if step % 1000 == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    f"[TRAIN] Epoch: {epoch+1}, Step: {step}, "
                    f"Loss: {loss.item() * grad_accum_steps:.4f}, LR: {current_lr}"
                )

        avg_train_loss = total_train_loss / len(train_dataloader)

        # Validation phase
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
                    with torch.amp.autocast(device):
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

        epoch_end_time = time.time()
        logger.info(
            f"Epoch {epoch + 1}/{num_epochs} - "
            f"Train Loss: {avg_train_loss:.4f} - "
            f"Val Loss: {avg_val_loss:.4f} - "
            f"Time: {epoch_end_time - epoch_start_time:.2f}s"
        )
        mlflow.log_metric("train_loss", avg_train_loss, step=epoch + 1)
        mlflow.log_metric("val_loss", avg_val_loss, step=epoch + 1)


        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improvement_epochs = 0
            if save_weights:
                torch.save(
                    model.state_dict(),
                    f"output/model_weights/{model_name.lower()}_weights_{epoch + 1}_epochs.pth",
                )
                logger.info(
                    f"Best model saved at epoch {epoch+1} with val loss {avg_val_loss:.4f}"
                )
        else:
            no_improvement_epochs += 1

        if (
            early_stopping_patience is not None
            and no_improvement_epochs >= early_stopping_patience
        ):
            logger.info(
                f"No improvement for {no_improvement_epochs} consecutive epochs. "
                f"Early stopping triggered!"
            )
            break

    logger.info(f"Total training time: {time.time() - total_start_time:.2f}s")
    return best_val_loss


def finetune_model_with_gridsearch_cv(
    model_class,
    base_params_model,
    train_articles,
    train_summaries,
    tokenizer,
    device,
    k_folds=3,
    num_epochs=3,
    batch_size=32,
    n_process=1,
    seed=42,
    grad_accum_steps=1,
    use_amp=False,
    early_stopping_patience=None,
):
    logger = setup_logger()
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    param_grid = {
        "learning_rate": [5e-6, 1e-6],
        "weight_decay": [0.0, 1e-2],
        "label_smoothing": [0.0, 0.1],
    }

    dataset_indices = np.arange(len(train_articles))
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    all_combos = list(ParameterGrid(param_grid))
    best_params = None
    best_loss = float("inf")

    for combo_idx, combo in enumerate(all_combos):
        logger.info(f"=== Combination {combo_idx + 1}/{len(all_combos)}: {combo} ===")

        with mlflow.start_run(run_name=f"grid_combo_{combo_idx+1}", nested=True):
            mlflow.set_tag("cv_fold_count", k_folds)
            for param, value in combo.items():
                mlflow.log_param(param, value)

            fold_losses = []

            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(dataset_indices)):
                logger.info(f"  -- Fold {fold_idx + 1}/{k_folds} --")
                mlflow.log_param(f"fold_{fold_idx+1}_size", len(train_idx))

                train_articles_fold = [train_articles[i] for i in train_idx]
                train_summaries_fold = [train_summaries[i] for i in train_idx]
                val_articles_fold = [train_articles[i] for i in val_idx]
                val_summaries_fold = [train_summaries[i] for i in val_idx]

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

                model = model_class(**base_params_model)

                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=combo["learning_rate"],
                    betas=(0.9, 0.98),
                    eps=1e-9,
                    weight_decay=combo["weight_decay"],
                )

                scheduler = create_scheduler(train_dataloader_fold, optimizer, num_epochs)

                loss_fn = torch.nn.CrossEntropyLoss(
                    ignore_index=tokenizer.pad_token_id,
                    label_smoothing=combo["label_smoothing"],
                )

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
                    save_weights=False,
                    grad_accum_steps=grad_accum_steps,
                    use_amp=use_amp,
                    early_stopping_patience=early_stopping_patience,
                )

                fold_losses.append(best_val_loss_fold)
                mlflow.log_metric(f"val_loss_fold_{fold_idx+1}", best_val_loss_fold)

            avg_val_loss = np.mean(fold_losses)
            mlflow.log_metric("avg_val_loss", avg_val_loss)

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_params = combo

    logger.info("===== GRID SEARCH RESULTS =====")
    logger.info(f"Best hyperparameters: {best_params}")
    logger.info(f"Best average validation loss: {best_loss:.4f}")
    mlflow.log_params(best_params)
    mlflow.log_metric("best_avg_val_loss", best_loss)

    return best_params
