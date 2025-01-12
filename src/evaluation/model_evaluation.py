import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer
from tqdm import tqdm

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


def generate_summaries_seq2seq(model, batch_size, tokenized_input, limit=None):
    """
    Generates summaries using a Seq2Seq model.

    Parameters:
    - model (torch.nn.Module): The pre-trained Seq2Seq model.
    - batch_size (int): Number of samples per batch.
    - tokenized_input (torch.Tensor): Tokenized input articles.
    - limit (int or None): Maximum number of batches to process. If None, processes all batches.

    Returns:
    - List of generated summaries.
    """
    model.to(device)
    model.eval()
    dataset = TensorDataset(tokenized_input)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    predictions_seq2seq = []
    with torch.no_grad():  # Disable gradient calculation for inference
        for i, batch in enumerate(
            tqdm(dataloader, desc="Processing Batches", unit="batch")
        ):
            if limit is not None and i == limit:  # Stop after reaching the limit
                break

            tokenized_inputs = batch[0].to(device)
            encoder_outputs, encoder_hidden = model.encoder(tokenized_inputs)

            batch_size = tokenized_inputs.size(0)
            decoder_input = torch.tensor(
                [tokenizer.cls_token_id] * batch_size, device=device
            )  # Start with [CLS] tokens
            decoder_hidden = encoder_hidden
            generated_tokens = torch.zeros(
                (batch_size, 128), dtype=torch.long, device=device
            )

            for step in range(127):
                logits, decoder_hidden = model.decoder(decoder_input, decoder_hidden)
                next_token = logits.argmax(dim=1)
                generated_tokens[:, step] = next_token

                if (
                    next_token == tokenizer.sep_token_id
                ).all():  # Stop if all sequences have [SEP]
                    break

                decoder_input = next_token

            for tokens in generated_tokens:
                summary = tokenizer.decode(tokens, skip_special_tokens=True)
                predictions_seq2seq.append(summary)

    return predictions_seq2seq


def generate_summaries_transformer(model, batch_size, tokenized_input, limit=None):
    """
    Generates summaries using a Transformer model.

    Parameters:
    - model (torch.nn.Module): The pre-trained Transformer model.
    - batch_size (int): Number of samples per batch.
    - tokenized_input (torch.Tensor): Tokenized input articles.
    - limit (int or None): Maximum number of batches to process. If None, processes all batches.

    Returns:
    - List of generated summaries.
    """
    model.to(device)
    model.eval()
    dataset = TensorDataset(tokenized_input)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    predictions_transformer = []
    with torch.no_grad():  # Disable gradient calculation for inference
        for i, batch in enumerate(
            tqdm(dataloader, desc="Processing Batches", unit="batch")
        ):
            if limit is not None and i == limit:  # Stop after reaching the limit
                break

            tokenized_inputs = batch[0].to(device)
            current_batch_size = tokenized_inputs.size(0)
            start_summaries = torch.zeros((current_batch_size, 128), device=device)
            start_summaries[:, 0] = 101  # [CLS] token

            for k in range(0, 127):
                output = model(tokenized_inputs.long(), start_summaries.long())
                start_summaries[:, k + 1] = output.argmax(dim=-1)[:, k].detach()
                if (
                    start_summaries[:, k + 1] == 102
                ).all():  # Stop if all sequences end with [SEP]
                    break

            for start_summary in start_summaries:
                summary = tokenizer.decode(
                    start_summary.long(), skip_special_tokens=True
                )
                predictions_transformer.append(summary)

    return predictions_transformer


def generate_summaries_bert(model, batch_size, tokenized_input, limit=None):
    """
    Generates summaries using a pre-trained BERT model.

    Parameters:
    - model (torch.nn.Module): The pre-trained BERT model.
    - batch_size (int): Number of samples per batch.
    - tokenized_input (torch.Tensor): Tokenized input articles.
    - limit (int or None): Maximum number of batches to process. If None, processes all batches.

    Returns:
    - List of generated summaries.
    """
    model.to(device)
    model.eval()
    dataset = TensorDataset(tokenized_input)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    predictions_bert = []
    with torch.no_grad():  # Disable gradient calculation for inference
        for i, batch in enumerate(
            tqdm(dataloader, desc="Processing Batches", unit="batch")
        ):
            if limit is not None and i == limit:  # Stop after reaching the limit
                break

            tokenized_inputs = batch[0].to(device)
            outputs = model(tokenized_inputs)  # Forward pass
            predicted_ids = torch.argmax(outputs, dim=-1)  # Argmax for token selection

            for predicted_id in predicted_ids:
                summary = tokenizer.decode(predicted_id, skip_special_tokens=True)
                predictions_bert.append(summary)

    return predictions_bert
