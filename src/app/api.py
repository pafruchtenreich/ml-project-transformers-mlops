# import torch
from fastapi import FastAPI, HTTPException
from transformers import BartTokenizer

from src.models.transformer import Transformer
from src.prediction.generate_summaries_transformer import generate_summaries_transformer

# Initialize FastAPI app
app = FastAPI(title="Génération de résumé d'articles")

# Initialize tokenizer and model
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

PARAMS_MODEL = {
    "pad_idx": tokenizer.pad_token_id,
    "hidden_size": 512,
    "n_head": 8,
    "max_len": 512,
    "dec_max_len": 150,
    "ffn_hidden": 2048,
    "n_layers": 6,
    "voc_size": len(tokenizer),
}

modelTransformer = Transformer(**PARAMS_MODEL)
# modelTransformer.load_state_dict(
# torch.load("output/model_weights/transformer_weights_3_epochs.pth")
# )
modelTransformer.eval()


@app.get("/", tags=["Welcome"])
def show_welcome_page():
    """
    Show welcome page with model name and version.
    """

    return {
        "Message": "API for articles' summary generation",
        "Model_name": "TO DO",
        "Model_version": "TO DO",
    }


@app.get("/summarize/", tags=["Summarize"])
async def summarize_article(article: str):
    """
    Endpoint to summarize an article.
    """
    try:
        # Tokenize input article
        tokenized_input = tokenizer(
            article,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding="max_length",
        )

        # Generate summary using the trained model
        summary_ids = generate_summaries_transformer(
            model=modelTransformer,
            batch_size=1,
            tokenized_input=tokenized_input["input_ids"],
        )

        # Decode generated summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return {"summary": summary}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating summary: {str(e)}"
        )
