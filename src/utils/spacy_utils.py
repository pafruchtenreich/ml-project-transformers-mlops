# Third-party libraries
import spacy
from spacy.cli import download


def load_spacy_model(model_name="en_core_web_sm"):
    """
    Loads a spaCy model, downloading it if necessary.

    Parameters:
    - model_name (str): The name of the spaCy model to load.

    Returns:
    - nlp: The loaded spaCy language model.
    """
    try:
        return spacy.load(model_name, disable=["parser", "ner"])
    except OSError:
        print(f"Model '{model_name}' not found. Downloading...")
        download(model_name)
        return spacy.load(model_name, disable=["parser", "ner"])
