# Third-party libraries
import spacy
from spacy.cli import download

# Internal modules
from src.utils.setup_logger import setup_logger


def load_spacy_model(model_name="en_core_web_sm"):
    """
    Loads a spaCy model, downloading it if necessary.

    Parameters:
    - model_name (str): The name of the spaCy model to load.

    Returns:
    - nlp: The loaded spaCy language model.
    """
    logger = setup_logger()
    try:
        return spacy.load(model_name, disable=["parser", "ner"])
    except OSError:
        logger.info(f"Model '{model_name}' not found. Downloading...")
        download(model_name)
        return spacy.load(model_name, disable=["parser", "ner"])
