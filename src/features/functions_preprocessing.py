import spacy

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")


def preprocess_article(text):
    """
    Preprocess a given text: tokenize, remove stopwords, and lemmatize.
    Args:
        text (str): Input text to preprocess.
    Returns:
        str: Preprocessed text as a single string of tokens.
    """
    # Apply spaCy model to the text
    doc = nlp(text)

    # Tokenize, remove stopwords, and lemmatize
    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop and not token.is_punct and not token.is_space
    ]

    # Join tokens into a single string
    return " ".join(tokens)


def preprocess_summary(text):
    """
    Preprocess a summary: minimal preprocessing (lowercase, add special tokens).
    Args:
        text (str): Input summary text.
    Returns:
        str: Preprocessed summary text with <START> and <END>.
    """
    # Tokenize and lowercase
    doc = nlp(text)
    tokens = [token.text.lower() for token in doc if not token.is_space]
    # Add special tokens
    return "<START> " + " ".join(tokens) + " <END>"
