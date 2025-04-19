# Third-party libraries
import numpy as np

# Internal modules
from src.utils.spacy_utils import load_spacy_model

# Load spaCy English model
nlp = load_spacy_model()


def remove_outlier(df, col, lower_percent=10, upper_percent=90):
    """
    Filter out "outlier" rows from df based on text length in the specified column.

    - df: Your pandas DataFrame.
    - col: The column in df containing text.
    - lower_percent, upper_percent: Percentiles for filtering. Rows whose text length
      is outside [lower_bound, upper_bound] are dropped.

    Returns a filtered DataFrame.
    """
    # Convert the column to an array of strings
    texts = df[col].values

    # Compute text lengths
    text_lengths = np.array([len(t) for t in texts])

    # Calculate percentile cutoffs
    lower_bound = np.percentile(text_lengths, lower_percent)
    upper_bound = np.percentile(text_lengths, upper_percent)

    # Create a boolean mask of valid (in-range) rows
    mask = (text_lengths >= lower_bound) & (text_lengths <= upper_bound)

    # Return the filtered DataFrame
    return df[mask].copy()


def preprocess_articles(texts, n_process, batch_size=32):
    cleaned_texts = []
    for doc in nlp.pipe(texts, batch_size=batch_size, n_process=n_process):
        tokens = [
            token.lemma_.lower()
            for token in doc
            if not token.is_stop and not token.is_punct and not token.is_space
        ]
        cleaned_texts.append(" ".join(tokens))
    return cleaned_texts


def preprocess_summaries(texts, n_process, batch_size=32):
    cleaned_summaries = []
    for doc in nlp.pipe(texts, batch_size=batch_size, n_process=n_process):
        tokens = [token.text.lower() for token in doc if not token.is_space]
        cleaned_summaries.append(" ".join(tokens))
    return cleaned_summaries
