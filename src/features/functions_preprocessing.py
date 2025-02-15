import matplotlib.pyplot as plt
import spacy

from src.setup_logger import setup_logger

# Load spaCy English model
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


def descriptive_statistics(df, column_name):
    """
    Calculates and logs descriptive statistics for text length
     in the specified column of a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data.
    column_name : str
        The name of the column to analyze.
    """
    logger = setup_logger()

    # Calculate text lengths
    text_lengths = df[column_name].astype(str).str.len()

    # Calculate descriptive statistics
    desc_stats = {
        "mean": text_lengths.mean(),
        "median": text_lengths.median(),
        "max": text_lengths.max(),
        "min": text_lengths.min(),
        "25%": text_lengths.quantile(0.25),
        "75%": text_lengths.quantile(0.75),
    }

    # Log descriptive statistics
    for key, value in desc_stats.items():
        logger.info(f"{key.capitalize()}: {value}")

    return desc_stats


def plot_text_length_distribution(df, column_name):
    """
    Plots a histogram of text lengths from the specified column in the given DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data.
    column_name : str
        The name of the column to analyze.
    """
    # Calculate the length (number of characters) of each text entry in the column
    text_lengths = df[column_name].astype(str).str.len()

    # Create the figure
    plt.figure(figsize=(10, 6))

    # Plot the histogram
    plt.hist(text_lengths, bins=50, edgecolor="black", alpha=0.7, color="blue")

    # Calculate and plot the mean
    mean_val = text_lengths.mean()
    plt.axvline(
        mean_val,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_val:.2f}",
    )

    # Configure labels, legend, and grid
    plt.xlabel(f"Length of '{column_name}' (characters)", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Display the plot
    plt.show()


def preprocess_articles(texts, n_process, batch_size=32):
    """
    Args:
        texts (List[str]): List of text documents to preprocess.
        batch_size (int): Batch size for parallel processing.
        n_process (int): Number of processes for parallel execution.

    Returns:
        List[str]: A list of cleaned articles (lemmatized, no stopwords/punct/spaces).
    """
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
    """
    Performs minimal preprocessing for summaries (lowercasing + tokens).

    Args:
        texts (List[str]): List of summary texts.
        batch_size (int): Batch size for parallel processing.
        n_process (int): Number of processes for parallel execution.

    Returns:
        List[str]: A list of preprocessed summaries.
    """
    cleaned_summaries = []
    for doc in nlp.pipe(texts, batch_size=batch_size, n_process=n_process):
        tokens = [token.text.lower() for token in doc if not token.is_space]
        cleaned_summaries.append(" ".join(tokens))
    return cleaned_summaries
