# Standard library
import os

# Third-party libraries
import matplotlib.pyplot as plt

# Internal modules
from src.utils.setup_logger import setup_logger


def descriptive_statistics(data, column_name):
    """
    Calculates and logs descriptive statistics for text length
    in the specified column of a DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame containing the data.
    column_name : str
        The name of the column to analyze.
    """
    logger = setup_logger()

    # Calculate text lengths
    text_lengths = data[column_name].astype(str).str.len()

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
        logger.info(f"{key.capitalize()} {column_name} length: {value}")

    return desc_stats


def plot_text_length_distribution(data, column_name):
    """
    Plots a histogram of text lengths from the specified column in the given DataFrame and saves it.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame containing the data.
    column_name : str
        The name of the column to analyze.
    """
    logger = setup_logger()

    text_lengths = data[column_name].astype(str).str.len()

    plt.figure(figsize=(10, 6))
    plt.hist(text_lengths, bins=50, edgecolor="black", alpha=0.7)
    mean_val = text_lengths.mean()
    plt.axvline(
        mean_val,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_val:.2f}",
    )
    plt.xlabel(f"Length of '{column_name}' (characters)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    output_dir = "output/graphs/"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    plot_path = os.path.join(output_dir, f"{column_name}_text_length_distribution.png")
    plt.savefig(plot_path)
    plt.close()

    logger.info(
        f"Histogram of {column_name} lengths' distribution saved at {plot_path}"
    )
