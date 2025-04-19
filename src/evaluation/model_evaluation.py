import evaluate

from src.utils.setup_logger import setup_logger


def evaluate_model(data, predictions):
    """
    Evaluate the model using ROUGE metrics.

    Parameters:
    - data (pd.DataFrame): The dataset used for evaluation.
    - predictions (List[str]): The generated summaries.

    Returns:
    - None
    """

    logger = setup_logger()
    rouge = evaluate.load("rouge")

    data.loc[:, "predictions_transformer"] = predictions

    reference_summaries = list(data["Summary"])
    results = rouge.compute(predictions=predictions, references=reference_summaries)
    logger.info(f"ROUGE metrics: {results}")
