import os

import torch

from src.setup_logger import setup_logger

# Initialize logger
logger = setup_logger()


def set_up_device():
    """
    Set up the device to use for the model.

    Parameters:
    -None

    Returns:
    -str: The device to use for the model.
    """
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info(f"Using {device} device")
    return device


def get_allowed_cpu_count() -> int:
    """
    Returns the number of CPU cores available for this process.

    Parameters:
    -None

    Returns:
    -int: The number of CPU cores available for this process.
    """
    try:
        nbr_cpu = len(os.sched_getaffinity(0))
    except AttributeError:
        nbr_cpu = os.cpu_count() or 1
    logger.info(f"Using {nbr_cpu} CPUs")
    return nbr_cpu


def set_up_config_device(cpu_count: int) -> int:
    """
    Set up the configuration for the device and the number of CPUs to use.

    Parameters:
    - cpu_count (int): The number of CPUs to use.

    Returns:
    - n_process (int): The number of processes used by torch.
    """

    n_process = max(1, cpu_count // 2)

    torch.set_num_threads(n_process)
    logger.info(f"torch set up to use {n_process} processes")
    return n_process
