"""
utils.py

Shared utility functions used across multiple modules in the project.

WHY THIS MODULE EXISTS:
Certain concerns -- reproducibility (seeding), consistent logging output,
and timing long-running stages -- are cross-cutting and needed by
preprocessing, training, evaluation, and recommendation alike. Placing
them in one shared module avoids duplicating this logic in every file
and keeps each pipeline module focused on its own core responsibility.
"""

import logging
import os
import random
import time
from contextlib import contextmanager
from typing import Iterator, Optional

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Seed all relevant random number generators for reproducibility.

    WHY: PyTorch, NumPy, and Python's built-in `random` module each
    maintain independent RNG state. Data shuffling (DataLoader), weight
    initialization, dropout masks, and negative sampling in evaluate.py
    all draw from these generators, so seeding only one of them would
    still leave other sources of run-to-run variance -- undermining the
    reproducibility that is explicitly required for academic evaluation.

    Args:
        seed: the seed value to apply across all RNGs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        # WHY seed both variants: manual_seed only seeds the current GPU;
        # manual_seed_all covers multi-GPU setups for full determinism.
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # WHY these two flags together: cudnn's default algorithm selection
    # (benchmark=True) picks the fastest convolution algorithm by timing
    # multiple candidates, which is itself non-deterministic. Disabling
    # benchmark mode and enabling deterministic mode trades a small amount
    # of speed for exactly reproducible results across runs.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    Create (or retrieve) a configured logger that writes to the console
    and, optionally, to a file.

    WHY a shared logger factory: using plain `print()` statements across
    preprocessing/train/evaluate/recommend makes it hard to distinguish
    which module produced which line once outputs are redirected to a
    file, and provides no severity levels (info/warning/error). A named
    logger with a consistent format solves both issues.

    Args:
        name: logger name, conventionally the calling module's __name__.
        log_file: optional path to also write log records to a file.

    Returns:
        logging.Logger: a configured logger instance.
    """
    logger = logging.getLogger(name)

    # WHY guard with `if not logger.handlers`: get_logger may be called
    # multiple times with the same name (e.g. across repeated imports in
    # a notebook); without this guard, handlers would be duplicated,
    # causing every log message to print multiple times.
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


@contextmanager
def timer(task_name: str, logger: Optional[logging.Logger] = None) -> Iterator[None]:
    """
    Context manager that times a block of code and reports the elapsed
    duration.

    WHY: preprocessing, training, and evaluation each involve
    long-running stages (e.g. parsing ~1M ratings, running 20 training
    epochs, sampling 99 negatives per test interaction). Wrapping these
    stages in a timer gives visibility into where time is spent, which is
    useful both for debugging performance and for reporting reproducible
    runtime figures in the README/report.

    Args:
        task_name: human-readable description of the timed task.
        logger: optional logger to report the elapsed time through
            (falls back to `print` if not provided).

    Yields:
        None
    """
    start_time = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        message = f"[timer] '{task_name}' completed in {elapsed:.2f} seconds."
        if logger is not None:
            logger.info(message)
        else:
            print(message)


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """
    Count the total number of trainable parameters in a PyTorch model.

    WHY: reporting model size (parameter count) is standard practice in
    ML reports/READMEs to characterize model complexity, and is useful
    for sanity-checking that the architecture matches expectations
    (e.g. catching an accidental embedding_dim misconfiguration).

    Args:
        model: any torch.nn.Module.

    Returns:
        int: total number of parameters with requires_grad=True.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_metrics_table(metrics: dict) -> str:
    """
    Format a flat dict of scalar metrics into an aligned, human-readable
    table string.

    WHY: `evaluate.py` produces a dict of metrics that is dumped to JSON
    for machine consumption, but a human skimming console output benefits
    from a cleanly aligned table rather than a raw dict repr.

    Args:
        metrics: dict mapping metric name -> scalar value (int/float).
            Non-scalar values (e.g. confusion_matrix as a nested list)
            are skipped.

    Returns:
        str: multi-line formatted table.
    """
    lines = []
    scalar_items = {
        k: v for k, v in metrics.items() if isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    if not scalar_items:
        return "(no scalar metrics to display)"

    max_key_len = max(len(k) for k in scalar_items)
    for key, value in scalar_items.items():
        lines.append(f"{key.ljust(max_key_len)} : {value:.4f}" if isinstance(value, float)
                     else f"{key.ljust(max_key_len)} : {value}")

    return "\n".join(lines)


if __name__ == "__main__":
    # Quick sanity check for the utilities in this module.
    set_seed(42)
    logger = get_logger("utils_demo")
    logger.info("Seed set and logger configured successfully.")

    with timer("dummy sleep task", logger=logger):
        time.sleep(0.5)

    sample_metrics = {"accuracy": 0.8231, "precision": 0.7912, "num_users": 6040}
    print("\nFormatted metrics table:")
    print(format_metrics_table(sample_metrics))
