"""
main.py

Orchestrating entry point for the NeuMF Movie Recommendation project.

WHY THIS MODULE EXISTS:
Each pipeline stage (preprocessing, training, evaluation, recommendation)
is independently runnable as its own script (`python -m src.preprocessing`,
etc.), which is useful during development. However, a grader or new user
should be able to run the entire project -- or any subset of stages -- from
a single, well-documented command. `main.py` provides that unified CLI
entry point, wiring the stages together in the correct order and applying
global reproducibility settings once at the start.

USAGE:
    python main.py --stage all --user_id 1
    python main.py --stage preprocess
    python main.py --stage train
    python main.py --stage evaluate
    python main.py --stage recommend --user_id 25 --top_k 10
"""

import argparse
import sys

from config import data_cfg, ensure_directories
from src.utils import set_seed, get_logger, timer


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments controlling which pipeline stage(s) to run.

    Returns:
        argparse.Namespace: parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="NeuMF Movie Recommendation System -- pipeline orchestrator."
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "preprocess", "train", "evaluate", "recommend"],
        help=(
            "Which pipeline stage to run. 'all' runs preprocess -> train -> "
            "evaluate -> recommend (demo) in sequence. Default: 'all'."
        ),
    )
    parser.add_argument(
        "--user_id",
        type=int,
        default=1,
        help=(
            "Raw MovieLens UserID to generate demo recommendations for "
            "(used only when --stage is 'all' or 'recommend'). Default: 1."
        ),
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of movie recommendations to generate. Default: 10.",
    )
    parser.add_argument(
        "--no_history",
        action="store_true",
        help=(
            "Skip printing the user's watch history before recommendations "
            "(used only when --stage is 'all' or 'recommend')."
        ),
    )
    parser.add_argument(
        "--no_explain",
        action="store_true",
        help=(
            "Skip attaching genre-overlap explanations to each recommendation "
            "(used only when --stage is 'all' or 'recommend')."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """
    Run the requested pipeline stage(s) in the correct order.

    WHY imports happen inside each branch rather than at module top-level:
    `src/preprocessing.py`, `src/train.py`, etc. each import heavier
    dependencies (torch, sklearn) at module load time. Deferring these
    imports until the specific stage is actually requested keeps
    `python main.py --stage recommend` from paying the (small) import
    cost of, say, the preprocessing module's sklearn dependency if it
    isn't needed -- and, more importantly, gives clearer tracebacks if a
    given stage's dependencies are missing.
    """
    args = parse_args()

    ensure_directories()

    # WHY seed once here, before any stage runs: this is the single choke
    # point through which every invocation of the pipeline passes,
    # guaranteeing reproducible splits, weight initialization, and
    # negative sampling regardless of which stage(s) are requested.
    set_seed(data_cfg.random_seed)

    logger = get_logger("main")
    logger.info(f"Starting pipeline with stage='{args.stage}'")

    if args.stage in ("all", "preprocess"):
        from src.preprocessing import run_preprocessing_pipeline

        with timer("Preprocessing stage", logger=logger):
            run_preprocessing_pipeline()

    if args.stage in ("all", "train"):
        from src.train import run_training_pipeline

        with timer("Training stage", logger=logger):
            run_training_pipeline()

    if args.stage in ("all", "evaluate"):
        from src.evaluate import run_evaluation_pipeline

        with timer("Evaluation stage", logger=logger):
            run_evaluation_pipeline()

    if args.stage in ("all", "recommend"):
        from src.recommend import run_recommendation_demo

        with timer("Recommendation demo stage", logger=logger):
            try:
                run_recommendation_demo(
                    raw_user_id=args.user_id,
                    top_k=args.top_k,
                    show_history=not args.no_history,
                    explain=not args.no_explain,
                )
            except ValueError as exc:
                # WHY catch specifically ValueError: recommend_top_k raises
                # ValueError for cold-start (unseen) user IDs -- a user
                # error, not a bug -- so we log it cleanly and exit
                # gracefully rather than printing a full stack trace.
                logger.error(str(exc))
                sys.exit(1)

    logger.info("Pipeline finished successfully.")


if __name__ == "__main__":
    main()
