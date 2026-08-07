"""
preprocessing.py

Data preprocessing pipeline for the MovieLens 1M dataset.

WHY THIS MODULE EXISTS:
Raw MovieLens files are not directly usable by a PyTorch model: user and
movie identifiers are sparse, non-contiguous integers; ratings are ordinal
(1-5) rather than the binary implicit-feedback labels NeuMF is trained on;
and there is no pre-defined train/val/test split. This module performs all
of that work once, offline, and persists the result so that `dataset.py`,
`train.py`, and `recommend.py` can simply load ready-to-use CSV files
instead of repeating this logic.
"""

import os
import pickle
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from config import data_cfg, paths, ensure_directories


def load_raw_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the three raw MovieLens 1M .dat files into pandas DataFrames.

    MovieLens 1M files use '::' as a field separator and are encoded in
    latin-1 (movie titles contain non-UTF-8 characters such as accented
    letters), so both must be specified explicitly to avoid parsing errors.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            (ratings_df, movies_df, users_df)

    Raises:
        FileNotFoundError: if any of the expected raw files are missing.
    """
    for path in (paths.ratings_file, paths.movies_file, paths.users_file):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Expected MovieLens 1M file not found at '{path}'. "
                "Please download the dataset from "
                "https://grouplens.org/datasets/movielens/1m/ and place "
                "ratings.dat, movies.dat, users.dat inside data/raw/."
            )

    # ratings.dat :: UserID::MovieID::Rating::Timestamp
    ratings_df = pd.read_csv(
        paths.ratings_file,
        sep=data_cfg.dat_separator,
        engine="python",
        header=None,
        names=["UserID", "MovieID", "Rating", "Timestamp"],
        encoding=data_cfg.dat_encoding,
    )

    # movies.dat :: MovieID::Title::Genres
    movies_df = pd.read_csv(
        paths.movies_file,
        sep=data_cfg.dat_separator,
        engine="python",
        header=None,
        names=["MovieID", "Title", "Genres"],
        encoding=data_cfg.dat_encoding,
    )

    # users.dat :: UserID::Gender::Age::Occupation::Zip-code
    users_df = pd.read_csv(
        paths.users_file,
        sep=data_cfg.dat_separator,
        engine="python",
        header=None,
        names=["UserID", "Gender", "Age", "Occupation", "ZipCode"],
        encoding=data_cfg.dat_encoding,
    )

    return ratings_df, movies_df, users_df


def clean_data(
    ratings_df: pd.DataFrame, movies_df: pd.DataFrame, users_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Clean the raw DataFrames by handling missing values and duplicates.

    WHY: Even well-known benchmark datasets can contain duplicate rows
    (e.g. accidental double-logging of an interaction) or malformed rows.
    Silently proceeding without checking would let corrupted rows
    propagate into label generation and model training.

    Args:
        ratings_df: raw ratings DataFrame.
        movies_df: raw movies DataFrame.
        users_df: raw users DataFrame.

    Returns:
        Tuple of cleaned (ratings_df, movies_df, users_df).
    """
    # Report missing values so data quality issues are visible, not hidden.
    for name, df in (("ratings", ratings_df), ("movies", movies_df), ("users", users_df)):
        n_missing = df.isnull().sum().sum()
        print(f"[clean_data] '{name}' missing value count: {n_missing}")

    # Drop rows with missing critical fields; MovieLens 1M is generally
    # complete, but this guards against corrupted downloads.
    ratings_df = ratings_df.dropna(subset=["UserID", "MovieID", "Rating"])
    movies_df = movies_df.dropna(subset=["MovieID", "Title"])
    users_df = users_df.dropna(subset=["UserID"])

    # Remove duplicate interactions: the same (UserID, MovieID) pair
    # appearing twice would bias the model toward that user-item pair.
    before = len(ratings_df)
    ratings_df = ratings_df.drop_duplicates(subset=["UserID", "MovieID"], keep="last")
    after = len(ratings_df)
    print(f"[clean_data] Removed {before - after} duplicate rating rows.")

    movies_df = movies_df.drop_duplicates(subset=["MovieID"], keep="first")
    users_df = users_df.drop_duplicates(subset=["UserID"], keep="first")

    return ratings_df, movies_df, users_df


def encode_ids(ratings_df: pd.DataFrame) -> Tuple[pd.DataFrame, LabelEncoder, LabelEncoder]:
    """
    Encode raw MovieLens UserID/MovieID into contiguous zero-indexed integers.

    WHY: PyTorch nn.Embedding layers require indices in the contiguous
    range [0, num_embeddings). Raw MovieLens IDs are sparse (e.g. MovieID
    can jump from 1 to 3952 with many gaps), so encoding is mandatory
    before the IDs can be used as embedding lookup indices.

    Args:
        ratings_df: cleaned ratings DataFrame with 'UserID' and 'MovieID'.

    Returns:
        Tuple containing:
            - ratings_df with new 'user_idx' and 'movie_idx' columns.
            - fitted user LabelEncoder.
            - fitted movie LabelEncoder.
    """
    user_encoder = LabelEncoder()
    movie_encoder = LabelEncoder()

    ratings_df = ratings_df.copy()
    ratings_df["user_idx"] = user_encoder.fit_transform(ratings_df["UserID"])
    ratings_df["movie_idx"] = movie_encoder.fit_transform(ratings_df["MovieID"])

    return ratings_df, user_encoder, movie_encoder


def generate_labels(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert explicit star ratings into implicit binary feedback labels.

    Rule (per project spec):
        Rating >= 4  -> label = 1 (positive interaction)
        Rating <  4  -> label = 0 (negative interaction)

    WHY implicit feedback: NeuMF is trained as a binary classifier over
    interactions, matching real-world recommendation settings where you
    typically observe whether a user engaged positively with an item
    rather than a fine-grained star rating.

    Args:
        ratings_df: DataFrame containing a 'Rating' column.

    Returns:
        pd.DataFrame with a new 'label' column of dtype int.
    """
    ratings_df = ratings_df.copy()
    ratings_df["label"] = (ratings_df["Rating"] >= data_cfg.positive_rating_threshold).astype(int)
    return ratings_df


def split_data(ratings_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the interaction data into train/validation/test sets.

    WHY stratify on 'label': the positive/negative class ratio in
    MovieLens 1M is imbalanced (most rated movies skew toward positive
    ratings). Stratified splitting ensures train/val/test sets share a
    similar label distribution, giving a fair validation/test signal.

    Args:
        ratings_df: DataFrame with a 'label' column.

    Returns:
        Tuple of (train_df, val_df, test_df) following the 80/10/10 ratio
        defined in config.DataConfig.
    """
    # First split off the test set (10%).
    train_val_df, test_df = train_test_split(
        ratings_df,
        test_size=data_cfg.test_ratio,
        random_state=data_cfg.random_seed,
        stratify=ratings_df["label"],
    )

    # Remaining 90% is split into train (80% of total) and val (10% of total),
    # so val_ratio must be expressed relative to the train_val subset.
    relative_val_ratio = data_cfg.val_ratio / (data_cfg.train_ratio + data_cfg.val_ratio)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_ratio,
        random_state=data_cfg.random_seed,
        stratify=train_val_df["label"],
    )

    return train_df, val_df, test_df


def build_user_positive_items(ratings_df: pd.DataFrame) -> Dict[int, set]:
    """
    Build a lookup of movies each user has positively interacted with.

    WHY: `recommend.py` must exclude already-watched movies from Top-K
    recommendations, and `evaluate.py` needs each user's full positive
    item set to correctly sample "unseen" negatives during ranking
    evaluation. Precomputing this once avoids repeated expensive filtering
    of the full ratings DataFrame at inference/evaluation time.

    Args:
        ratings_df: full (unsplit) ratings DataFrame with 'user_idx',
            'movie_idx', and 'label' columns.

    Returns:
        Dict[int, set]: mapping from user_idx -> set of movie_idx the user
        rated positively (label == 1).
    """
    positive_df = ratings_df[ratings_df["label"] == 1]
    return positive_df.groupby("user_idx")["movie_idx"].apply(set).to_dict()


def save_processed_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    user_encoder: LabelEncoder,
    movie_encoder: LabelEncoder,
    user_positive_items: Dict[int, set],
) -> None:
    """
    Persist all preprocessing artifacts to disk.

    WHY: separating preprocessing (run once) from training/evaluation/
    recommendation (run many times during experimentation) drastically
    speeds up iteration -- there is no need to re-parse and re-encode the
    raw .dat files every time a model is trained.

    Args:
        train_df: training split.
        val_df: validation split.
        test_df: test split.
        movies_df: cleaned movies metadata DataFrame.
        user_encoder: fitted user LabelEncoder.
        movie_encoder: fitted movie LabelEncoder.
        user_positive_items: mapping of user_idx -> set of positively
            rated movie_idx values.
    """
    ensure_directories()

    columns_to_save = ["user_idx", "movie_idx", "label"]
    train_df[columns_to_save].to_csv(paths.train_file, index=False)
    val_df[columns_to_save].to_csv(paths.val_file, index=False)
    test_df[columns_to_save].to_csv(paths.test_file, index=False)

    movies_df.to_csv(paths.movies_processed_file, index=False)

    with open(paths.user_encoder_file, "wb") as f:
        pickle.dump(user_encoder, f)

    with open(paths.movie_encoder_file, "wb") as f:
        pickle.dump(movie_encoder, f)

    with open(paths.user_positive_items_file, "wb") as f:
        pickle.dump(user_positive_items, f)

    print(f"[save_processed_data] Train set: {len(train_df)} rows -> {paths.train_file}")
    print(f"[save_processed_data] Val set:   {len(val_df)} rows -> {paths.val_file}")
    print(f"[save_processed_data] Test set:  {len(test_df)} rows -> {paths.test_file}")
    print(f"[save_processed_data] Encoders and lookup tables saved to {paths.processed_data_dir}")


def attach_movie_idx_to_movies(
    movies_df: pd.DataFrame, movie_encoder: LabelEncoder
) -> pd.DataFrame:
    """
    Attach the encoded movie_idx to the movies metadata DataFrame.

    WHY: `recommend.py` needs to map a predicted movie_idx back to its
    Title/Genres for display. Movies present in movies.dat but never rated
    (and thus never seen by the encoder) are dropped, since the model has
    no embedding for them and they cannot be recommended.

    Args:
        movies_df: cleaned movies metadata DataFrame.
        movie_encoder: fitted movie LabelEncoder (fit on rated MovieIDs).

    Returns:
        pd.DataFrame: movies_df filtered to only encoder-known MovieIDs,
        with an added 'movie_idx' column.
    """
    known_ids = set(movie_encoder.classes_)
    movies_df = movies_df[movies_df["MovieID"].isin(known_ids)].copy()
    movies_df["movie_idx"] = movie_encoder.transform(movies_df["MovieID"])
    return movies_df


def run_preprocessing_pipeline() -> None:
    """
    Execute the full preprocessing pipeline end-to-end.

    WHY a single orchestrating function: `main.py` should be able to
    trigger the entire preprocessing stage with one call, while each
    individual step remains independently unit-testable.
    """
    print("=" * 70)
    print("STEP 1: Loading raw MovieLens 1M data")
    print("=" * 70)
    ratings_df, movies_df, users_df = load_raw_data()
    print(f"Loaded {len(ratings_df)} ratings, {len(movies_df)} movies, {len(users_df)} users.")

    print("\n" + "=" * 70)
    print("STEP 2: Cleaning data (missing values, duplicates)")
    print("=" * 70)
    ratings_df, movies_df, users_df = clean_data(ratings_df, movies_df, users_df)

    print("\n" + "=" * 70)
    print("STEP 3: Encoding User/Movie IDs into contiguous indices")
    print("=" * 70)
    ratings_df, user_encoder, movie_encoder = encode_ids(ratings_df)
    print(f"Number of unique users: {len(user_encoder.classes_)}")
    print(f"Number of unique movies: {len(movie_encoder.classes_)}")

    print("\n" + "=" * 70)
    print("STEP 4: Generating implicit feedback labels (rating >= 4 -> 1)")
    print("=" * 70)
    ratings_df = generate_labels(ratings_df)
    positive_ratio = ratings_df["label"].mean()
    print(f"Positive interaction ratio: {positive_ratio:.4f}")

    print("\n" + "=" * 70)
    print("STEP 5: Splitting into train/val/test (80/10/10)")
    print("=" * 70)
    train_df, val_df, test_df = split_data(ratings_df)
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    print("\n" + "=" * 70)
    print("STEP 6: Building per-user positive item lookup")
    print("=" * 70)
    user_positive_items = build_user_positive_items(ratings_df)
    print(f"Built positive-item sets for {len(user_positive_items)} users.")

    print("\n" + "=" * 70)
    print("STEP 7: Attaching movie_idx to movie metadata")
    print("=" * 70)
    movies_df = attach_movie_idx_to_movies(movies_df, movie_encoder)

    print("\n" + "=" * 70)
    print("STEP 8: Saving processed artifacts to disk")
    print("=" * 70)
    save_processed_data(
        train_df, val_df, test_df, movies_df, user_encoder, movie_encoder, user_positive_items
    )

    print("\nPreprocessing pipeline completed successfully.")


if __name__ == "__main__":
    run_preprocessing_pipeline()
