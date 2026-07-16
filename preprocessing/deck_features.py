from __future__ import annotations

import json

import pandas as pd

from features.feature_engine import FeatureEngine


def add_deck_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered deck features for both players to the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'player_deck' and 'opponent_deck' columns.

    Returns
    -------
    pd.DataFrame
        Original dataframe with engineered deck features appended.
    """

    engine = FeatureEngine()


    unique_decks = (
        pd.concat(
            [
                df["player_deck"],
                df["opponent_deck"],
            ]
        )
        .drop_duplicates()
        .tolist()
    )

    lookup = _build_feature_lookup(
        engine,
        unique_decks,
    )

    df = _merge_feature_lookup(
        df,
        lookup,
        "player_deck",
        "p1_",
    )

    df = _merge_feature_lookup(
        df,
        lookup,
        "opponent_deck",
        "p2_",
    )

    return df



def _build_feature_lookup(
    engine: FeatureEngine,
    unique_decks: list[str],
) -> dict[str, dict]:
    """
    Compute engineered features once for every unique deck.
    """

    lookup: dict[str, dict] = {}

    for deck in unique_decks:
        card_ids = json.loads(deck)

        if len(card_ids) != 8:
            raise ValueError(
                f"Expected an 8-card deck, got {len(card_ids)} cards: {deck}"
            )

        lookup[deck] = engine.extract_features(card_ids)
    return lookup


def _merge_feature_lookup(
    df: pd.DataFrame,
    lookup: dict[str, dict],
    deck_column: str,
    prefix: str,
) -> pd.DataFrame:
    """
    Merge engineered deck features into the dataframe.
    """

    feature_df = (
        df[deck_column]
        .map(lookup)
        .apply(pd.Series)
        .add_prefix(prefix)
    )

    return pd.concat([df, feature_df], axis=1)