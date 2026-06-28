import json
from pathlib import Path

# ==========================================================
# Load Card Library
# ==========================================================

CARD_LIBRARY_PATH = (
    Path(__file__).resolve().parent.parent / "card_library.json"
)

with open(CARD_LIBRARY_PATH, "r", encoding="utf-8") as f:
    CARD_LIBRARY = json.load(f)


# ==========================================================
# Internal Helper
# ==========================================================

def _card(card_id):
    """
    Returns the card dictionary.

    Raises ValueError if the card doesn't exist.
    """

    card = CARD_LIBRARY.get(str(card_id))

    if card is None:
        raise ValueError(f"Unknown card id: {card_id}")

    return card

# ==========================================================
# Basic Properties
# ==========================================================

def name(card_id):
    return _card(card_id)["name"]


def rarity(card_id):
    return _card(card_id)["rarity"]


def elixir(card_id):
    return _card(card_id)["elixir"]


def max_level(card_id):
    return _card(card_id)["max_level"]


# ==========================================================
# Structural Properties
# ==========================================================

def has_evolution(card_id):
    return _card(card_id)["max_evolution_level"] > 0


def is_champion(card_id):
    return rarity(card_id) == "champion"