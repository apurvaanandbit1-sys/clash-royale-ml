from features.feature_engine import FeatureEngine


def test_card_library():
    """Verify the integrity of card_library.json."""

    engine = FeatureEngine()

    # -------------------------------
    # Test 1: Total number of cards
    # -------------------------------
    assert len(engine.card_library) == 122, (
        f"Expected 122 cards, found {len(engine.card_library)}"
    )

    # -------------------------------
    # Test 2: Unique IDs
    # -------------------------------
    ids = list(engine.card_library.keys())

    assert len(ids) == len(set(ids)), (
        "Duplicate card IDs found."
    )

    # -------------------------------
    # Test 3: Required fields
    # -------------------------------
    required_fields = {
        "id",
        "name",
        "elixir",
        "rarity",
        "max_level",
        "max_evolution_level",
        "icon_url",
        "hero_icon_url",
        "evolution_icon_url",
    }

    for card_id, card in engine.card_library.items():

        missing = required_fields - set(card.keys())

        assert not missing, (
            f"{card['name']} is missing fields: {missing}"
        )
        # -------------------------------
        # Test 4: Valid elixir cost
        # -------------------------------

        if card["name"] == "Mirror":
            assert card["elixir"] is None, (
                "Mirror should have elixir = None."
            )
        else:
            assert isinstance(card["elixir"], int), (
                f"{card['name']} has non-integer elixir."
            )

            assert 0 <= card["elixir"] <= 10, (
                f"{card['name']} has invalid elixir cost: {card['elixir']}"
            )
        # -------------------------------
        # Test 5: Valid max level
        # -------------------------------
        assert card["max_level"] == 16, (
            f"{card['name']} has unexpected max level: {card['max_level']}"
        )

        # -------------------------------
        # Test 6: Non-empty name
        # -------------------------------
        assert card["name"].strip(), (
            f"Card ID {card_id} has an empty name."
        )

    print("[OK] Card library integrity test passed.")

if __name__ == "__main__":
    test_card_library()