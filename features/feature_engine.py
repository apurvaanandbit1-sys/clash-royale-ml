import json
from pathlib import Path
from typing import Dict, List, Any


KNOWLEDGE_PATH = Path(__file__).parent / "knowledge" / "card_knowledge.json"

CARD_LIBRARY_PATH = Path(__file__).parent / "card_library.json"

HP_SCORE = {
    "VERY_LOW": 1,
    "LOW": 2,
    "MEDIUM": 3,
    "HIGH": 4,
    "VERY_HIGH": 5,
}

DPS_SCORE = {
    "VERY_LOW": 1,
    "LOW": 2,
    "MEDIUM": 3,
    "HIGH": 4,
    "VERY_HIGH": 5,
}

BIG_SPELLS = {
    # Heavy damage/removal spells that typically occupy the
    # deck's primary spell slot.
    "Fireball",
    "Poison",
    "Rocket",
    "Lightning",
    "Void",
}

SMALL_SPELLS = {
    # Cheap utility spells primarily used for swarm control
    # and cycle support.
    "Zap",
    "The Log",
    "Barbarian Barrel",
    "Giant Snowball",
    "Rage",
    "Arrows",
}


def load_card_knowledge() -> Dict[str, Any]:
    """Load the card knowledge base from JSON."""

    with open(KNOWLEDGE_PATH, "r", encoding="utf-8") as file:
        return json.load(file)
    
def load_card_library() -> Dict[str, Any]:
    """Load the Clash Royale card library."""

    with open(CARD_LIBRARY_PATH, "r", encoding="utf-8") as file:
        return json.load(file)
    

class FeatureEngine:
    """Main feature extraction engine."""

    def __init__(self):
        self.card_knowledge = load_card_knowledge()
        self.card_library = load_card_library()

    def get_card(self, card_name: str) -> Dict[str, Any]:
        """Return the knowledge entry for a single card."""

        if card_name in self.card_knowledge:
            return self.card_knowledge[card_name]

        # Backward compatibility for legacy key names.
        legacy_names = {
            "Mini P.E.K.K.A": "Mini P.E.K.K.A.",
            "P.E.K.K.A": "P.E.K.K.A.",
        }

        if card_name in legacy_names:
            return self.card_knowledge[legacy_names[card_name]]

        raise KeyError(
            f"Card '{card_name}' is missing from card_knowledge.json"
        )
    def get_library_card(self, card_id: int) -> Dict[str, Any]:
        """Return card information from the card library."""

        return self.card_library[str(card_id)]

    def get_card_name(self, card_id: int) -> str:
        """Return the card name for a given card ID."""

        return self.get_library_card(card_id)["name"]
    
    def get_full_card(self, card_id: int) -> Dict[str, Any]:
        """Return both library and knowledge data for a card."""

        card_name = self.get_card_name(card_id)

        return {
            "library": self.get_library_card(card_id),
            "knowledge": self.get_card(card_name)
        }

    def build_deck(self, deck: List[int]) -> List[Dict[str, Any]]:
        """Convert a deck of card IDs into full card objects."""

        return [self.get_full_card(card_id) for card_id in deck]
    
    def iter_cards(self, cards: List[Dict[str, Any]]):
        """Iterate over resolved card objects."""

        for card in cards:
            yield card

    def compute_average_elixir(self, cards: List[Dict[str, Any]]) -> float:
        """Compute the average elixir cost of an 8-card deck."""

        total_elixir = 0

        for card in cards:
            total_elixir += card["library"]["elixir"]

        return total_elixir / len(cards)
    

    def compute_spell_count(self, cards: List[Dict[str, Any]]) -> int:
        """Count spell cards in a deck."""

        count = 0

        for card in self.iter_cards(cards):
            if card["knowledge"]["structural"]["card_type"] == "SPELL":
                count += 1

        return count
    
    def compute_building_count(self, cards: List[Dict[str, Any]]) -> int:
        """Count building cards in a deck."""

        count = 0

        for card in self.iter_cards(cards):
            if card["knowledge"]["structural"]["card_type"] == "BUILDING":
                count += 1

        return count
    
    def compute_has_champion(self, cards: List[Dict[str, Any]]) -> bool:
        """Check if the deck contains a Champion."""

        for card in self.iter_cards(cards):
            if card["library"]["rarity"].lower() == "champion":
                return True

        return False
    

    def compute_has_big_spell(self, cards: List[Dict[str, Any]]) -> bool:
        """Check if the deck contains a big spell."""

        for card in self.iter_cards(cards):
            if card["library"]["name"] in BIG_SPELLS:
                return True

        return False
    
    def compute_has_small_spell(self, cards: List[Dict[str, Any]]) -> bool:
        """Check if the deck contains a small spell."""

        for card in self.iter_cards(cards):
            if card["library"]["name"] in SMALL_SPELLS:
                return True

        return False
    


   
    def compute_has_evolution(self, cards: List[Dict[str, Any]]) -> bool:
        """Check if the deck contains at least one evolvable card."""

        for card in self.iter_cards(cards):
            if card["knowledge"]["abilities"]["evolution_ability"]:
                return True

        return False

    def compute_air_hitting_count(self, cards: List[Dict[str, Any]]) -> int:
        """Count cards that can attack air."""

        count = 0

        for card in self.iter_cards(cards):
            if card["knowledge"]["combat"]["can_attack_air"]:
                count += 1

        return count
    
    def compute_splash_count(self, cards: List[Dict[str, Any]]) -> int:
        """Count splash damage cards."""

        count = 0

        for card in self.iter_cards(cards):
            if card["knowledge"]["combat"]["splash_damage"]:
                count += 1

        return count
    
    def compute_win_condition_count(self, cards: List[Dict[str, Any]]) -> int:
        """Count win condition cards."""

        count = 0

        for card in self.iter_cards(cards):
            if card["knowledge"]["strategic"]["win_condition"]:
                count += 1

        return count
    
    def compute_durability_index(self, cards: List[Dict[str, Any]]) -> int:
        """Compute deck durability score."""

        total = 0

        for card in self.iter_cards(cards):
            hp = card["knowledge"]["combat"]["hp_level"]
            if hp is not None:
                total += HP_SCORE[hp]

        return total
    
    def compute_damage_index(self, cards: List[Dict[str, Any]]) -> int:
        """Compute deck damage score."""

        total = 0

        for card in self.iter_cards(cards):
            dps = card["knowledge"]["combat"]["dps_level"]
            if dps is not None:
                total += DPS_SCORE[dps]

        return total

    def extract_features(self, deck: List[int]) -> Dict[str, Any]:
        """
        Extract all engineered features from an 8-card deck.
        """

        cards = self.build_deck(deck)

        features = {
            "average_elixir": self.compute_average_elixir(cards),
            "spell_count": self.compute_spell_count(cards),
            "building_count": self.compute_building_count(cards),

            "has_evolution": self.compute_has_evolution(cards),
            "has_champion": self.compute_has_champion(cards),

            "has_big_spell": self.compute_has_big_spell(cards),
            "has_small_spell": self.compute_has_small_spell(cards),

            "air_hitting_count": self.compute_air_hitting_count(cards),
            "splash_count": self.compute_splash_count(cards),
            "win_condition_count": self.compute_win_condition_count(cards),

            "durability_index": self.compute_durability_index(cards),
            "damage_index": self.compute_damage_index(cards),
        }

        return features
