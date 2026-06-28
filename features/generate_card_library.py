import json
from pathlib import Path

from collector.api_client import ClashRoyaleAPI

OUTPUT = Path(__file__).parent / "card_library.json"

api = ClashRoyaleAPI()

cards = api.get_cards()

library = {}

for card in cards:

    library[str(card["id"])] = {
        "id": card["id"],
        "name": card["name"],
        "elixir": card.get("elixirCost"),
        "rarity": card.get("rarity"),
        "max_level": card.get("maxLevel"),
        "max_evolution_level": card.get("maxEvolutionLevel", 0),
        "icon_url": card.get("iconUrls", {}).get("medium"),
        "hero_icon_url": card.get("iconUrls", {}).get("heroMedium"),
        "evolution_icon_url": card.get("iconUrls", {}).get("evolutionMedium"),
    }

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(library, f, indent=4, ensure_ascii=False)

print("=" * 60)
print(f"Saved {len(library)} cards.")
print(f"Output : {OUTPUT}")
print("=" * 60)