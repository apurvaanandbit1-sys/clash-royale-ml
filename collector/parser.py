import json
from hashlib import sha256
from datetime import datetime, timezone


class BattleParser:

    @staticmethod
    def _extract_deck(cards):
        """
        Extract card IDs, sort them and return as JSON string.
        """
        deck = sorted(card["id"] for card in cards)
        return json.dumps(deck)

    @staticmethod
    def _extract_card_levels(cards):
        """
        Extract card IDs and their levels, return as a JSON-serialized dictionary.
        """
        levels = {str(card["id"]): card["level"] for card in cards}
        return json.dumps(levels)

    @staticmethod
    def _generate_battle_id(player_tag, opponent_tag, battle_time):
        """
        Generate the same battle ID regardless of which player
        we crawled first.
        """
        players = sorted([player_tag, opponent_tag])

        text = f"{players[0]}_{players[1]}_{battle_time}"

        return sha256(text.encode()).hexdigest()

    @staticmethod
    def parse(battle):

        team = battle["team"][0]
        opponent = battle["opponent"][0]

        # Validate that both players have exactly 8-card decks.
        # Draft modes or special events can return non-8 cards.
        if "cards" not in team or len(team["cards"]) != 8 or "cards" not in opponent or len(opponent["cards"]) != 8:
            return None

        battle_time = battle["battleTime"]

        battle_id = BattleParser._generate_battle_id(
            team["tag"],
            opponent["tag"],
            battle_time
        )

        player_crowns = team.get("crowns", 0)
        opponent_crowns = opponent.get("crowns", 0)

        return {

            "battle_id": battle_id,

            "battle_time": battle_time,

            "battle_mode": battle.get("type", "unknown"),

            "player_tag": team["tag"],
            "player_name": team["name"],
            "player_trophies": team.get("startingTrophies", 0),
            "player_crowns": player_crowns,

            "opponent_tag": opponent["tag"],
            "opponent_name": opponent["name"],
            "opponent_trophies": opponent.get("startingTrophies", 0),
            "opponent_crowns": opponent_crowns,

            "player_deck": BattleParser._extract_deck(team["cards"]),
            "opponent_deck": BattleParser._extract_deck(opponent["cards"]),

            "winner": "player" if player_crowns > opponent_crowns else "opponent",

            "win": int(player_crowns > opponent_crowns),

            "raw_battle_json": json.dumps(battle),

            "player_card_levels": BattleParser._extract_card_levels(team["cards"]),

            "opponent_card_levels": BattleParser._extract_card_levels(opponent["cards"]),

            "player_king_level": None,

            "opponent_king_level": None,

            "arena_name": battle.get("arena", {}).get("name", "unknown"),

            "collector_version": 2,

            "api_fetch_time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        }