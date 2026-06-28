from collector.api_client import ClashRoyaleAPI
from collector.parser import BattleParser

api = ClashRoyaleAPI()

battles = api.get_battle_log("#CG0V8QC9J")

parsed = BattleParser.parse(battles[0])

for k, v in parsed.items():
    print(f"{k}: {v}")