from collector.api_client import ClashRoyaleAPI
from collector.parser import BattleParser
from collector.database import Database

api = ClashRoyaleAPI()

battle = api.get_battle_log("#CG0V8QC9J")[0]

parsed = BattleParser.parse(battle)

with Database() as db:

    db.add_battle(parsed)

    db.commit()

    print(db.battle_exists(parsed["battle_id"]))