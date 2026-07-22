import os
import tempfile
from pathlib import Path

# Set up an isolated temporary database for testing
temp_dir = tempfile.mkdtemp()
temp_db_path = Path(temp_dir) / "test_clashroyale.db"
os.environ["CLASH_ROYALE_DB_PATH"] = str(temp_db_path)

# Initialize schema in the temporary database
from database.storage import create_database
create_database()

from collector.api_client import ClashRoyaleAPI
from collector.parser import BattleParser
from collector.database import Database

api = ClashRoyaleAPI()
battles = api.get_battle_log("#CG0V8QC9J")

if battles and len(battles) > 0:
    battle = battles[0]
    parsed = BattleParser.parse(battle)

    with Database() as db:
        db.add_battle(parsed)
        db.commit()
        assert db.battle_exists(parsed["battle_id"]) is True
        print("True")
else:
    print("[SKIP] Live API call skipped (no network or invalid API token).")

# Cleanup temporary database
try:
    if temp_db_path.exists():
        temp_db_path.unlink()
    temp_db_path.parent.rmdir()
except Exception as e:
    print(f"Cleanup warning: {e}")