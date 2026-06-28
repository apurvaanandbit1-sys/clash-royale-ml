from collector.database import Database

with Database() as db:
    db.clear_battles()
    db.clear_players()
    db.clear_queue()

print("Database reset successfully!")