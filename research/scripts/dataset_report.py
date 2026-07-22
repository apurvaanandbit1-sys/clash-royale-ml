import os
import sys
import sqlite3
import json
from pathlib import Path
from datetime import datetime

# Resolve paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from config.settings import DB_PATH

def generate_report():
    if not DB_PATH.exists():
        print(f"Error: Database not found at {DB_PATH}")
        sys.exit(1)

    print("=" * 60)
    print(f"CLASH ROYALE DATASET HEALTH REPORT")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Connect to the database read-only
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # ---------------------------------------------------------
    # 1. GENERAL STATISTICS
    # ---------------------------------------------------------
    cursor.execute("SELECT COUNT(*) FROM battles")
    total_battles = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM players")
    total_players = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM crawl_queue")
    total_queue = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM crawl_queue WHERE status='pending'")
    pending_queue = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM crawl_queue WHERE status='processing'")
    processing_queue = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM crawl_queue WHERE status='done'")
    completed_queue = cursor.fetchone()[0]

    db_size_mb = os.path.getsize(DB_PATH) / (1024 * 1024)

    # Calculate crawl duration and average crawl speed
    cursor.execute("SELECT MIN(last_crawled), MAX(last_crawled) FROM players WHERE processed=1")
    min_time_str, max_time_str = cursor.fetchone()
    
    crawl_duration_mins = 0.0
    avg_crawl_speed_battles_pm = 0.0
    avg_crawl_speed_players_pm = 0.0
    
    if min_time_str and max_time_str:
        try:
            min_time = datetime.strptime(min_time_str.split('.')[0], "%Y-%m-%d %H:%M:%S")
            max_time = datetime.strptime(max_time_str.split('.')[0], "%Y-%m-%d %H:%M:%S")
            duration = max_time - min_time
            crawl_duration_mins = duration.total_seconds() / 60.0
            if crawl_duration_mins > 0:
                avg_crawl_speed_battles_pm = total_battles / crawl_duration_mins
                avg_crawl_speed_players_pm = completed_queue / crawl_duration_mins
        except Exception as e:
            pass

    print("\nGENERAL STATISTICS")
    print("-" * 30)
    print(f"Total Battles:              {total_battles:,}")
    print(f"Unique Players:             {total_players:,}")
    print(f"Database File Size:         {db_size_mb:.2f} MB")
    print(f"Estimated Crawl Duration:   {crawl_duration_mins:.2f} minutes")
    print(f"Average Crawl Speed:        {avg_crawl_speed_battles_pm:.2f} battles/min ({avg_crawl_speed_players_pm:.2f} players/min)")
    print(f"Queue Status Summary:")
    print(f"  • Pending:                {pending_queue:,}")
    print(f"  • Processing:             {processing_queue:,}")
    print(f"  • Completed:              {completed_queue:,}")
    print(f"  • Total Queue Size:       {total_queue:,}")

    # ---------------------------------------------------------
    # 2. QUALITY METRICS
    # ---------------------------------------------------------
    # Estimate duplicate rate
    # Standard player log gives ~25 battles.
    expected_battles = completed_queue * 25
    duplicate_battle_rate = 0.0
    if expected_battles > 0:
        duplicate_battle_rate = max(0.0, 1.0 - (total_battles / expected_battles))

    # Duplicate players rate: unique players vs total players ever enqueued
    duplicate_player_rate = 0.0
    if total_queue > 0:
        duplicate_player_rate = max(0.0, 1.0 - (total_players / total_queue))

    print("\nDATA QUALITY METRICS")
    print("-" * 30)
    print(f"Estimated Duplicate Battle Rate: {duplicate_battle_rate * 100:.2f}%")
    print(f"Duplicate Player Tag Rate:       {duplicate_player_rate * 100:.2f}%")
    print(f"Failed API requests:             See logs")

    # ---------------------------------------------------------
    # 3. GAMEPLAY DISTRIBUTIONS
    # ---------------------------------------------------------
    print("\nGAMEPLAY DISTRIBUTIONS")
    print("-" * 30)

    # Trophy distribution
    cursor.execute("SELECT AVG(player_trophies), MIN(player_trophies), MAX(player_trophies) FROM battles")
    avg_tr, min_tr, max_tr = cursor.fetchone()
    print(f"Trophy Range:               {min_tr:,} - {max_tr:,} (Avg: {int(avg_tr or 0):,})")

    # Win/Loss Balance
    cursor.execute("SELECT win, COUNT(*) FROM battles GROUP BY win")
    win_balance = {r[0]: r[1] for r in cursor.fetchall()}
    win_p = (win_balance.get(1, 0) / total_battles) * 100 if total_battles > 0 else 0
    loss_p = (win_balance.get(0, 0) / total_battles) * 100 if total_battles > 0 else 0
    print(f"Win/Loss Balance (Player):  Win: {win_p:.1f}% ({win_balance.get(1, 0):,}) | Loss: {loss_p:.1f}% ({win_balance.get(0, 0):,})")

    # Arena distribution
    cursor.execute("SELECT arena_name, COUNT(*) FROM battles GROUP BY arena_name ORDER BY COUNT(*) DESC LIMIT 5")
    arenas = cursor.fetchall()
    print(f"Top 5 Arenas:")
    for arena in arenas:
        pct = (arena[1] / total_battles) * 100 if total_battles > 0 else 0
        print(f"  • {arena[0]}: {pct:.1f}% ({arena[1]:,})")

    # Load card library for matching name and champion/evolution flags
    card_lib_path = PROJECT_ROOT / "features" / "card_library.json"
    card_names = {}
    champions = set()
    evos = set()
    
    if card_lib_path.exists():
        with open(card_lib_path, "r") as f:
            lib = json.load(f)
            for cid, card in lib.items():
                card_names[int(cid)] = card.get("name", "Unknown")
                # Champions have hero_icon_url or special rarity
                if card.get("rarity") == "champion":
                    champions.add(int(cid))
                if card.get("max_evolution_level", 0) > 0:
                    evos.add(int(cid))

    # Parse decks and collect card/champion/evolution counts
    cursor.execute("SELECT player_deck, opponent_deck FROM battles")
    rows = cursor.fetchall()
    
    card_counts = {}
    champion_decks_count = 0
    evolution_decks_count = 0
    total_elixir_sum = 0
    total_cards_count = 0

    for row in rows:
        p_deck = json.loads(row["player_deck"])
        o_deck = json.loads(row["opponent_deck"])
        
        for deck in [p_deck, o_deck]:
            has_champion = False
            has_evolution = False
            
            for cid in deck:
                card_counts[cid] = card_counts.get(cid, 0) + 1
                if cid in champions:
                    has_champion = True
                
                # Check if card is evolved from the raw_battle_json or check database.
                # Since evolution card IDs in API are same but marked with extraLevel/evolutionLevel,
                # we can approximate if the card can evolve and check raw json, or just count potential evos.
                
            if has_champion:
                champion_decks_count += 1

    # Top 5 most played cards
    sorted_cards = sorted(card_counts.items(), key=lambda x: x[1], reverse=True)
    print(f"Top 5 Most Played Cards:")
    for cid, count in sorted_cards[:5]:
        name = card_names.get(cid, f"ID {cid}")
        pct = (count / (total_battles * 2)) * 100 if total_battles > 0 else 0
        print(f"  • {name}: {pct:.1f}% ({count:,} appearances)")

    print(f"Champion Usage Rate:        {(champion_decks_count / (total_battles * 2)) * 100:.2f}% of decks")

    # ---------------------------------------------------------
    # 4. DATABASE HEALTH
    # ---------------------------------------------------------
    print("\nDATABASE HEALTH")
    print("-" * 30)
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cursor.fetchall()]
    
    for tbl in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {tbl}")
        t_count = cursor.fetchone()[0]
        print(f"Table '{tbl}': {t_count:,} rows")

    conn.close()
    print("=" * 60)

if __name__ == "__main__":
    generate_report()
