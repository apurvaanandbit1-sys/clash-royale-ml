from pathlib import Path
import sqlite3

# ==========================================================
# Database Configuration
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "clashroyale.db"

# Create data directory if it doesn't exist
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def create_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # ======================================================
    # BATTLES TABLE
    # ======================================================

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS battles (

        battle_id TEXT PRIMARY KEY,

        battle_time TEXT,

        battle_mode TEXT,

        player_tag TEXT,

        opponent_tag TEXT,

        player_name TEXT,

        opponent_name TEXT,

        player_trophies INTEGER,

        opponent_trophies INTEGER,

        player_crowns INTEGER,

        opponent_crowns INTEGER,

        player_deck TEXT,

        opponent_deck TEXT,

        winner TEXT,

        win INTEGER

    );
    """)

    # ======================================================
    # PLAYERS TABLE
    # ======================================================

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS players (

        player_tag TEXT PRIMARY KEY,

        player_name TEXT,

        trophies INTEGER,

        processed INTEGER DEFAULT 0,

        last_crawled TEXT

    );
    """)

    # ======================================================
    # CRAWL QUEUE
    # ======================================================

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS crawl_queue (

        player_tag TEXT PRIMARY KEY,

        depth INTEGER,

        status TEXT

    );
    """)

    # ======================================================
    # INDEXES
    # ======================================================

    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_player_tag
    ON battles(player_tag);
    """)

    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_opponent_tag
    ON battles(opponent_tag);
    """)

    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_battle_time
    ON battles(battle_time);
    """)

    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_processed
    ON players(processed);
    """)

    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_queue_status
    ON crawl_queue(status);
    """)

    conn.commit()
    conn.close()

    print("\n" + "=" * 60)
    print(" Clash Royale Database Initialized Successfully")
    print("=" * 60)
    print(f"Database Location : {DB_PATH}")
    print("\nTables Created:")
    print("   • battles")
    print("   • players")
    print("   • crawl_queue")
    print("\nIndexes Created:")
    print("   • idx_player_tag")
    print("   • idx_opponent_tag")
    print("   • idx_battle_time")
    print("   • idx_processed")
    print("   • idx_queue_status")
    print("=" * 60)


if __name__ == "__main__":
    create_database()