import sqlite3
from pathlib import Path

try:
    from config.settings import DB_PATH
except ImportError:
    BASE_DIR = Path(__file__).resolve().parent.parent
    DB_PATH = BASE_DIR / "data" / "clashroyale.db"


class Database:
    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()

    def commit(self):
        self.conn.commit()

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):

        if exc_type is None:
            self.commit()
        else:
            self.conn.rollback()

        self.close()
        # ======================================================
    # PLAYER METHODS
    # ======================================================

    def add_player(self, tag, name="", trophies=0):

        self.cursor.execute(
            """
            INSERT OR IGNORE INTO players
            (
                player_tag,
                player_name,
                trophies
            )
            VALUES (?, ?, ?)
            """,
            (tag, name, trophies)
        )

    def player_exists(self, tag):

        self.cursor.execute(
            """
            SELECT 1
            FROM players
            WHERE player_tag=?
            """,
            (tag,)
        )

        return self.cursor.fetchone() is not None

    def mark_processed(self, tag):

        self.cursor.execute(
            """
            UPDATE players
            SET processed=1,
                last_crawled=datetime('now')
            WHERE player_tag=?
            """,
            (tag,)
        )

        # ======================================================
    # QUEUE METHODS
    # ======================================================

    def recover_queue(self):
        """
        On startup, reset all 'processing' nodes back to 'pending'.
        This ensures interrupted runs can resume correctly.
        """
        self.cursor.execute(
            """
            UPDATE crawl_queue
            SET status='pending'
            WHERE status='processing'
            """
        )

    def enqueue_player(self, tag, depth=0):

        self.cursor.execute(
            """
            INSERT OR IGNORE INTO crawl_queue
            (
                player_tag,
                depth,
                status
            )

            VALUES
            (
                ?,
                ?,
                'pending'
            )
            """,
            (tag, depth)
        )

    def get_next_player(self):

        self.cursor.execute(
            """
            SELECT player_tag, depth

            FROM crawl_queue

            WHERE status='pending'

            ORDER BY depth ASC

            LIMIT 1
            """
        )

        row = self.cursor.fetchone()

        if row is None:
            return None

        return dict(row)

    def set_processing(self, tag):

        self.cursor.execute(
            """
            UPDATE crawl_queue

            SET status='processing'

            WHERE player_tag=?
            """,
            (tag,)
        )

    def finish_player(self, tag):

        self.cursor.execute(
            """
            UPDATE crawl_queue

            SET status='done'

            WHERE player_tag=?
            """,
            (tag,)
        )

        # ======================================================
    # BATTLE METHODS
    # ======================================================

    def battle_exists(self, battle_id):

        self.cursor.execute(
            """
            SELECT 1
            FROM battles
            WHERE battle_id=?
            """,
            (battle_id,)
        )

        return self.cursor.fetchone() is not None


    def add_battle(self, battle):

        self.cursor.execute(
            """
            INSERT OR IGNORE INTO battles
            (
                battle_id,
                battle_time,
                battle_mode,

                player_tag,
                opponent_tag,

                player_name,
                opponent_name,

                player_trophies,
                opponent_trophies,

                player_crowns,
                opponent_crowns,

                player_deck,
                opponent_deck,

                winner,
                win,

                raw_battle_json,
                player_card_levels,
                opponent_card_levels,
                player_king_level,
                opponent_king_level,
                arena_name,
                collector_version,
                api_fetch_time
            )

            VALUES
            (
                :battle_id,
                :battle_time,
                :battle_mode,

                :player_tag,
                :opponent_tag,

                :player_name,
                :opponent_name,

                :player_trophies,
                :opponent_trophies,

                :player_crowns,
                :opponent_crowns,

                :player_deck,
                :opponent_deck,

                :winner,
                :win,

                :raw_battle_json,
                :player_card_levels,
                :opponent_card_levels,
                :player_king_level,
                :opponent_king_level,
                :arena_name,
                :collector_version,
                :api_fetch_time
            )
            """,
            battle
        )

        
    def add_battles(self, battles):
        """
        Insert multiple battles in one transaction.
        """

        self.cursor.executemany(
            """
            INSERT OR IGNORE INTO battles
            (
                battle_id,
                battle_time,
                battle_mode,

                player_tag,
                opponent_tag,

                player_name,
                opponent_name,

                player_trophies,
                opponent_trophies,

                player_crowns,
                opponent_crowns,

                player_deck,
                opponent_deck,

                winner,
                win,

                raw_battle_json,
                player_card_levels,
                opponent_card_levels,
                player_king_level,
                opponent_king_level,
                arena_name,
                collector_version,
                api_fetch_time
            )

            VALUES
            (
                :battle_id,
                :battle_time,
                :battle_mode,

                :player_tag,
                :opponent_tag,

                :player_name,
                :opponent_name,

                :player_trophies,
                :opponent_trophies,

                :player_crowns,
                :opponent_crowns,

                :player_deck,
                :opponent_deck,

                :winner,
                :win,

                :raw_battle_json,
                :player_card_levels,
                :opponent_card_levels,
                :player_king_level,
                :opponent_king_level,
                :arena_name,
                :collector_version,
                :api_fetch_time
            )
            """,
            battles
        )

    def get_stats(self):

        stats = {}

        self.cursor.execute("SELECT COUNT(*) FROM battles")
        stats["battles"] = self.cursor.fetchone()[0]

        self.cursor.execute("SELECT COUNT(*) FROM players")
        stats["players"] = self.cursor.fetchone()[0]

        self.cursor.execute("""
            SELECT COUNT(*)
            FROM players
            WHERE processed=1
        """)
        stats["processed"] = self.cursor.fetchone()[0]

        self.cursor.execute("""
            SELECT COUNT(*)
            FROM crawl_queue
            WHERE status='pending'
        """)
        stats["queue"] = self.cursor.fetchone()[0]

        return stats
    
    def clear_players(self):
        self.cursor.execute("DELETE FROM players")

    def clear_battles(self):
        self.cursor.execute("DELETE FROM battles")

    def clear_queue(self):
        self.cursor.execute("DELETE FROM crawl_queue")

    def queue_empty(self):

        self.cursor.execute("""
            SELECT COUNT(*)
            FROM crawl_queue
            WHERE status='pending'
        """)

        return self.cursor.fetchone()[0] == 0