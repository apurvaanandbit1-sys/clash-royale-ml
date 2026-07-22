import time

from config.settings import (
    REQUEST_DELAY,
    MAX_DEPTH,
    MIN_TROPHIES,
    COMMIT_INTERVAL,
    PRINT_EVERY,
)


from pathlib import Path

from collector.api_client import ClashRoyaleAPI
from collector.database import Database
from collector.parser import BattleParser

SEED_FILE = Path(__file__).resolve().parent.parent / "config" / "seed_players.txt"


# ==========================================================
# Load Seed Players
# ==========================================================

def load_seed_players():

    players = []

    with open(SEED_FILE, "r") as f:

        for line in f:

            tag = line.strip()

            if tag:
                players.append(tag)

    return players


# ==========================================================
# Initialize Queue
# ==========================================================

def initialize_queue(db):

    if not db.queue_empty():
        print("Queue already initialized.")
        return

    print("Initializing queue...")

    for tag in load_seed_players():
        db.enqueue_player(tag)

    db.commit()

    print("Seed players added.")


# ==========================================================
# Main
# ==========================================================

def main():

    print("=" * 60)
    print("Clash Royale Battle Collector")
    print("=" * 60)

    api = ClashRoyaleAPI()

    with Database() as db:

        initialize_queue(db)
        
        # Task 3: Reset any stuck 'processing' tags back to 'pending'
        print("Recovering queue state from any previous interruptions...")
        db.recover_queue()
        db.commit()

        processed = 0

        while True:

            player = db.get_next_player()

            if player is None:
                print("\nQueue Empty.")
                break
            tag = player["player_tag"]
            depth = player["depth"]

            print("\n" + "=" * 60)
            print(f"Processing Player : {tag}")
            print(f"Crawl Depth      : {depth}")

            db.set_processing(tag)
            db.commit()  # Save queue state immediately before network operations

            # Task 2: Handle bounded retry exhaustion
            try:
                battles = api.get_battle_log(tag)
            except Exception as e:
                print(f"CRITICAL: Bounded retries exhausted for {tag}: {e}")
                # Reset this player back to pending so they can be retried on next start
                db.cursor.execute(
                    "UPDATE crawl_queue SET status='pending' WHERE player_tag=?",
                    (tag,)
                )
                db.commit()
                print("Database state committed. Terminating collector loop.")
                raise e

            # Task 1: Handle failed / None responses gracefully
            if battles is None:
                print(f"WARNING: Failed to download battles for {tag}. Skipping player.")
                db.finish_player(tag)
                db.mark_processed(tag)
                db.commit()
                processed += 1
                time.sleep(REQUEST_DELAY)
                continue

            print(f"Downloaded {len(battles)} battles")

            parsed_battles = []

            for battle in battles:

                try:

                    parsed = BattleParser.parse(battle)
                    if parsed is None:
                        continue
                    parsed_battles.append(parsed)

                    db.add_player(
                        parsed["player_tag"],
                        parsed["player_name"],
                        parsed["player_trophies"]
                    )

                    db.add_player(
                        parsed["opponent_tag"],
                        parsed["opponent_name"],
                        parsed["opponent_trophies"]
                    )

                    # -------- FILTER --------
                    if (
                        parsed["opponent_trophies"] >= MIN_TROPHIES
                        and depth < MAX_DEPTH
                    ):

                        db.enqueue_player(
                            parsed["opponent_tag"],
                            depth + 1
                        )
                except Exception as e:

                    print(f"Parser Error for {tag}: {e}")

            db.add_battles(parsed_battles)

            db.finish_player(tag)
            print(f"Finished {tag}")

            db.mark_processed(tag)

            processed += 1

            if processed % COMMIT_INTERVAL == 0:
                db.commit()

            stats = db.get_stats()

            print("\nCurrent Statistics")

            print("=" * 60)
            print(f"Players Processed : {stats['processed']}")
            print(f"Players Stored    : {stats['players']}")
            print(f"Battles Stored    : {stats['battles']}")
            print(f"Queue Remaining   : {stats['queue']}")
            print("=" * 60)

            time.sleep(REQUEST_DELAY)

        db.commit()

if __name__ == "__main__":
    main()