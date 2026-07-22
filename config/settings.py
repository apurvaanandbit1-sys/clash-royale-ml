import os
from pathlib import Path

# ==========================
# API
# ==========================

REQUEST_DELAY = 0.15    # seconds
MAX_RETRIES = 5        # Bounded retries
RETRY_DELAY = 3        # Base retry delay in seconds

# ==========================
# DATABASE
# ==========================

# Resolve database path priority:
# 1. Environment variable CLASH_ROYALE_DB_PATH
# 2. Default location in project data directory
env_db_path = os.getenv("CLASH_ROYALE_DB_PATH")
if env_db_path:
    DB_PATH = Path(env_db_path)
else:
    DB_PATH = Path(__file__).resolve().parent.parent / "data" / "clashroyale.db"

# ==========================
# CRAWLER
# ==========================

MAX_PLAYERS = 1000000

MAX_DEPTH = 100

MIN_TROPHIES = 8500

COMMIT_INTERVAL = 1    # Commit per player for crash recovery

# ==========================
# LOGGING
# ==========================

PRINT_EVERY = 10