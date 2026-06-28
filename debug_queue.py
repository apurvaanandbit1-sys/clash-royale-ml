# debug_queue.py

import sqlite3
from pathlib import Path

DB = Path("data/clashroyale.db")

conn = sqlite3.connect(DB)
cur = conn.cursor()

cur.execute("SELECT * FROM crawl_queue")

rows = cur.fetchall()

print(rows)

conn.close()