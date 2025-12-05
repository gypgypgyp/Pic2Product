# scripts/init_db_from_csv.py
import csv
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # go up from scripts/
DB_PATH = BASE_DIR / "catalog" / "catalog.db"
CSV_PATH = BASE_DIR / "catalog" / "catalog.csv"

def get_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_schema(conn):
    conn.execute("""
    CREATE TABLE IF NOT EXISTS products (
        sku_id     TEXT PRIMARY KEY,
        title      TEXT NOT NULL,
        brand      TEXT,
        image_path TEXT NOT NULL,
        source     TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS embeddings (
        sku_id     TEXT PRIMARY KEY,
        embedding  BLOB NOT NULL,
        dim        INTEGER NOT NULL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (sku_id) REFERENCES products(sku_id) ON DELETE CASCADE
    )
    """)
    conn.commit()

def load_csv_into_products(conn):
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            rows.append((
                r["sku_id"],
                r["title"],
                r.get("brand"),
                r["image_path"],
                r.get("source", "csv"),
            ))
    conn.executemany("""
        INSERT OR REPLACE INTO products (sku_id, title, brand, image_path, source)
        VALUES (?,?,?,?,?)
    """, rows)
    conn.commit()

if __name__ == "__main__":
    conn = get_conn()
    init_schema(conn)
    load_csv_into_products(conn)
    conn.close()
    print("✅ catalog.db initialized from catalog.csv")
