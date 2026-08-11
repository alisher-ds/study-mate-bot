import sqlite3
from datetime import datetime
from config import DB_NAME


def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            ism TEXT,
            telefon TEXT,
            shahar TEXT,
            registered_at TEXT
        )
    """)
    conn.commit()
    conn.close()


def add_user(user_id: int, ism: str, telefon: str, shahar: str):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO users (user_id, ism, telefon, shahar, registered_at)
        VALUES (?, ?, ?, ?, ?)
    """, (user_id, ism, telefon, shahar, datetime.now().strftime("%Y-%m-%d %H:%M")))
    conn.commit()
    conn.close()


def get_user(user_id: int):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
    user = cursor.fetchone()
    conn.close()
    return user


init_db()
