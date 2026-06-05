"""
Layer 1 – Database Schema
Creates and manages the SQLite database for the soccer draw probability model.
Tables: leagues, teams, matches, xg_data, live_events, kalshi_contracts
"""

import sqlite3
import os
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "soccer_draw.db"


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def create_schema():
    conn = get_connection()
    c = conn.cursor()

    # ── Leagues ────────────────────────────────────────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS leagues (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            fd_code         TEXT UNIQUE NOT NULL,   -- Football-Data.org code e.g. "PL"
            understat_name  TEXT,                   -- Understat league slug e.g. "EPL"
            display_name    TEXT NOT NULL,
            country         TEXT,
            season          TEXT NOT NULL,          -- "2024/25"
            created_at      TEXT DEFAULT (datetime('now'))
        )
    """)

    # ── Teams ──────────────────────────────────────────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS teams (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            fd_id           INTEGER UNIQUE,         -- Football-Data.org team ID
            understat_id    INTEGER,
            short_name      TEXT NOT NULL,
            full_name       TEXT,
            league_id       INTEGER REFERENCES leagues(id),
            created_at      TEXT DEFAULT (datetime('now'))
        )
    """)

    # ── Matches (completed + scheduled) ────────────────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS matches (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            fd_match_id     INTEGER UNIQUE NOT NULL,
            understat_match_id INTEGER,
            league_id       INTEGER REFERENCES leagues(id),
            home_team_id    INTEGER REFERENCES teams(id),
            away_team_id    INTEGER REFERENCES teams(id),
            match_date      TEXT NOT NULL,          -- ISO 8601
            status          TEXT DEFAULT 'SCHEDULED', -- SCHEDULED|LIVE|FINISHED
            home_score      INTEGER,
            away_score      INTEGER,
            -- Pre-match xG (from Understat historical)
            home_xg         REAL,
            away_xg         REAL,
            -- Result flags for aggregation
            is_draw         INTEGER,                -- 0/1
            -- Signal cache (populated by scorer)
            pre_draw_prob   REAL,
            created_at      TEXT DEFAULT (datetime('now')),
            updated_at      TEXT DEFAULT (datetime('now'))
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(match_date)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_matches_status ON matches(status)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_matches_teams ON matches(home_team_id, away_team_id)")

    # ── Per-match xG data (Understat shot-by-shot) ──────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS xg_data (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id        INTEGER REFERENCES matches(id),
            minute          INTEGER NOT NULL,
            team_side       TEXT NOT NULL,          -- 'home'|'away'
            player          TEXT,
            xg_value        REAL NOT NULL,
            cumulative_home_xg REAL,
            cumulative_away_xg REAL,
            created_at      TEXT DEFAULT (datetime('now'))
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_xg_match ON xg_data(match_id, minute)")

    # ── Live events (score changes + live xG increments during match) ──────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS live_events (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id        INTEGER REFERENCES matches(id),
            event_type      TEXT NOT NULL,          -- 'goal'|'xg_update'|'card'|'sub'
            minute          INTEGER NOT NULL,
            team_side       TEXT,                   -- 'home'|'away'
            home_score      INTEGER,
            away_score      INTEGER,
            live_home_xg    REAL,
            live_away_xg    REAL,
            raw_payload     TEXT,                   -- JSON blob for full event data
            created_at      TEXT DEFAULT (datetime('now'))
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_live_match ON live_events(match_id, minute)")

    # ── League-level draw statistics (rolling window) ──────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS league_draw_stats (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            league_id       INTEGER REFERENCES leagues(id),
            season          TEXT NOT NULL,
            matches_played  INTEGER DEFAULT 0,
            draws           INTEGER DEFAULT 0,
            draw_rate       REAL,                   -- draws / matches_played
            avg_home_xg     REAL,
            avg_away_xg     REAL,
            updated_at      TEXT DEFAULT (datetime('now')),
            UNIQUE(league_id, season)
        )
    """)

    # ── Head-to-head draw stats ────────────────────────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS h2h_stats (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            home_team_id    INTEGER REFERENCES teams(id),
            away_team_id    INTEGER REFERENCES teams(id),
            lookback_games  INTEGER DEFAULT 10,
            matches_counted INTEGER DEFAULT 0,
            draws           INTEGER DEFAULT 0,
            draw_rate       REAL,
            updated_at      TEXT DEFAULT (datetime('now')),
            UNIQUE(home_team_id, away_team_id, lookback_games)
        )
    """)

    # ── Team form (last N matches) ─────────────────────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS team_form (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            team_id         INTEGER REFERENCES teams(id),
            as_of_date      TEXT NOT NULL,
            lookback_games  INTEGER DEFAULT 5,
            wins            INTEGER DEFAULT 0,
            draws           INTEGER DEFAULT 0,
            losses          INTEGER DEFAULT 0,
            goals_scored    INTEGER DEFAULT 0,
            goals_conceded  INTEGER DEFAULT 0,
            xg_scored       REAL DEFAULT 0,
            xg_conceded     REAL DEFAULT 0,
            form_draw_rate  REAL,
            updated_at      TEXT DEFAULT (datetime('now')),
            UNIQUE(team_id, as_of_date, lookback_games)
        )
    """)

    # ── League table (for table-gap signal) ───────────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS league_table (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            league_id       INTEGER REFERENCES leagues(id),
            team_id         INTEGER REFERENCES teams(id),
            season          TEXT NOT NULL,
            as_of_date      TEXT NOT NULL,
            position        INTEGER,
            played          INTEGER DEFAULT 0,
            points          INTEGER DEFAULT 0,
            goal_diff       INTEGER DEFAULT 0,
            updated_at      TEXT DEFAULT (datetime('now')),
            UNIQUE(league_id, team_id, season, as_of_date)
        )
    """)

    # ── Kalshi contracts ───────────────────────────────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS kalshi_contracts (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id        INTEGER REFERENCES matches(id),
            ticker          TEXT UNIQUE NOT NULL,   -- Kalshi market ticker
            market_title    TEXT,
            contract_type   TEXT DEFAULT 'draw',    -- 'draw'|'home'|'away'
            yes_price       REAL,                   -- cents (0-100)
            no_price        REAL,
            volume          INTEGER,
            open_interest   INTEGER,
            fetched_at      TEXT DEFAULT (datetime('now'))
        )
    """)

    # ── Trade alerts log ───────────────────────────────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS trade_alerts (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id        INTEGER REFERENCES matches(id),
            kalshi_ticker   TEXT,
            checkpoint_min  INTEGER NOT NULL,       -- 60|70|80
            model_prob      REAL NOT NULL,
            contract_price  REAL NOT NULL,          -- implied probability (0-1)
            edge            REAL NOT NULL,           -- model_prob - contract_price
            action          TEXT NOT NULL,           -- 'BUY_YES'|'BUY_NO'|'PASS'
            blend_weights   TEXT,                   -- JSON {pre_game: 0.3, live: 0.7}
            pre_game_prob   REAL,
            live_prob       REAL,
            fired_at        TEXT DEFAULT (datetime('now'))
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_alerts_match ON trade_alerts(match_id)")

    conn.commit()
    conn.close()
    print(f"[schema] Database created at {DB_PATH}")


def seed_leagues():
    """Seed the supported leagues with Football-Data + Understat mappings."""
    leagues = [
        ("PL",  "EPL",        "Premier League",       "England",     "2024/25"),
        ("PD",  "La_liga",    "La Liga",              "Spain",       "2024/25"),
        ("BL1", "Bundesliga", "Bundesliga",           "Germany",     "2024/25"),
        ("SA",  "Serie_A",    "Serie A",              "Italy",       "2024/25"),
        ("FL1", "Ligue_1",    "Ligue 1",              "France",      "2024/25"),
        ("PPL", None,         "Primeira Liga",        "Portugal",    "2024/25"),
        ("DED", None,         "Eredivisie",           "Netherlands", "2024/25"),
    ]
    conn = get_connection()
    c = conn.cursor()
    c.executemany("""
        INSERT OR IGNORE INTO leagues (fd_code, understat_name, display_name, country, season)
        VALUES (?, ?, ?, ?, ?)
    """, leagues)
    conn.commit()
    conn.close()
    print(f"[schema] Seeded {len(leagues)} leagues")


if __name__ == "__main__":
    create_schema()
    seed_leagues()
    print("[schema] Setup complete.")
