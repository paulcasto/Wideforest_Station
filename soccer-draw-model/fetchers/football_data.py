"""
Layer 1 – Football-Data.org Fetcher
Fetches: scheduled/completed matches, standings (league table)
API docs: https://www.football-data.org/documentation/quickstart
Free tier: 10 calls/min, select competitions.
Set FD_API_KEY env var.
"""

import os
import time
import json
import sqlite3
import logging
from datetime import date, timedelta
from typing import Optional

import requests

from db.schema import get_connection

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [FD] %(levelname)s %(message)s")

FD_BASE = "https://api.football-data.org/v4"
FD_KEY  = os.getenv("FD_API_KEY", "DEMO")        # set your key

HEADERS = {
    "X-Auth-Token": FD_KEY,
    "Accept": "application/json",
}

# Rate limiting
_last_call = 0.0
RATE_LIMIT_SECS = 6.1   # 10 req/min safe margin


def _get(path: str, params: dict = None) -> dict:
    global _last_call
    elapsed = time.monotonic() - _last_call
    if elapsed < RATE_LIMIT_SECS:
        time.sleep(RATE_LIMIT_SECS - elapsed)
    _last_call = time.monotonic()

    url = FD_BASE + path
    r = requests.get(url, headers=HEADERS, params=params, timeout=15)
    r.raise_for_status()
    return r.json()


# ── Matches ────────────────────────────────────────────────────────────────────

def fetch_matches_for_league(fd_code: str, season_year: int = 2024) -> int:
    """
    Pull all matches for a competition+season and upsert into DB.
    Returns count of matches inserted/updated.
    """
    conn = get_connection()
    c = conn.cursor()

    # Resolve league_id
    c.execute("SELECT id FROM leagues WHERE fd_code=?", (fd_code,))
    row = c.fetchone()
    if not row:
        logger.warning("Unknown league fd_code: %s", fd_code)
        conn.close()
        return 0
    league_id = row["id"]

    logger.info("Fetching matches: %s season %s", fd_code, season_year)
    data = _get(f"/competitions/{fd_code}/matches", {"season": season_year})

    count = 0
    for m in data.get("matches", []):
        home_id = _upsert_team(c, m["homeTeam"], league_id)
        away_id = _upsert_team(c, m["awayTeam"], league_id)

        score   = m.get("score", {})
        ft      = score.get("fullTime", {})
        h_score = ft.get("home")
        a_score = ft.get("away")
        status  = m.get("status", "SCHEDULED")

        is_draw = None
        if h_score is not None and a_score is not None and status == "FINISHED":
            is_draw = 1 if h_score == a_score else 0

        c.execute("""
            INSERT INTO matches
                (fd_match_id, league_id, home_team_id, away_team_id,
                 match_date, status, home_score, away_score, is_draw)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(fd_match_id) DO UPDATE SET
                status      = excluded.status,
                home_score  = excluded.home_score,
                away_score  = excluded.away_score,
                is_draw     = excluded.is_draw,
                updated_at  = datetime('now')
        """, (
            m["id"], league_id, home_id, away_id,
            m["utcDate"], status, h_score, a_score, is_draw
        ))
        count += 1

    conn.commit()
    conn.close()
    logger.info("Upserted %d matches for %s", count, fd_code)
    return count


def _upsert_team(c: sqlite3.Cursor, team_dict: dict, league_id: int) -> int:
    """Insert or ignore team; return internal id."""
    fd_id      = team_dict.get("id")
    short_name = team_dict.get("shortName") or team_dict.get("name", "Unknown")
    full_name  = team_dict.get("name", short_name)

    c.execute("""
        INSERT INTO teams (fd_id, short_name, full_name, league_id)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(fd_id) DO UPDATE SET
            short_name = excluded.short_name,
            full_name  = excluded.full_name
    """, (fd_id, short_name, full_name, league_id))

    c.execute("SELECT id FROM teams WHERE fd_id=?", (fd_id,))
    return c.fetchone()["id"]


# ── Standings (league table) ───────────────────────────────────────────────────

def fetch_standings(fd_code: str, season_year: int = 2024) -> int:
    """Fetch current standings and upsert into league_table."""
    conn = get_connection()
    c = conn.cursor()

    c.execute("SELECT id FROM leagues WHERE fd_code=?", (fd_code,))
    row = c.fetchone()
    if not row:
        conn.close()
        return 0
    league_id = row["id"]

    logger.info("Fetching standings: %s", fd_code)
    data = _get(f"/competitions/{fd_code}/standings", {"season": season_year})

    today     = date.today().isoformat()
    season    = f"{season_year}/{str(season_year + 1)[-2:]}"
    count     = 0

    for stage in data.get("standings", []):
        if stage.get("type") != "TOTAL":
            continue
        for entry in stage.get("table", []):
            team_fd_id = entry["team"]["id"]
            c.execute("SELECT id FROM teams WHERE fd_id=?", (team_fd_id,))
            team_row = c.fetchone()
            if not team_row:
                continue
            team_id = team_row["id"]

            c.execute("""
                INSERT INTO league_table
                    (league_id, team_id, season, as_of_date, position, played, points, goal_diff)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(league_id, team_id, season, as_of_date) DO UPDATE SET
                    position   = excluded.position,
                    played     = excluded.played,
                    points     = excluded.points,
                    goal_diff  = excluded.goal_diff,
                    updated_at = datetime('now')
            """, (
                league_id, team_id, season, today,
                entry["position"], entry["playedGames"],
                entry["points"],
                entry.get("goalDifference", 0)
            ))
            count += 1

    conn.commit()
    conn.close()
    logger.info("Upserted %d table rows for %s", count, fd_code)
    return count


# ── Live match state ───────────────────────────────────────────────────────────

def fetch_live_matches(fd_code: str) -> list:
    """
    Poll in-progress matches for a competition.
    Returns list of dicts with live score + minute.
    """
    data = _get(f"/competitions/{fd_code}/matches", {"status": "IN_PLAY,PAUSED"})
    results = []
    for m in data.get("matches", []):
        score = m.get("score", {}).get("fullTime", {})
        results.append({
            "fd_match_id": m["id"],
            "minute":      m.get("minute", 0),
            "home_score":  score.get("home", 0),
            "away_score":  score.get("away", 0),
            "status":      m.get("status"),
        })
    return results


def update_live_scores(fd_code: str) -> int:
    """Poll live scores and write live_events rows for changed scores."""
    conn = get_connection()
    c    = conn.cursor()
    live = fetch_live_matches(fd_code)
    count = 0

    for m in live:
        c.execute("SELECT id, home_score, away_score FROM matches WHERE fd_match_id=?",
                  (m["fd_match_id"],))
        row = c.fetchone()
        if not row:
            continue
        match_id = row["id"]

        prev_h = row["home_score"] or 0
        prev_a = row["away_score"] or 0
        new_h  = m["home_score"]
        new_a  = m["away_score"]

        # Detect goal events
        if new_h > prev_h:
            for _ in range(new_h - prev_h):
                c.execute("""
                    INSERT INTO live_events
                        (match_id, event_type, minute, team_side, home_score, away_score)
                    VALUES (?, 'goal', ?, 'home', ?, ?)
                """, (match_id, m["minute"], new_h, new_a))
                count += 1
        if new_a > prev_a:
            for _ in range(new_a - prev_a):
                c.execute("""
                    INSERT INTO live_events
                        (match_id, event_type, minute, team_side, home_score, away_score)
                    VALUES (?, 'goal', ?, 'away', ?, ?)
                """, (match_id, m["minute"], new_h, new_a))
                count += 1

        # Update match row
        c.execute("""
            UPDATE matches SET home_score=?, away_score=?, status=?, updated_at=datetime('now')
            WHERE id=?
        """, (new_h, new_a, m["status"], match_id))

    conn.commit()
    conn.close()
    return count


# ── Batch bootstrap ────────────────────────────────────────────────────────────

def bootstrap_all(leagues: list = None, season_year: int = 2024):
    """Full initial load: matches + standings for all supported leagues."""
    if leagues is None:
        leagues = ["PL", "PD", "BL1", "SA", "FL1"]

    for fd_code in leagues:
        try:
            fetch_matches_for_league(fd_code, season_year)
            fetch_standings(fd_code, season_year)
        except requests.HTTPError as e:
            logger.error("HTTPError for %s: %s", fd_code, e)
        except Exception as e:
            logger.error("Unexpected error for %s: %s", fd_code, e)


if __name__ == "__main__":
    from db.schema import create_schema, seed_leagues
    create_schema()
    seed_leagues()
    bootstrap_all(["PL"])   # start with PL for testing
