"""
Layer 1 – Understat xG Fetcher
Fetches shot-level xG data from Understat (unofficial API via HTML scraping).
Understat embeds JSON in <script> tags – no API key required.
Populates: xg_data, league_draw_stats, team_form tables.
"""

import re
import json
import time
import logging
from datetime import date, datetime
from typing import Optional

import requests

from db.schema import get_connection

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [US] %(levelname)s %(message)s")

US_BASE = "https://understat.com"

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (compatible; SoccerDrawModel/1.0)",
    "Accept-Language": "en-US,en;q=0.9",
})

RATE_LIMIT = 2.0   # seconds between requests (be polite)
_last_call = 0.0


def _fetch_html(url: str) -> str:
    global _last_call
    elapsed = time.monotonic() - _last_call
    if elapsed < RATE_LIMIT:
        time.sleep(RATE_LIMIT - elapsed)
    _last_call = time.monotonic()
    r = SESSION.get(url, timeout=20)
    r.raise_for_status()
    return r.text


def _extract_json(html: str, var_name: str):
    """Extract JSON from Understat's embedded JavaScript variable."""
    pattern = rf"var\s+{var_name}\s*=\s*JSON\.parse\('(.+?)'\)"
    m = re.search(pattern, html)
    if not m:
        raise ValueError(f"Could not find var {var_name} in HTML")
    raw = m.group(1)
    # Understat escapes single quotes and backslashes
    raw = raw.encode().decode("unicode_escape")
    return json.loads(raw)


# ── League-level match data ────────────────────────────────────────────────────

def fetch_league_matches(understat_name: str, season_year: int = 2024) -> list:
    """
    Returns all matches for a league+season with pre-match xG totals.
    understat_name: 'EPL', 'La_liga', 'Bundesliga', 'Serie_A', 'Ligue_1'
    """
    url  = f"{US_BASE}/league/{understat_name}/{season_year}"
    html = _fetch_html(url)
    try:
        data = _extract_json(html, "datesData")
    except ValueError:
        logger.warning("Could not parse datesData from %s", url)
        return []

    matches = []
    for m in data:
        matches.append({
            "understat_id": int(m["id"]),
            "home_team":    m["h"]["title"],
            "away_team":    m["a"]["title"],
            "home_xg":      float(m.get("xG", {}).get("h", 0) or 0),
            "away_xg":      float(m.get("xG", {}).get("a", 0) or 0),
            "home_score":   int(m.get("goals", {}).get("h", 0) or 0),
            "away_score":   int(m.get("goals", {}).get("a", 0) or 0),
            "date":         m.get("datetime", ""),
            "is_result":    m.get("isResult", False),
        })
    return matches


def sync_xg_to_matches(understat_name: str, season_year: int = 2024) -> int:
    """
    Match Understat data to our matches table by team name + date proximity.
    Updates home_xg, away_xg, understat_match_id.
    """
    us_matches = fetch_league_matches(understat_name, season_year)
    if not us_matches:
        return 0

    conn = get_connection()
    c    = conn.cursor()
    updated = 0

    for um in us_matches:
        if not um["is_result"]:
            continue

        # Fuzzy-match by team short_name and date within ±1 day
        us_date = um["date"][:10] if um["date"] else ""
        c.execute("""
            SELECT m.id
            FROM matches m
            JOIN teams ht ON ht.id = m.home_team_id
            JOIN teams at ON at.id = m.away_team_id
            WHERE (ht.short_name LIKE ? OR ht.full_name LIKE ?)
              AND (at.short_name LIKE ? OR at.full_name LIKE ?)
              AND date(m.match_date) BETWEEN date(?, '-1 day') AND date(?, '+1 day')
            LIMIT 1
        """, (
            f"%{um['home_team'][:6]}%", f"%{um['home_team'][:6]}%",
            f"%{um['away_team'][:6]}%", f"%{um['away_team'][:6]}%",
            us_date, us_date
        ))
        row = c.fetchone()
        if not row:
            continue

        c.execute("""
            UPDATE matches SET
                understat_match_id = ?,
                home_xg = ?,
                away_xg = ?,
                updated_at = datetime('now')
            WHERE id = ?
        """, (um["understat_id"], um["home_xg"], um["away_xg"], row["id"]))
        updated += 1

    conn.commit()
    conn.close()
    logger.info("Synced xG for %d/%d matches in %s", updated, len(us_matches), understat_name)
    return updated


# ── Shot-level xG for a single match ──────────────────────────────────────────

def fetch_match_shot_xg(understat_match_id: int) -> list:
    """
    Returns shot-level xG data for a specific match.
    Each dict: {minute, team_side, player, xg_value}
    """
    url  = f"{US_BASE}/match/{understat_match_id}"
    html = _fetch_html(url)
    try:
        shots = _extract_json(html, "shotsData")
    except ValueError:
        logger.warning("Could not parse shotsData for match %d", understat_match_id)
        return []

    events = []
    for side in ("h", "a"):
        team_side = "home" if side == "h" else "away"
        for shot in shots.get(side, []):
            events.append({
                "minute":    int(shot.get("minute", 0)),
                "team_side": team_side,
                "player":    shot.get("player", ""),
                "xg_value":  float(shot.get("xG", 0) or 0),
            })

    # Sort by minute and compute cumulative xG
    events.sort(key=lambda e: e["minute"])
    cum_home = 0.0
    cum_away = 0.0
    for e in events:
        if e["team_side"] == "home":
            cum_home += e["xg_value"]
        else:
            cum_away += e["xg_value"]
        e["cumulative_home_xg"] = round(cum_home, 4)
        e["cumulative_away_xg"] = round(cum_away, 4)

    return events


def store_shot_xg(match_id: int, understat_match_id: int) -> int:
    """Fetch and store shot-level xG for a match."""
    shots = fetch_match_shot_xg(understat_match_id)
    if not shots:
        return 0

    conn = get_connection()
    c    = conn.cursor()
    c.execute("DELETE FROM xg_data WHERE match_id=?", (match_id,))

    for shot in shots:
        c.execute("""
            INSERT INTO xg_data
                (match_id, minute, team_side, player, xg_value,
                 cumulative_home_xg, cumulative_away_xg)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            match_id, shot["minute"], shot["team_side"], shot["player"],
            shot["xg_value"], shot["cumulative_home_xg"], shot["cumulative_away_xg"]
        ))

    conn.commit()
    conn.close()
    logger.info("Stored %d shots for match_id=%d", len(shots), match_id)
    return len(shots)


# ── League draw stats aggregation ─────────────────────────────────────────────

def compute_league_draw_stats(fd_code: str, season: str = "2024/25") -> dict:
    """
    Aggregate draw_rate and avg xG from completed matches in DB.
    Returns dict with stats and upserts into league_draw_stats.
    """
    conn = get_connection()
    c    = conn.cursor()

    c.execute("SELECT id FROM leagues WHERE fd_code=?", (fd_code,))
    row = c.fetchone()
    if not row:
        conn.close()
        return {}
    league_id = row["id"]

    c.execute("""
        SELECT
            COUNT(*) AS total,
            SUM(is_draw) AS draws,
            AVG(home_xg) AS avg_home_xg,
            AVG(away_xg) AS avg_away_xg
        FROM matches
        WHERE league_id=? AND status='FINISHED' AND is_draw IS NOT NULL
    """, (league_id,))
    r = c.fetchone()
    if not r or r["total"] == 0:
        conn.close()
        return {}

    draw_rate = (r["draws"] or 0) / r["total"]
    stats = {
        "league_id":      league_id,
        "season":         season,
        "matches_played": r["total"],
        "draws":          int(r["draws"] or 0),
        "draw_rate":      round(draw_rate, 4),
        "avg_home_xg":    round(r["avg_home_xg"] or 0, 3),
        "avg_away_xg":    round(r["avg_away_xg"] or 0, 3),
    }

    c.execute("""
        INSERT INTO league_draw_stats
            (league_id, season, matches_played, draws, draw_rate, avg_home_xg, avg_away_xg)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(league_id, season) DO UPDATE SET
            matches_played = excluded.matches_played,
            draws          = excluded.draws,
            draw_rate      = excluded.draw_rate,
            avg_home_xg    = excluded.avg_home_xg,
            avg_away_xg    = excluded.avg_away_xg,
            updated_at     = datetime('now')
    """, (
        stats["league_id"], stats["season"], stats["matches_played"],
        stats["draws"], stats["draw_rate"], stats["avg_home_xg"], stats["avg_away_xg"]
    ))

    conn.commit()
    conn.close()
    logger.info("League %s draw_rate=%.3f over %d matches", fd_code, draw_rate, r["total"])
    return stats


# ── H2H draw stats ─────────────────────────────────────────────────────────────

def compute_h2h_stats(home_team_id: int, away_team_id: int, lookback: int = 10) -> dict:
    """Compute and store H2H draw rate for a fixture pair."""
    conn = get_connection()
    c    = conn.cursor()

    c.execute("""
        SELECT COUNT(*) AS n, SUM(is_draw) AS draws
        FROM matches
        WHERE (home_team_id=? AND away_team_id=?)
           OR (home_team_id=? AND away_team_id=?)
           AND status='FINISHED' AND is_draw IS NOT NULL
        ORDER BY match_date DESC
        LIMIT ?
    """, (home_team_id, away_team_id, away_team_id, home_team_id, lookback))
    r = c.fetchone()

    n     = r["n"] or 0
    draws = int(r["draws"] or 0)
    rate  = draws / n if n > 0 else 0.27   # fallback to league average

    c.execute("""
        INSERT INTO h2h_stats
            (home_team_id, away_team_id, lookback_games, matches_counted, draws, draw_rate)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(home_team_id, away_team_id, lookback_games) DO UPDATE SET
            matches_counted = excluded.matches_counted,
            draws           = excluded.draws,
            draw_rate       = excluded.draw_rate,
            updated_at      = datetime('now')
    """, (home_team_id, away_team_id, lookback, n, draws, round(rate, 4)))

    conn.commit()
    conn.close()
    return {"matches": n, "draws": draws, "draw_rate": rate}


# ── Team form ──────────────────────────────────────────────────────────────────

def compute_team_form(team_id: int, as_of_date: str = None, lookback: int = 5) -> dict:
    """Compute rolling form for a team and store in team_form table."""
    if as_of_date is None:
        as_of_date = date.today().isoformat()

    conn = get_connection()
    c    = conn.cursor()

    c.execute("""
        SELECT
            home_team_id, away_team_id,
            home_score, away_score,
            home_xg, away_xg, is_draw
        FROM matches
        WHERE (home_team_id=? OR away_team_id=?)
          AND status='FINISHED'
          AND is_draw IS NOT NULL
          AND match_date <= ?
        ORDER BY match_date DESC
        LIMIT ?
    """, (team_id, team_id, as_of_date, lookback))

    rows  = c.fetchall()
    wins = draws = losses = gf = ga = xgf = xga = 0

    for r in rows:
        is_home = r["home_team_id"] == team_id
        if is_home:
            ts, ta = r["home_score"], r["away_score"]
            xgf_m  = r["home_xg"] or 0
            xga_m  = r["away_xg"] or 0
        else:
            ts, ta = r["away_score"], r["home_score"]
            xgf_m  = r["away_xg"] or 0
            xga_m  = r["home_xg"] or 0

        gf += ts; ga += ta; xgf += xgf_m; xga += xga_m
        if ts > ta:    wins   += 1
        elif ts == ta: draws  += 1
        else:          losses += 1

    n    = len(rows)
    rate = draws / n if n > 0 else None

    c.execute("""
        INSERT INTO team_form
            (team_id, as_of_date, lookback_games, wins, draws, losses,
             goals_scored, goals_conceded, xg_scored, xg_conceded, form_draw_rate)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(team_id, as_of_date, lookback_games) DO UPDATE SET
            wins           = excluded.wins,
            draws          = excluded.draws,
            losses         = excluded.losses,
            goals_scored   = excluded.goals_scored,
            goals_conceded = excluded.goals_conceded,
            xg_scored      = excluded.xg_scored,
            xg_conceded    = excluded.xg_conceded,
            form_draw_rate = excluded.form_draw_rate,
            updated_at     = datetime('now')
    """, (
        team_id, as_of_date, lookback, wins, draws, losses,
        gf, ga, round(xgf, 3), round(xga, 3),
        round(rate, 4) if rate is not None else None
    ))

    conn.commit()
    conn.close()
    return {"played": n, "wins": wins, "draws": draws, "losses": losses,
            "draw_rate": rate}


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from db.schema import create_schema, seed_leagues

    create_schema()
    seed_leagues()

    # Sync xG for Premier League
    synced = sync_xg_to_matches("EPL", 2024)
    print(f"xG synced for {synced} PL matches")

    # Aggregate draw stats
    stats = compute_league_draw_stats("PL")
    print("PL draw stats:", stats)
