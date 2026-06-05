"""
Kalshi Market Fetcher + Ticker Matcher
========================================
Fetches active soccer draw markets from Kalshi, fuzzy-matches them to
matches in our SQLite DB, and stores ticker + live price in kalshi_contracts.

Kalshi API v2 reference:
  GET /trade-api/v2/markets
    params: category=soccer, status=open, limit, cursor
  GET /trade-api/v2/markets/{ticker}
    returns: market.yes_ask, yes_bid, open_interest, volume, close_time

Team-name normalisation strategy
─────────────────────────────────
Kalshi titles look like:
  "Will Man City vs Arsenal end in a draw?"
  "SOCCER-EPL-20250112-MANCITY-ARSENAL-DRAW"

We extract candidate team names from both the title string and the ticker
slug, normalise both sides with a shared lookup table of common
abbreviations, then use rapidfuzz token_sort_ratio for fuzzy matching.
A match is accepted when BOTH home and away scores ≥ MATCH_THRESHOLD (70).
"""

import os
import re
import time
import logging
from datetime import date, timedelta, datetime, timezone
from typing import Optional

import requests
from rapidfuzz import fuzz

from db.schema import get_connection

logger = logging.getLogger(__name__)

# ── Kalshi API ─────────────────────────────────────────────────────────────────
KALSHI_BASE   = "https://trading-api.kalshi.com/trade-api/v2"
KALSHI_KEY    = os.getenv("KALSHI_API_KEY", "")
KALSHI_SECRET = os.getenv("KALSHI_API_SECRET", "")

# Minimum rapidfuzz score (0–100) to accept a team-name match
MATCH_THRESHOLD = 70

# Date window for matching: look ±1 day around the Kalshi market close date
DATE_WINDOW_DAYS = 1

# ── Abbreviation / alias table ─────────────────────────────────────────────────
# Keys are fragments that appear in Kalshi names/tickers; values are canonical
# fragments we'd expect in Football-Data.org team names.
# All comparisons are lower-cased before lookup.
ALIASES: dict[str, str] = {
    # Premier League
    "man city":        "manchester city",
    "mancity":         "manchester city",
    "man utd":         "manchester united",
    "manutd":          "manchester united",
    "man united":      "manchester united",
    "spurs":           "tottenham",
    "tottenham hotspur": "tottenham",
    "wolves":          "wolverhampton",
    "wolverhampton wanderers": "wolverhampton",
    "west ham":        "west ham united",
    "westham":         "west ham united",
    "newcastle":       "newcastle united",
    "leicester":       "leicester city",
    "norwich":         "norwich city",
    "brighton":        "brighton & hove albion",
    "brentford":       "brentford",
    "nott forest":     "nottingham forest",
    "nott'm forest":   "nottingham forest",
    "nottm forest":    "nottingham forest",
    # La Liga
    "real madrid":     "real madrid",
    "atletico":        "atlético madrid",
    "atletico madrid": "atlético madrid",
    "atl madrid":      "atlético madrid",
    "atleti":          "atlético madrid",
    "barca":           "barcelona",
    "sociedad":        "real sociedad",
    "real sociedad":   "real sociedad",
    "villareal":       "villarreal",
    # Bundesliga
    "bayer leverkusen": "bayer leverkusen",
    "leverkusen":      "bayer leverkusen",
    "rb leipzig":      "rb leipzig",
    "leipzig":         "rb leipzig",
    "dortmund":        "borussia dortmund",
    "bvb":             "borussia dortmund",
    "m'gladbach":      "borussia m'gladbach",
    "gladbach":        "borussia m'gladbach",
    "hertha":          "hertha bsc",
    "schalke":         "schalke 04",
    "frankfurt":       "eintracht frankfurt",
    "eintracht":       "eintracht frankfurt",
    # Serie A
    "ac milan":        "ac milan",
    "milan":           "ac milan",
    "inter milan":     "internazionale",
    "inter":           "internazionale",
    "juventus":        "juventus",
    "juve":            "juventus",
    "napoli":          "ssc napoli",
    "as roma":         "as roma",
    "roma":            "as roma",
    "lazio":           "ss lazio",
    "atalanta":        "atalanta",
    "fiorentina":      "acf fiorentina",
    # Ligue 1
    "psg":             "paris saint-germain",
    "paris sg":        "paris saint-germain",
    "paris saint germain": "paris saint-germain",
    "marseille":       "olympique de marseille",
    "om":              "olympique de marseille",
    "lyon":            "olympique lyonnais",
    "ol":              "olympique lyonnais",
    "monaco":          "as monaco",
    "lille":           "losc lille",
    "rennes":          "stade rennais",
}


def _normalise(name: str) -> str:
    """Lowercase, strip punctuation, expand known aliases."""
    name = name.lower().strip()
    # Remove common noise words that appear in Kalshi titles
    for noise in ("fc", "cf", "afc", "sc", "1.", "fsv", "tsg", "vfb", "vfl",
                  "sv", "bsc", "ssc", "acf", "ssv", "rb"):
        name = re.sub(rf"\b{re.escape(noise)}\b", "", name)
    name = re.sub(r"['\-&,.]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    # Apply alias substitution (longest match wins)
    for alias, canonical in sorted(ALIASES.items(), key=lambda x: -len(x[0])):
        if alias in name:
            name = name.replace(alias, canonical)
    return name.strip()


def _extract_teams_from_title(title: str) -> Optional[tuple[str, str]]:
    """
    Parse Kalshi market title to extract home/away team name strings.

    Handles patterns:
      "Will <Home> vs <Away> end in a draw?"
      "<Home> vs <Away> Draw"
      "<Home> vs <Away>"
    """
    # Strip leading "Will " and trailing question/result phrases
    clean = re.sub(r"^will\s+", "", title, flags=re.IGNORECASE)
    clean = re.sub(
        r"\s+(end in a draw|draw\??|result in a draw|finish|match result)[?]?$",
        "", clean, flags=re.IGNORECASE
    )
    # Split on " vs " or " v "
    parts = re.split(r"\s+v\.?s\.?\s+", clean, flags=re.IGNORECASE)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return None


def _extract_teams_from_ticker(ticker: str) -> Optional[tuple[str, str]]:
    """
    Parse a Kalshi ticker slug like:
      SOCCER-EPL-20250112-MANCITY-ARSENAL-DRAW
      KXSOCCER-EPL-MANCITY-ARSENAL
    Returns (home, away) as space-separated strings, or None.
    """
    # Remove known prefixes/suffixes
    slug = ticker.upper()
    slug = re.sub(r"^(KX)?SOCCER-[A-Z0-9]+-", "", slug)   # strip SOCCER-EPL- etc.
    slug = re.sub(r"-?\d{8}-?", "-", slug)                  # strip YYYYMMDD date
    slug = re.sub(r"-(DRAW|HOME|AWAY|RESULT|WIN)$", "", slug)
    parts = slug.split("-")
    # Need at least two non-empty fragments
    parts = [p for p in parts if p]
    if len(parts) >= 2:
        # Convert slug tokens like MANCITY → man city
        def detokenise(tok: str) -> str:
            # Insert space before each uppercase run after lowercase (camelCase guard)
            spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", tok)
            return spaced.lower()
        return detokenise(parts[0]), detokenise(parts[1])
    return None


def _team_score(kalshi_raw: str, db_short: str, db_full: str) -> int:
    """Return the best rapidfuzz score comparing a Kalshi team string to DB names."""
    kn = _normalise(kalshi_raw)
    scores = [
        fuzz.token_sort_ratio(kn, _normalise(db_short or "")),
        fuzz.token_sort_ratio(kn, _normalise(db_full or "")),
        fuzz.partial_ratio(kn, _normalise(db_full or "")),
    ]
    return max(scores)


# ── Kalshi HTTP helpers ────────────────────────────────────────────────────────

def _kalshi_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "Authorization": f"Bearer {KALSHI_KEY}",
        "Accept":        "application/json",
    })
    return s


def fetch_soccer_markets(
    date_from: str = None,
    date_to:   str = None,
    status:    str = "open",
    limit:     int = 200,
) -> list[dict]:
    """
    Fetch all active Kalshi soccer/draw markets, paging through cursor.

    date_from / date_to: ISO date strings "YYYY-MM-DD" to filter by
    market close_time.  If omitted, fetches all open soccer markets.

    Returns list of raw market dicts from the Kalshi API.
    """
    if not KALSHI_KEY:
        logger.warning("KALSHI_API_KEY not set — cannot fetch markets")
        return []

    session  = _kalshi_session()
    markets  = []
    cursor   = None
    page     = 0

    while True:
        params: dict = {
            "category": "soccer",
            "status":   status,
            "limit":    min(limit, 200),
        }
        if cursor:
            params["cursor"] = cursor

        try:
            r = session.get(f"{KALSHI_BASE}/markets", params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
        except requests.HTTPError as e:
            logger.error("Kalshi markets API error: %s", e)
            break
        except Exception as e:
            logger.error("Kalshi request failed: %s", e)
            break

        batch = data.get("markets", [])
        if not batch:
            break

        # Optional date filter on close_time
        for m in batch:
            close = m.get("close_time") or m.get("expected_expiration_time", "")
            if date_from and close and close[:10] < date_from:
                continue
            if date_to and close and close[:10] > date_to:
                continue
            # Only keep draw markets (skip home/away outcome markets)
            title = (m.get("title") or "").lower()
            ticker = (m.get("ticker") or "").upper()
            if "draw" in title or "draw" in ticker or "-DRAW" in ticker:
                markets.append(m)

        cursor = data.get("cursor")
        page  += 1
        logger.debug("Fetched page %d, %d draw markets so far", page, len(markets))

        if not cursor or len(batch) < params["limit"]:
            break

        time.sleep(0.3)   # polite pacing

    logger.info("Fetched %d Kalshi draw markets", len(markets))
    return markets


def refresh_market_prices(tickers: list[str]) -> dict[str, dict]:
    """
    Fetch current yes_ask/yes_bid/open_interest for a list of tickers.
    Returns {ticker: market_dict}.
    """
    if not KALSHI_KEY:
        return {}
    session = _kalshi_session()
    results = {}
    for ticker in tickers:
        try:
            r = session.get(f"{KALSHI_BASE}/markets/{ticker}", timeout=10)
            r.raise_for_status()
            m = r.json().get("market", {})
            if m:
                results[ticker] = m
        except Exception as e:
            logger.warning("Price refresh failed for %s: %s", ticker, e)
        time.sleep(0.15)
    return results


# ── DB helpers ─────────────────────────────────────────────────────────────────

def _load_db_teams(conn) -> list[dict]:
    """Load all teams with their match date ranges for quick lookup."""
    c = conn.cursor()
    c.execute("""
        SELECT t.id, t.short_name, t.full_name, t.league_id
        FROM teams t
    """)
    return [dict(r) for r in c.fetchall()]


def _load_db_matches(conn, date_from: str, date_to: str) -> list[dict]:
    """Load matches within a date window, joining team names."""
    c = conn.cursor()
    c.execute("""
        SELECT
            m.id            AS match_id,
            m.match_date,
            m.status,
            m.fd_match_id,
            ht.short_name   AS home_short,
            ht.full_name    AS home_full,
            at.short_name   AS away_short,
            at.full_name    AS away_full,
            kc.ticker       AS existing_ticker
        FROM matches m
        JOIN teams ht ON ht.id = m.home_team_id
        JOIN teams at ON at.id = m.away_team_id
        LEFT JOIN kalshi_contracts kc ON kc.match_id = m.id AND kc.contract_type = 'draw'
        WHERE date(m.match_date) BETWEEN date(?) AND date(?)
        ORDER BY m.match_date
    """, (date_from, date_to))
    return [dict(r) for r in c.fetchall()]


# ── Core matching logic ────────────────────────────────────────────────────────

def _match_market_to_db(
    market:     dict,
    db_matches: list[dict],
    db_teams:   list[dict],
) -> Optional[dict]:
    """
    Try to find a single DB match for a Kalshi market dict.
    Returns the db_match row dict on success, None otherwise.
    """
    title  = market.get("title", "")
    ticker = market.get("ticker", "")

    # Determine date window from Kalshi close_time
    close_raw = market.get("close_time") or market.get("expected_expiration_time", "")
    if close_raw:
        try:
            close_date = close_raw[:10]
        except Exception:
            close_date = None
    else:
        close_date = None

    # Extract team pair candidates from title, then ticker as fallback
    team_pair = _extract_teams_from_title(title)
    if not team_pair:
        team_pair = _extract_teams_from_ticker(ticker)
    if not team_pair:
        logger.debug("Could not extract teams from: %r / %r", title, ticker)
        return None

    kalshi_home_raw, kalshi_away_raw = team_pair

    # Build a lookup: team_id → team row
    team_by_id = {t["id"]: t for t in db_teams}

    best_match = None
    best_score = 0

    for db_m in db_matches:
        # Optionally filter by date proximity
        if close_date:
            try:
                diff = abs((datetime.fromisoformat(db_m["match_date"][:10]) -
                            datetime.fromisoformat(close_date)).days)
                if diff > DATE_WINDOW_DAYS + 1:
                    continue
            except Exception:
                pass

        home_score = _team_score(
            kalshi_home_raw, db_m["home_short"], db_m["home_full"]
        )
        away_score = _team_score(
            kalshi_away_raw, db_m["away_short"], db_m["away_full"]
        )

        if home_score >= MATCH_THRESHOLD and away_score >= MATCH_THRESHOLD:
            combined = home_score + away_score
            if combined > best_score:
                best_score = combined
                best_match = db_m
                logger.debug(
                    "  Candidate: %s vs %s — h=%.0f a=%.0f [%s vs %s]",
                    kalshi_home_raw, kalshi_away_raw,
                    home_score, away_score,
                    db_m["home_full"], db_m["away_full"],
                )

    return best_match


def _upsert_contract(conn, match_id: int, market: dict):
    """Write or update a kalshi_contracts row."""
    ticker        = market.get("ticker", "")
    title         = market.get("title", "")
    yes_ask       = market.get("yes_ask") or market.get("last_price") or 0
    yes_bid       = market.get("yes_bid", 0) or 0
    yes_price     = (yes_ask + yes_bid) / 2 if (yes_ask or yes_bid) else None
    open_interest = market.get("open_interest", 0)
    volume        = market.get("volume", 0)

    conn.execute("""
        INSERT INTO kalshi_contracts
            (match_id, ticker, market_title, contract_type,
             yes_price, open_interest, volume, fetched_at)
        VALUES (?, ?, ?, 'draw', ?, ?, ?, datetime('now'))
        ON CONFLICT(ticker) DO UPDATE SET
            match_id      = excluded.match_id,
            market_title  = excluded.market_title,
            yes_price     = excluded.yes_price,
            open_interest = excluded.open_interest,
            volume        = excluded.volume,
            fetched_at    = datetime('now')
    """, (match_id, ticker, title, yes_price, open_interest, volume))


# ── Public entry points ────────────────────────────────────────────────────────

def sync_tickers(date_from: str = None, date_to: str = None) -> dict:
    """
    Fetch Kalshi soccer draw markets, match to DB, store tickers.

    date_from / date_to: "YYYY-MM-DD".  Defaults to today + 7 days.

    Returns {"fetched": N, "matched": M, "unmatched": U, "unmatched_titles": [...]}
    """
    today = date.today()
    if date_from is None:
        date_from = today.isoformat()
    if date_to is None:
        date_to = (today + timedelta(days=7)).isoformat()

    logger.info("Syncing Kalshi tickers %s → %s", date_from, date_to)

    markets = fetch_soccer_markets(date_from=date_from, date_to=date_to)
    if not markets:
        return {"fetched": 0, "matched": 0, "unmatched": 0, "unmatched_titles": []}

    conn = get_connection()
    # Widen DB window by ±1 day to account for timezone differences
    db_from = (datetime.fromisoformat(date_from) - timedelta(days=1)).date().isoformat()
    db_to   = (datetime.fromisoformat(date_to)   + timedelta(days=1)).date().isoformat()

    db_teams   = _load_db_teams(conn)
    db_matches = _load_db_matches(conn, db_from, db_to)

    matched   = 0
    unmatched = []

    for market in markets:
        db_m = _match_market_to_db(market, db_matches, db_teams)
        if db_m:
            _upsert_contract(conn, db_m["match_id"], market)
            matched += 1
            logger.info(
                "MATCHED  %-40s → match_id=%d  %s vs %s",
                market.get("ticker"), db_m["match_id"],
                db_m["home_full"], db_m["away_full"],
            )
        else:
            unmatched.append(market.get("title") or market.get("ticker"))
            logger.warning(
                "UNMATCHED  %s / %s",
                market.get("ticker"), market.get("title"),
            )

    conn.commit()
    conn.close()

    result = {
        "fetched":           len(markets),
        "matched":           matched,
        "unmatched":         len(unmatched),
        "unmatched_titles":  unmatched,
    }
    logger.info("Ticker sync complete: %s", result)
    return result


def get_watchlist(days_ahead: int = 7) -> list[dict]:
    """
    Return upcoming matches that have a matched Kalshi ticker.
    Each item: {match_date, home, away, ticker, yes_price, open_interest,
                pre_draw_prob, match_id}
    Prices are refreshed live from Kalshi before returning.
    """
    today = date.today().isoformat()
    until = (date.today() + timedelta(days=days_ahead)).isoformat()

    conn = get_connection()
    c    = conn.cursor()
    c.execute("""
        SELECT
            m.id            AS match_id,
            m.match_date,
            m.status,
            m.pre_draw_prob,
            ht.full_name    AS home,
            at.full_name    AS away,
            kc.ticker,
            kc.yes_price,
            kc.open_interest,
            kc.fetched_at
        FROM matches m
        JOIN teams ht ON ht.id = m.home_team_id
        JOIN teams at ON at.id = m.away_team_id
        JOIN kalshi_contracts kc ON kc.match_id = m.id AND kc.contract_type = 'draw'
        WHERE date(m.match_date) BETWEEN ? AND ?
          AND m.status IN ('SCHEDULED', 'TIMED')
        ORDER BY m.match_date
    """, (today, until))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()

    if not rows:
        return rows

    # Refresh prices for all tickers in one pass
    tickers       = [r["ticker"] for r in rows if r["ticker"]]
    fresh_prices  = refresh_market_prices(tickers) if KALSHI_KEY else {}

    now = datetime.now(timezone.utc).isoformat()
    for row in rows:
        live = fresh_prices.get(row["ticker"])
        if live:
            yes_ask   = live.get("yes_ask", 0) or 0
            yes_bid   = live.get("yes_bid", 0) or 0
            mid       = (yes_ask + yes_bid) / 2 if (yes_ask or yes_bid) else row["yes_price"]
            row["yes_price"]     = mid
            row["open_interest"] = live.get("open_interest", row["open_interest"])
            row["fetched_at"]    = now

    return rows
