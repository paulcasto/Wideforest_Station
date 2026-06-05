"""
Layer 2 – Pre-Game Weighted Signal Scorer
Combines 5 signals into a calibrated draw probability estimate
before kick-off.

Signals:
  1. xG Similarity     – how close are historical xG totals (tight games = more draws)
  2. H2H Draw Rate     – fixture-specific historical draw frequency
  3. League Draw Rate  – baseline for the competition
  4. Table Gap         – small points gap → more competitive → higher draw chance
  5. Form Draw Rate    – combined recent form draw tendency

Output: pre_draw_prob (float, 0–1) stored in matches.pre_draw_prob
"""

import math
import json
import logging
from dataclasses import dataclass, asdict
from typing import Optional

from db.schema import get_connection

logger = logging.getLogger(__name__)

# ── Signal Weights (must sum to 1.0) ──────────────────────────────────────────
WEIGHTS = {
    "xg_similarity":    0.30,
    "h2h_draw_rate":    0.20,
    "league_draw_rate": 0.20,
    "table_gap":        0.15,
    "form_draw_rate":   0.15,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"

# ── Calibration constants ──────────────────────────────────────────────────────
BASELINE_DRAW_RATE  = 0.26
XG_SIMILARITY_MAX   = 2.5    # xG difference beyond this → near-zero similarity score
TABLE_GAP_MAX       = 15     # points gap beyond this → near-zero gap score
FORM_LOOKBACK       = 5      # last N matches for form


@dataclass
class SignalBreakdown:
    xg_similarity:    float
    h2h_draw_rate:    float
    league_draw_rate: float
    table_gap:        float
    form_draw_rate:   float
    weighted_score:   float
    pre_draw_prob:    float
    confidence:       float   # 0–1, based on data availability


class PreGameScorer:
    """Compute pre-game draw probability for a given match_id."""

    def score(self, match_id: int) -> Optional[SignalBreakdown]:
        conn = get_connection()
        c    = conn.cursor()

        # Fetch match + team + league context
        c.execute("""
            SELECT
                m.id, m.home_team_id, m.away_team_id, m.league_id,
                m.home_xg, m.away_xg,
                l.fd_code, l.season
            FROM matches m
            JOIN leagues l ON l.id = m.league_id
            WHERE m.id = ?
        """, (match_id,))
        match = c.fetchone()
        if not match:
            logger.warning("match_id=%d not found", match_id)
            conn.close()
            return None

        home_id   = match["home_team_id"]
        away_id   = match["away_team_id"]
        league_id = match["league_id"]
        season    = match["season"]

        signals      = {}
        data_sources = 0   # for confidence scoring

        # ── Signal 1: xG Similarity ───────────────────────────────────────────
        xg_score = self._xg_similarity_signal(
            home_id, away_id, match["home_xg"], match["away_xg"], c
        )
        signals["xg_similarity"] = xg_score["value"]
        data_sources += xg_score["has_data"]

        # ── Signal 2: H2H Draw Rate ───────────────────────────────────────────
        h2h = self._h2h_signal(home_id, away_id, c)
        signals["h2h_draw_rate"] = h2h["value"]
        data_sources += h2h["has_data"]

        # ── Signal 3: League Draw Rate ────────────────────────────────────────
        league = self._league_signal(league_id, season, c)
        signals["league_draw_rate"] = league["value"]
        data_sources += league["has_data"]

        # ── Signal 4: Table Gap ───────────────────────────────────────────────
        gap = self._table_gap_signal(home_id, away_id, league_id, season, c)
        signals["table_gap"] = gap["value"]
        data_sources += gap["has_data"]

        # ── Signal 5: Form Draw Rate ──────────────────────────────────────────
        form = self._form_signal(home_id, away_id, c)
        signals["form_draw_rate"] = form["value"]
        data_sources += form["has_data"]

        conn.close()

        # ── Weighted blend ────────────────────────────────────────────────────
        weighted_score = sum(WEIGHTS[k] * signals[k] for k in WEIGHTS)

        # Sigmoid calibration
        pre_draw_prob = self._calibrate(weighted_score)

        confidence = data_sources / len(signals)

        result = SignalBreakdown(
            xg_similarity    = round(signals["xg_similarity"], 4),
            h2h_draw_rate    = round(signals["h2h_draw_rate"], 4),
            league_draw_rate = round(signals["league_draw_rate"], 4),
            table_gap        = round(signals["table_gap"], 4),
            form_draw_rate   = round(signals["form_draw_rate"], 4),
            weighted_score   = round(weighted_score, 4),
            pre_draw_prob    = round(pre_draw_prob, 4),
            confidence       = round(confidence, 2),
        )

        # Persist to DB
        self._save(match_id, result)
        logger.info(
            "match_id=%d pre_draw_prob=%.3f confidence=%.2f "
            "[xg=%.3f h2h=%.3f lg=%.3f gap=%.3f form=%.3f]",
            match_id, pre_draw_prob, confidence,
            result.xg_similarity, result.h2h_draw_rate, result.league_draw_rate,
            result.table_gap, result.form_draw_rate,
        )
        return result

    # ── Signal implementations ─────────────────────────────────────────────────

    def _xg_similarity_signal(
        self, home_id: int, away_id: int,
        match_home_xg: float, match_away_xg: float,
        c
    ) -> dict:
        """
        Draws tend to happen when xG totals are close.
        Score = gaussian(|xG_home - xG_away|, sigma=0.8)
        """
        h_xg = match_home_xg
        a_xg = match_away_xg

        if h_xg is None or a_xg is None:
            # Fall back to team historical averages
            c.execute("""
                SELECT AVG(CASE WHEN home_team_id=? THEN home_xg
                                WHEN away_team_id=? THEN away_xg END) AS avg_xg
                FROM matches WHERE (home_team_id=? OR away_team_id=?) AND home_xg IS NOT NULL
            """, (home_id, home_id, home_id, home_id))
            h_row = c.fetchone()
            c.execute("""
                SELECT AVG(CASE WHEN home_team_id=? THEN home_xg
                                WHEN away_team_id=? THEN away_xg END) AS avg_xg
                FROM matches WHERE (home_team_id=? OR away_team_id=?) AND away_xg IS NOT NULL
            """, (away_id, away_id, away_id, away_id))
            a_row = c.fetchone()

            h_xg = (h_row["avg_xg"] or 1.3) if h_row else 1.3
            a_xg = (a_row["avg_xg"] or 1.1) if a_row else 1.1
            has_data = 0
        else:
            has_data = 1

        diff  = abs(h_xg - a_xg)
        sigma = 0.8
        score = math.exp(-(diff ** 2) / (2 * sigma ** 2))
        return {"value": score, "has_data": has_data}

    def _h2h_signal(self, home_id: int, away_id: int, c) -> dict:
        c.execute("""
            SELECT draw_rate, matches_counted
            FROM h2h_stats
            WHERE home_team_id=? AND away_team_id=?
            ORDER BY updated_at DESC LIMIT 1
        """, (home_id, away_id))
        row = c.fetchone()
        if row and row["matches_counted"] >= 3:
            return {"value": row["draw_rate"], "has_data": 1}
        return {"value": BASELINE_DRAW_RATE, "has_data": 0}

    def _league_signal(self, league_id: int, season: str, c) -> dict:
        c.execute("""
            SELECT draw_rate, matches_played
            FROM league_draw_stats
            WHERE league_id=? AND season=?
            ORDER BY updated_at DESC LIMIT 1
        """, (league_id, season))
        row = c.fetchone()
        if row and row["matches_played"] >= 10:
            return {"value": row["draw_rate"], "has_data": 1}
        return {"value": BASELINE_DRAW_RATE, "has_data": 0}

    def _table_gap_signal(
        self, home_id: int, away_id: int,
        league_id: int, season: str, c
    ) -> dict:
        """
        Smaller points gap between teams → more even fixture → higher draw prob.
        Score = 0.18 + (0.35 - 0.18) * exp(-gap / TABLE_GAP_MAX)
        """
        c.execute("""
            SELECT team_id, position, points
            FROM league_table
            WHERE league_id=? AND season=?
              AND team_id IN (?, ?)
            ORDER BY as_of_date DESC
        """, (league_id, season, home_id, away_id))
        rows = c.fetchall()

        if len(rows) < 2:
            return {"value": BASELINE_DRAW_RATE, "has_data": 0}

        pts = {r["team_id"]: r["points"] for r in rows}
        gap = abs(pts.get(home_id, 0) - pts.get(away_id, 0))

        score = 0.18 + (0.35 - 0.18) * math.exp(-gap / TABLE_GAP_MAX)
        return {"value": score, "has_data": 1}

    def _form_signal(self, home_id: int, away_id: int, c) -> dict:
        """Average of home+away team recent draw rates from team_form."""
        today = __import__("datetime").date.today().isoformat()
        rates = []
        for team_id in (home_id, away_id):
            c.execute("""
                SELECT form_draw_rate FROM team_form
                WHERE team_id=? AND lookback_games=? AND as_of_date<=?
                ORDER BY as_of_date DESC LIMIT 1
            """, (team_id, FORM_LOOKBACK, today))
            row = c.fetchone()
            if row and row["form_draw_rate"] is not None:
                rates.append(row["form_draw_rate"])

        if not rates:
            return {"value": BASELINE_DRAW_RATE, "has_data": 0}

        return {"value": sum(rates) / len(rates), "has_data": 1}

    # ── Calibration ────────────────────────────────────────────────────────────

    def _calibrate(self, raw: float) -> float:
        """
        Logistic calibration that maps raw score (0–1) through a sigmoid
        centred on BASELINE_DRAW_RATE, constraining output to [0.05, 0.65].
        """
        eps   = 1e-6
        raw   = max(eps, min(1 - eps, raw))
        logit = math.log(raw / (1 - raw))
        baseline_logit = math.log(BASELINE_DRAW_RATE / (1 - BASELINE_DRAW_RATE))
        adjusted_logit = baseline_logit + 0.8 * (logit - baseline_logit)
        prob = 1 / (1 + math.exp(-adjusted_logit))
        return max(0.05, min(0.65, prob))

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self, match_id: int, result: SignalBreakdown):
        conn = get_connection()
        conn.execute("""
            UPDATE matches SET pre_draw_prob=?, updated_at=datetime('now')
            WHERE id=?
        """, (result.pre_draw_prob, match_id))
        conn.commit()
        conn.close()


# ── Batch scoring ──────────────────────────────────────────────────────────────

def score_upcoming_matches(days_ahead: int = 3) -> list:
    """Score all SCHEDULED matches in the next N days."""
    from datetime import date, timedelta
    today = date.today().isoformat()
    until = (date.today() + timedelta(days=days_ahead)).isoformat()

    conn = get_connection()
    c    = conn.cursor()
    c.execute("""
        SELECT id FROM matches
        WHERE status='SCHEDULED'
          AND match_date BETWEEN ? AND ?
    """, (today, until))
    ids = [r["id"] for r in c.fetchall()]
    conn.close()

    scorer  = PreGameScorer()
    results = []
    for match_id in ids:
        r = scorer.score(match_id)
        if r:
            results.append(r)

    logger.info("Scored %d upcoming matches", len(results))
    return results


if __name__ == "__main__":
    import sys
    from db.schema import create_schema, seed_leagues

    create_schema()
    seed_leagues()

    if len(sys.argv) > 1:
        match_id = int(sys.argv[1])
        scorer   = PreGameScorer()
        result   = scorer.score(match_id)
        if result:
            print(json.dumps(asdict(result), indent=2))
    else:
        results = score_upcoming_matches()
        for r in results:
            print(f"pre_draw_prob={r.pre_draw_prob:.3f}  confidence={r.confidence:.2f}")
