"""
Layer 3 – Live Poisson Draw Model
Computes real-time draw probability using:
  - Current scoreline
  - Minutes elapsed / remaining
  - Live cumulative xG as λ (Poisson rate) estimator
  - Score-state transition matrix

The model simulates the remaining match as two independent Poisson processes,
one for each team, and returns P(final_score is a draw).
"""

import math
import logging
from dataclasses import dataclass
from typing import Optional

from db.schema import get_connection

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
MATCH_DURATION    = 90          # nominal minutes
AVG_XG_RATE       = 1.2        # goals per 90 min baseline if no xG available
XG_WEIGHT         = 0.70       # blend: live xG vs historical rate
HISTORICAL_WEIGHT = 0.30
MAX_GOALS         = 8           # truncate Poisson sum at 8 goals


@dataclass
class LivePoissonResult:
    minute:        int
    home_score:    int
    away_score:    int
    live_home_xg:  float
    live_away_xg:  float
    minutes_left:  float
    lambda_home:   float    # expected additional goals for home
    lambda_away:   float    # expected additional goals for away
    draw_prob:     float    # P(match ends as draw)
    home_win_prob: float
    away_win_prob: float
    score_state:   str      # 'drawing'|'home_leading'|'away_leading'


class LivePoissonModel:
    """
    Computes draw probability at any point in a live match.
    """

    def compute(
        self,
        match_id:     int,
        minute:       int,
        home_score:   int,
        away_score:   int,
        live_home_xg: Optional[float] = None,
        live_away_xg: Optional[float] = None,
    ) -> LivePoissonResult:
        """
        Main entry point.  Call this at 60', 70', 80' or on any goal event.
        """
        # Pull pre-match historical xG from DB for blending
        hist_h_xg, hist_a_xg = self._get_historical_xg(match_id)

        # Estimate full-match xG rates from live xG (pace-adjusted)
        pace_factor = MATCH_DURATION / max(minute, 1)

        if live_home_xg is not None:
            projected_home_xg = live_home_xg * pace_factor
            blended_home_rate = (XG_WEIGHT * projected_home_xg +
                                 HISTORICAL_WEIGHT * hist_h_xg)
        else:
            blended_home_rate = hist_h_xg

        if live_away_xg is not None:
            projected_away_xg = live_away_xg * pace_factor
            blended_away_rate = (XG_WEIGHT * projected_away_xg +
                                 HISTORICAL_WEIGHT * hist_a_xg)
        else:
            blended_away_rate = hist_a_xg

        # Scale down to remaining minutes
        minutes_left = max(0.0, MATCH_DURATION - minute)
        frac_left    = minutes_left / MATCH_DURATION

        # Expected additional goals in remaining time
        lambda_home = blended_home_rate * frac_left
        lambda_away = blended_away_rate * frac_left

        # Score differential
        diff = home_score - away_score

        # Compute P(draw) via Poisson convolution
        draw_prob, home_win_prob, away_win_prob = self._score_prob(
            diff, lambda_home, lambda_away
        )

        score_state = (
            "drawing"      if diff == 0 else
            "home_leading" if diff > 0  else
            "away_leading"
        )

        result = LivePoissonResult(
            minute        = minute,
            home_score    = home_score,
            away_score    = away_score,
            live_home_xg  = round(live_home_xg or 0, 3),
            live_away_xg  = round(live_away_xg or 0, 3),
            minutes_left  = round(minutes_left, 1),
            lambda_home   = round(lambda_home, 4),
            lambda_away   = round(lambda_away, 4),
            draw_prob     = round(draw_prob, 4),
            home_win_prob = round(home_win_prob, 4),
            away_win_prob = round(away_win_prob, 4),
            score_state   = score_state,
        )

        logger.info(
            "LIVE [%d'] %d-%d xG(%.2f-%.2f) λ(%.3f-%.3f) "
            "draw=%.3f home=%.3f away=%.3f",
            minute, home_score, away_score,
            live_home_xg or 0, live_away_xg or 0,
            lambda_home, lambda_away,
            draw_prob, home_win_prob, away_win_prob,
        )
        return result

    # ── Poisson probability engine ─────────────────────────────────────────────

    def _score_prob(
        self, current_diff: int, lam_h: float, lam_a: float
    ) -> tuple:
        """
        Given current goal difference (home − away) and expected additional
        Poisson rates, return (P_draw, P_home_win, P_away_win).
        """
        draw = home_win = away_win = 0.0

        for k in range(MAX_GOALS + 1):
            pk = _poisson_pmf(k, lam_h)
            if pk < 1e-9:
                continue
            for j in range(MAX_GOALS + 1):
                pj  = _poisson_pmf(j, lam_a)
                if pj < 1e-9:
                    continue
                p   = pk * pj
                net = current_diff + k - j
                if net == 0:
                    draw      += p
                elif net > 0:
                    home_win  += p
                else:
                    away_win  += p

        # Normalise rounding errors
        total = draw + home_win + away_win
        if total > 0:
            draw     /= total
            home_win /= total
            away_win /= total

        return draw, home_win, away_win

    # ── Data helpers ───────────────────────────────────────────────────────────

    def _get_historical_xg(self, match_id: int) -> tuple:
        conn = get_connection()
        c    = conn.cursor()
        c.execute("""
            SELECT home_xg, away_xg, home_team_id, away_team_id
            FROM matches WHERE id=?
        """, (match_id,))
        row = c.fetchone()

        if row and row["home_xg"] is not None:
            h = row["home_xg"]
            a = row["away_xg"]
            conn.close()
            return h, a

        # Fall back to team historical averages
        h = a = AVG_XG_RATE
        if row:
            for col, team_id in [("home", row["home_team_id"]), ("away", row["away_team_id"])]:
                c.execute("""
                    SELECT AVG(CASE WHEN home_team_id=? THEN home_xg
                                   WHEN away_team_id=? THEN away_xg END) AS avg
                    FROM matches
                    WHERE (home_team_id=? OR away_team_id=?)
                      AND home_xg IS NOT NULL
                """, (team_id, team_id, team_id, team_id))
                avg_row = c.fetchone()
                val = (avg_row["avg"] or AVG_XG_RATE) if avg_row else AVG_XG_RATE
                if col == "home":
                    h = val
                else:
                    a = val

        conn.close()
        return h, a

    # ── Latest live state from DB ──────────────────────────────────────────────

    def latest_from_db(self, match_id: int) -> Optional[LivePoissonResult]:
        """Read most recent live_event and run model."""
        conn = get_connection()
        c    = conn.cursor()
        c.execute("""
            SELECT minute, home_score, away_score, live_home_xg, live_away_xg
            FROM live_events
            WHERE match_id=?
            ORDER BY minute DESC, created_at DESC
            LIMIT 1
        """, (match_id,))
        row = c.fetchone()
        conn.close()

        if not row:
            return self.compute(match_id, 0, 0, 0)

        return self.compute(
            match_id,
            row["minute"],
            row["home_score"] or 0,
            row["away_score"] or 0,
            row["live_home_xg"],
            row["live_away_xg"],
        )


# ── Pure Poisson helper ────────────────────────────────────────────────────────

def _poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


# ── Scenario analysis helper ───────────────────────────────────────────────────

def draw_prob_table(lam_h: float, lam_a: float, current_diff: int = 0) -> str:
    model = LivePoissonModel()
    header = f"{'Minute':>8} {'Min Left':>9} {'λ_h':>6} {'λ_a':>6} {'P(draw)':>9}"
    lines  = [header, "-" * len(header)]
    for minute in [0, 45, 60, 70, 80, 85, 90]:
        frac   = max(0, (90 - minute)) / 90
        lh     = lam_h * frac
        la     = lam_a * frac
        draw, _, _ = model._score_prob(current_diff, lh, la)
        lines.append(
            f"{minute:>8} {90 - minute:>9} {lh:>6.3f} {la:>6.3f} {draw:>9.4f}"
        )
    return "\n".join(lines)


# ── CLI / Quick test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = LivePoissonModel()

    print("=== P(Draw) at various minutes, 0-0 scoreline, λ_h=1.3, λ_a=1.1 ===")
    print(draw_prob_table(1.3, 1.1, 0))

    print("\n=== P(Draw) at various minutes, 1-0 scoreline, λ_h=1.3, λ_a=1.1 ===")
    print(draw_prob_table(1.3, 1.1, 1))

    result = model.compute(
        match_id=1,
        minute=70,
        home_score=1,
        away_score=1,
        live_home_xg=1.1,
        live_away_xg=0.9,
    )
    print(f"\n[70' 1-1] draw={result.draw_prob:.3f}  "
          f"home={result.home_win_prob:.3f}  "
          f"away={result.away_win_prob:.3f}")
