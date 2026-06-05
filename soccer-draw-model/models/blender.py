"""
Layer 4 – Probability Blender
Shifts weight from pre-game scorer to live Poisson model
as the match progresses.

Blending schedule:
  0'  → 100% pre-game / 0% live
  45' → 40% pre-game / 60% live  (halftime: live model has 45min of data)
  60' → 25% pre-game / 75% live
  70' → 15% pre-game / 85% live
  80' → 8%  pre-game / 92% live
  90' → 0%  pre-game / 100% live

The schedule uses a sigmoid curve for smooth transition.
Additionally, certainty adjustments are applied at extreme minutes
(e.g., 85'+ with a 1-goal lead → near-deterministic outcome).
"""

import math
import json
import logging
from dataclasses import dataclass, asdict
from typing import Optional

from db.schema import get_connection
from models.pre_game_scorer import PreGameScorer, SignalBreakdown
from models.live_poisson import LivePoissonModel, LivePoissonResult

logger = logging.getLogger(__name__)


# ── Sigmoid blend schedule ─────────────────────────────────────────────────────
# Weight for live model at minute t:
#   w_live(t) = sigmoid( (t - MID) / SCALE )
# Tuned so:  w_live(0) ≈ 0.05, w_live(45) ≈ 0.62, w_live(90) ≈ 0.97
SIGMOID_MID   = 40.0
SIGMOID_SCALE = 15.0

# Minimum/maximum live weights
LIVE_MIN = 0.05
LIVE_MAX = 0.97


@dataclass
class BlendedResult:
    match_id:      int
    minute:        int
    home_score:    int
    away_score:    int
    pre_game_prob: float
    live_prob:     float
    pre_weight:    float
    live_weight:   float
    blended_prob:  float
    score_state:   str
    confidence:    float


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def live_weight(minute: int) -> float:
    """Return the live-model weight [0, 1] for a given minute."""
    raw = sigmoid((minute - SIGMOID_MID) / SIGMOID_SCALE)
    return max(LIVE_MIN, min(LIVE_MAX, raw))


def pre_weight(minute: int) -> float:
    return 1.0 - live_weight(minute)


# ── Blender ───────────────────────────────────────────────────────────────────

class ProbabilityBlender:
    def __init__(self):
        self._scorer  = PreGameScorer()
        self._poisson = LivePoissonModel()

    def blend(
        self,
        match_id:     int,
        minute:       int,
        home_score:   int,
        away_score:   int,
        live_home_xg: Optional[float] = None,
        live_away_xg: Optional[float] = None,
    ) -> BlendedResult:
        # ── Pre-game probability ───────────────────────────────────────────────
        pre_prob = self._get_pre_game_prob(match_id)

        # ── Live Poisson probability ───────────────────────────────────────────
        live_result = self._poisson.compute(
            match_id, minute, home_score, away_score,
            live_home_xg, live_away_xg
        )
        live_prob = live_result.draw_prob

        # ── Blend ─────────────────────────────────────────────────────────────
        w_live = live_weight(minute)
        w_pre  = 1.0 - w_live
        blended = w_pre * pre_prob + w_live * live_prob

        # ── Late-game certainty clamp ──────────────────────────────────────────
        diff = home_score - away_score
        if minute >= 85:
            if abs(diff) >= 2:
                blended = blended * 0.05
            elif abs(diff) == 1:
                blended = blended * 0.35
        elif minute >= 75 and abs(diff) >= 2:
            blended = blended * 0.12

        blended = max(0.01, min(0.99, blended))

        # ── Confidence = weighted average of pre + live confidences ───────────
        pre_confidence  = self._get_pre_confidence(match_id)
        live_confidence = min(1.0, minute / 45.0)
        confidence = w_pre * pre_confidence + w_live * live_confidence

        result = BlendedResult(
            match_id      = match_id,
            minute        = minute,
            home_score    = home_score,
            away_score    = away_score,
            pre_game_prob = round(pre_prob, 4),
            live_prob     = round(live_prob, 4),
            pre_weight    = round(w_pre, 4),
            live_weight   = round(w_live, 4),
            blended_prob  = round(blended, 4),
            score_state   = live_result.score_state,
            confidence    = round(confidence, 3),
        )

        logger.info(
            "BLEND [%d'] %d-%d  pre=%.3f(×%.2f) live=%.3f(×%.2f) → %.3f",
            minute, home_score, away_score,
            pre_prob, w_pre, live_prob, w_live, blended
        )
        return result

    def _get_pre_game_prob(self, match_id: int) -> float:
        """Return cached pre-draw prob, computing it if missing."""
        conn = get_connection()
        c    = conn.cursor()
        c.execute("SELECT pre_draw_prob FROM matches WHERE id=?", (match_id,))
        row = c.fetchone()
        conn.close()

        if row and row["pre_draw_prob"] is not None:
            return row["pre_draw_prob"]

        # Not yet scored – compute now
        result = self._scorer.score(match_id)
        return result.pre_draw_prob if result else 0.26

    def _get_pre_confidence(self, match_id: int) -> float:
        return 0.7


# ── Blend weight schedule table ───────────────────────────────────────────────

def blend_schedule_table() -> str:
    header = f"{'Minute':>8} {'w_pre':>8} {'w_live':>8}"
    lines  = [header, "-" * 28]
    for t in [0, 15, 30, 45, 60, 70, 75, 80, 85, 90]:
        wl = live_weight(t)
        wp = 1.0 - wl
        lines.append(f"{t:>8} {wp:>8.4f} {wl:>8.4f}")
    return "\n".join(lines)


# ── Checkpoint blends ─────────────────────────────────────────────────────────

CHECKPOINTS = [60, 70, 80]


def blend_at_checkpoints(
    match_id:     int,
    home_score:   int,
    away_score:   int,
    live_home_xg: Optional[float] = None,
    live_away_xg: Optional[float] = None,
) -> dict:
    """Compute blended probability at each trade checkpoint minute."""
    blender = ProbabilityBlender()
    results = {}
    for minute in CHECKPOINTS:
        results[minute] = blender.blend(
            match_id, minute, home_score, away_score,
            live_home_xg, live_away_xg
        )
    return results


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Blend Weight Schedule ===")
    print(blend_schedule_table())

    print("\n=== Live Poisson demo: 0-0, xG 1.1-0.9 at checkpoints ===")
    from models.live_poisson import LivePoissonModel
    lm = LivePoissonModel()
    for minute in CHECKPOINTS:
        wl = live_weight(minute)
        wp = 1.0 - wl
        live_r = lm.compute(1, minute, 0, 0, 1.1, 0.9)
        pre_p  = 0.28
        blend  = wp * pre_p + wl * live_r.draw_prob
        print(
            f"  [{minute}'] pre={pre_p:.3f}×{wp:.2f}  "
            f"live={live_r.draw_prob:.3f}×{wl:.2f}  "
            f"→ blended={blend:.3f}"
        )
