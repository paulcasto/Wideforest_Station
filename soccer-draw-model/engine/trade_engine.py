"""
Layer 5 – Kalshi Trade Engine
Compares blended model probability to Kalshi contract price.
Fires BUY_YES / BUY_NO / PASS alerts at the 60', 70', 80' checkpoints.

Edge thresholds:
  BUY_YES when model_prob > contract_implied_prob + EDGE_THRESHOLD_BUY
  BUY_NO  when model_prob < contract_implied_prob - EDGE_THRESHOLD_SELL
  PASS    otherwise

Kelly criterion sizing (fractional) computed for position sizing reference.
"""

import os
import json
import time
import logging
from dataclasses import dataclass, asdict
from typing import Optional
from datetime import datetime, timezone

import requests

from db.schema import get_connection
from models.blender import ProbabilityBlender, CHECKPOINTS, live_weight

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [KALSHI] %(levelname)s %(message)s")

# ── Kalshi API ─────────────────────────────────────────────────────────────────
KALSHI_BASE   = "https://trading-api.kalshi.com/trade-api/v2"
KALSHI_KEY    = os.getenv("KALSHI_API_KEY", "")
KALSHI_SECRET = os.getenv("KALSHI_API_SECRET", "")

# ── Edge thresholds (in probability points) ────────────────────────────────────
EDGE_THRESHOLD_BUY  = 0.06    # need 6pp edge to BUY YES
EDGE_THRESHOLD_SELL = 0.06    # need 6pp edge to BUY NO (bet against draw)
MIN_CONFIDENCE      = 0.50    # skip if model confidence < 50%
MIN_VOLUME_GUARD    = 100     # skip thin markets (open interest < 100 contracts)

# ── Kelly fraction ────────────────────────────────────────────────────────────
KELLY_FRACTION = 0.25         # quarter-Kelly for risk management


@dataclass
class TradeAlert:
    match_id:       int
    kalshi_ticker:  str
    checkpoint_min: int
    model_prob:     float
    contract_price: float    # implied probability (0–1)
    edge:           float    # model_prob – contract_price
    action:         str      # 'BUY_YES' | 'BUY_NO' | 'PASS'
    kelly_fraction: float    # suggested bet size as fraction of bankroll
    pre_game_prob:  float
    live_prob:      float
    pre_weight:     float
    live_weight:    float
    score_state:    str
    confidence:     float
    fired_at:       str


class KalshiClient:
    """Thin wrapper for Kalshi REST API."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {KALSHI_KEY}",
            "Content-Type":  "application/json",
        })

    def get_market(self, ticker: str) -> Optional[dict]:
        """Fetch market details including yes_bid / yes_ask prices."""
        try:
            r = self.session.get(
                f"{KALSHI_BASE}/markets/{ticker}", timeout=10
            )
            r.raise_for_status()
            return r.json().get("market", {})
        except requests.HTTPError as e:
            logger.error("Kalshi API error for %s: %s", ticker, e)
            return None

    def get_orderbook(self, ticker: str) -> Optional[dict]:
        """Fetch current orderbook for spread analysis."""
        try:
            r = self.session.get(
                f"{KALSHI_BASE}/markets/{ticker}/orderbook", timeout=10
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.error("Orderbook error for %s: %s", ticker, e)
            return None


class TradeEngine:
    """
    Polls Kalshi prices for live matches and fires trade alerts
    at the 60', 70', 80' checkpoints.
    """

    def __init__(self):
        self.blender = ProbabilityBlender()
        self.kalshi  = KalshiClient()

    # ── Main checkpoint evaluation ─────────────────────────────────────────────

    def evaluate_checkpoint(
        self,
        match_id:     int,
        minute:       int,
        home_score:   int,
        away_score:   int,
        live_home_xg: Optional[float] = None,
        live_away_xg: Optional[float] = None,
        kalshi_ticker: str = "",
    ) -> Optional[TradeAlert]:
        """
        Evaluate a single checkpoint minute.
        Returns TradeAlert (may have action='PASS').
        """
        if minute not in CHECKPOINTS:
            logger.warning("Checkpoint %d not in %s", minute, CHECKPOINTS)
            return None

        # ── Get blended model probability ─────────────────────────────────────
        blend = self.blender.blend(
            match_id, minute, home_score, away_score,
            live_home_xg, live_away_xg
        )

        if blend.confidence < MIN_CONFIDENCE:
            logger.info(
                "[%d'] Skipping – low model confidence %.2f",
                minute, blend.confidence
            )
            return None

        # ── Get Kalshi price ──────────────────────────────────────────────────
        contract_price = self._get_contract_price(match_id, kalshi_ticker)
        if contract_price is None:
            logger.info("[%d'] No Kalshi price available for %s", minute, kalshi_ticker)
            contract_price = 0.26   # fallback to market average

        # ── Compute edge and action ───────────────────────────────────────────
        edge   = blend.blended_prob - contract_price
        action = self._decide_action(edge, blend.blended_prob, contract_price)

        # ── Kelly sizing ──────────────────────────────────────────────────────
        kelly = self._kelly(blend.blended_prob, contract_price, action)

        alert = TradeAlert(
            match_id       = match_id,
            kalshi_ticker  = kalshi_ticker or f"SOCCER-DRAW-{match_id}",
            checkpoint_min = minute,
            model_prob     = blend.blended_prob,
            contract_price = contract_price,
            edge           = round(edge, 4),
            action         = action,
            kelly_fraction = round(kelly, 4),
            pre_game_prob  = blend.pre_game_prob,
            live_prob      = blend.live_prob,
            pre_weight     = blend.pre_weight,
            live_weight    = blend.live_weight,
            score_state    = blend.score_state,
            confidence     = blend.confidence,
            fired_at       = datetime.now(timezone.utc).isoformat(),
        )

        self._save_alert(alert)
        self._log_alert(alert)
        return alert

    def _decide_action(
        self, edge: float, model_prob: float, contract_price: float
    ) -> str:
        if edge > EDGE_THRESHOLD_BUY:
            return "BUY_YES"
        elif edge < -EDGE_THRESHOLD_SELL:
            return "BUY_NO"
        else:
            return "PASS"

    def _kelly(
        self, model_prob: float, contract_price: float, action: str
    ) -> float:
        """
        Fractional Kelly criterion.
        For BUY_YES: b = (1/contract_price) - 1
        For BUY_NO:  b = (1/(1-contract_price)) - 1
        Kelly = (b*p - q) / b  × KELLY_FRACTION
        """
        if action == "PASS":
            return 0.0
        if action == "BUY_YES":
            b = (1.0 / max(contract_price, 0.01)) - 1
            p = model_prob
        else:  # BUY_NO
            b = (1.0 / max(1 - contract_price, 0.01)) - 1
            p = 1.0 - model_prob

        q     = 1.0 - p
        kelly = max(0.0, (b * p - q) / b) * KELLY_FRACTION
        return min(kelly, 0.20)   # cap at 20% of bankroll

    # ── Kalshi price resolution ────────────────────────────────────────────────

    def _get_contract_price(
        self, match_id: int, ticker: str
    ) -> Optional[float]:
        """
        1. Check DB for cached price (within last 3 min).
        2. Fetch live from Kalshi API.
        3. Return mid-market price as probability (0–1).
        """
        conn = get_connection()
        c    = conn.cursor()
        c.execute("""
            SELECT yes_price, fetched_at FROM kalshi_contracts
            WHERE (match_id=? OR ticker=?)
            ORDER BY fetched_at DESC LIMIT 1
        """, (match_id, ticker))
        row = c.fetchone()
        conn.close()

        if row and row["yes_price"] is not None:
            fetched  = datetime.fromisoformat(row["fetched_at"])
            age_secs = (datetime.now(timezone.utc) - fetched.replace(tzinfo=timezone.utc)).total_seconds()
            if age_secs < 180:
                return row["yes_price"] / 100.0

        # Fetch from API
        if ticker and KALSHI_KEY:
            market = self.kalshi.get_market(ticker)
            if market:
                yes_ask = market.get("yes_ask", 0)
                yes_bid = market.get("yes_bid", 0)
                mid     = (yes_ask + yes_bid) / 2
                open_interest = market.get("open_interest", 0)

                self._cache_contract(match_id, ticker, mid, open_interest, market)

                if open_interest < MIN_VOLUME_GUARD:
                    logger.warning("Thin market %s (OI=%d)", ticker, open_interest)

                return mid / 100.0

        return None

    def _cache_contract(
        self, match_id: int, ticker: str,
        yes_price: float, open_interest: int, raw: dict
    ):
        conn = get_connection()
        conn.execute("""
            INSERT INTO kalshi_contracts
                (match_id, ticker, market_title, yes_price, open_interest)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(ticker) DO UPDATE SET
                yes_price     = excluded.yes_price,
                open_interest = excluded.open_interest,
                fetched_at    = datetime('now')
        """, (match_id, ticker, raw.get("title", ""), yes_price, open_interest))
        conn.commit()
        conn.close()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save_alert(self, alert: TradeAlert):
        conn = get_connection()
        blend_meta = json.dumps({
            "pre_weight":  alert.pre_weight,
            "live_weight": alert.live_weight,
        })
        conn.execute("""
            INSERT INTO trade_alerts
                (match_id, kalshi_ticker, checkpoint_min, model_prob,
                 contract_price, edge, action, blend_weights,
                 pre_game_prob, live_prob, fired_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            alert.match_id, alert.kalshi_ticker, alert.checkpoint_min,
            alert.model_prob, alert.contract_price, alert.edge,
            alert.action, blend_meta, alert.pre_game_prob,
            alert.live_prob, alert.fired_at,
        ))
        conn.commit()
        conn.close()

    def _log_alert(self, alert: TradeAlert):
        icon = {"BUY_YES": "[YES]", "BUY_NO": "[NO]", "PASS": "[--]"}.get(alert.action, "")
        logger.info(
            "%s [%d'] %s | model=%.3f Kalshi=%.3f edge=%+.3f Kelly=%.2f%% | %s",
            icon, alert.checkpoint_min, alert.kalshi_ticker,
            alert.model_prob, alert.contract_price, alert.edge,
            alert.kelly_fraction * 100,
            alert.score_state,
        )


# ── Match monitor (poll loop) ──────────────────────────────────────────────────

class MatchMonitor:
    """
    Polls a live match every POLL_INTERVAL seconds.
    Fires evaluate_checkpoint at 60', 70', 80'.
    """
    POLL_INTERVAL = 30   # seconds

    def __init__(self, match_id: int, fd_code: str = "PL"):
        self.match_id = match_id
        self.fd_code  = fd_code
        self.engine   = TradeEngine()
        self._fired   = set()   # checkpoints already fired

    def run(self):
        from fetchers.football_data import update_live_scores

        logger.info("Starting monitor for match_id=%d", self.match_id)

        while True:
            # Refresh live score
            update_live_scores(self.fd_code)
            state = self._current_state()
            if not state:
                time.sleep(self.POLL_INTERVAL)
                continue

            minute = state["minute"] or 0
            logger.debug("Poll: [%d'] %d-%d", minute, state["home_score"] or 0, state["away_score"] or 0)

            # Fire checkpoints in order
            for cp in CHECKPOINTS:
                if minute >= cp and cp not in self._fired:
                    logger.info("Checkpoint %d' reached", cp)
                    self.engine.evaluate_checkpoint(
                        match_id      = self.match_id,
                        minute        = cp,
                        home_score    = state["home_score"] or 0,
                        away_score    = state["away_score"] or 0,
                        live_home_xg  = state.get("live_home_xg"),
                        live_away_xg  = state.get("live_away_xg"),
                        kalshi_ticker = state.get("kalshi_ticker", ""),
                    )
                    self._fired.add(cp)

            # Stop after final checkpoint + some buffer, or match finished
            if state["status"] == "FINISHED" or (minute > 92 and len(self._fired) == len(CHECKPOINTS)):
                logger.info("Match complete. All checkpoints fired: %s", self._fired)
                break

            time.sleep(self.POLL_INTERVAL)

    def _current_state(self) -> Optional[dict]:
        conn = get_connection()
        c    = conn.cursor()
        c.execute("""
            SELECT m.home_score, m.away_score, m.status,
                   kc.ticker AS kalshi_ticker,
                   le.minute, le.live_home_xg, le.live_away_xg
            FROM matches m
            LEFT JOIN kalshi_contracts kc ON kc.match_id = m.id
            LEFT JOIN live_events le ON le.match_id = m.id
                AND le.id = (SELECT MAX(id) FROM live_events WHERE match_id=m.id)
            WHERE m.id = ?
        """, (self.match_id,))
        row = c.fetchone()
        conn.close()
        if not row:
            return None
        return dict(row)


# ── Backtest helper ────────────────────────────────────────────────────────────

def backtest_alerts(fd_code: str = "PL", limit: int = 50) -> dict:
    """
    Simulate trade alerts on historical matches (where we know outcome).
    Returns simple P&L stats.
    """
    conn = get_connection()
    c    = conn.cursor()

    c.execute("""
        SELECT m.id, m.home_score, m.away_score, m.is_draw,
               m.home_xg, m.away_xg, m.pre_draw_prob
        FROM matches m
        JOIN leagues l ON l.id = m.league_id
        WHERE l.fd_code=? AND m.status='FINISHED' AND m.is_draw IS NOT NULL
          AND m.pre_draw_prob IS NOT NULL
        ORDER BY m.match_date DESC
        LIMIT ?
    """, (fd_code, limit))
    rows = c.fetchall()
    conn.close()

    engine  = TradeEngine()
    results = {"buy_yes": [], "buy_no": [], "pass": []}

    for row in rows:
        for minute in CHECKPOINTS:
            alert = engine.evaluate_checkpoint(
                match_id      = row["id"],
                minute        = minute,
                home_score    = row["home_score"] or 0,
                away_score    = row["away_score"] or 0,
                live_home_xg  = row["home_xg"],
                live_away_xg  = row["away_xg"],
                kalshi_ticker = "",
            )
            if not alert:
                continue

            is_draw = bool(row["is_draw"])
            action  = alert.action
            win     = (action == "BUY_YES" and is_draw) or (action == "BUY_NO" and not is_draw)
            pnl_units = (1 - alert.contract_price) if win else -alert.contract_price

            results[action.lower()].append({
                "match_id": row["id"], "minute": minute,
                "model_prob": alert.model_prob, "price": alert.contract_price,
                "edge": alert.edge, "is_draw": is_draw, "win": win,
                "pnl": pnl_units,
            })

    # Summary stats
    summary = {}
    for action, trades in results.items():
        if not trades:
            summary[action] = {"count": 0}
            continue
        wins  = sum(1 for t in trades if t["win"])
        total = len(trades)
        pnl   = sum(t["pnl"] for t in trades)
        summary[action] = {
            "count":           total,
            "win_rate":        round(wins / total, 3),
            "total_pnl_units": round(pnl, 3),
            "avg_edge":        round(sum(t["edge"] for t in trades) / total, 4),
        }

    return summary


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from db.schema import create_schema, seed_leagues

    create_schema()
    seed_leagues()

    if "--backtest" in sys.argv:
        print("Running backtest on PL historical matches...")
        stats = backtest_alerts("PL", limit=20)
        print(json.dumps(stats, indent=2))

    elif "--eval" in sys.argv:
        engine = TradeEngine()
        alert = engine.evaluate_checkpoint(
            match_id      = 1,
            minute        = 70,
            home_score    = 1,
            away_score    = 1,
            live_home_xg  = 1.2,
            live_away_xg  = 1.0,
            kalshi_ticker = "SOCCER-EPL-DRAW-TEST",
        )
        if alert:
            print(json.dumps(asdict(alert), indent=2))

    else:
        print("Usage: python -m engine.trade_engine [--backtest | --eval]")
