"""
Soccer Draw Probability Model – Main Runner
============================================
Usage:
  python main.py --setup                     # Create DB schema, seed leagues
  python main.py --fetch PL 2024             # Pull Football-Data.org matches + standings
  python main.py --xg EPL 2024               # Sync Understat xG data
  python main.py --stats PL                  # Compute league draw stats
  python main.py --score                     # Pre-game score upcoming matches (next 3 days)
  python main.py --live PL                   # Update live scores
  python main.py --eval 1 70 1 1             # Evaluate match_id=1 at 70' (1-1 score)
  python main.py --monitor 1 PL              # Start live match monitor for match_id=1
  python main.py --backtest PL 50            # Backtest on last 50 finished PL matches
  python main.py --tickers                   # Match Kalshi markets → DB (next 7 days)
  python main.py --tickers 2025-01-10 2025-01-17  # Same, explicit date range
  python main.py --watchlist                 # Print upcoming matches with Kalshi tickers
  python main.py --watchlist 14              # Same, 14-day lookahead

Environment variables:
  FD_API_KEY        – Football-Data.org API key (free tier: 10 calls/min)
  KALSHI_API_KEY    – Kalshi API key
  KALSHI_API_SECRET – Kalshi API secret
"""

import sys
import argparse
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(levelname)s %(message)s")


def main():
    p = argparse.ArgumentParser(description="Soccer Draw Model – Kalshi Trade Engine")
    p.add_argument("--setup",    action="store_true",
                   help="Create DB schema, seed leagues")
    p.add_argument("--fetch",    nargs=2, metavar=("FD_CODE", "SEASON_YEAR"),
                   help="Fetch matches + standings from Football-Data.org")
    p.add_argument("--xg",       nargs=2, metavar=("US_NAME", "SEASON_YEAR"),
                   help="Sync Understat xG data (US_NAME e.g. EPL, La_liga)")
    p.add_argument("--stats",    metavar="FD_CODE",
                   help="Compute and display league draw stats")
    p.add_argument("--score",    action="store_true",
                   help="Pre-game score upcoming matches (next 3 days)")
    p.add_argument("--live",     metavar="FD_CODE",
                   help="Update live scores for a competition")
    p.add_argument("--eval",     nargs=4, metavar=("MATCH_ID", "MINUTE", "H_SCORE", "A_SCORE"),
                   help="Full evaluation: pre-game + live + blended + Kalshi decision")
    p.add_argument("--monitor",  nargs=2, metavar=("MATCH_ID", "FD_CODE"),
                   help="Start live match monitor (polls every 30s, fires at 60/70/80')")
    p.add_argument("--backtest", nargs=2, metavar=("FD_CODE", "LIMIT"),
                   help="Backtest trade alerts on historical finished matches")
    p.add_argument("--tickers", nargs="*", metavar="DATE",
                   help="Sync Kalshi draw-market tickers to DB. "
                        "Optional: DATE_FROM DATE_TO (YYYY-MM-DD). "
                        "Defaults to today + 7 days.")
    p.add_argument("--watchlist", nargs="?", const=7, type=int, metavar="DAYS",
                   help="Print upcoming matches with matched Kalshi tickers "
                        "and live yes_price. Optional: days lookahead (default 7).")
    args = p.parse_args()

    if args.setup:
        from db.schema import create_schema, seed_leagues
        create_schema()
        seed_leagues()
        print("Database ready.")

    elif args.fetch:
        from fetchers.football_data import fetch_matches_for_league, fetch_standings
        fd_code, year = args.fetch[0], int(args.fetch[1])
        n = fetch_matches_for_league(fd_code, year)
        fetch_standings(fd_code, year)
        print(f"Fetched {n} matches + standings for {fd_code}/{year}")

    elif args.xg:
        from fetchers.understat import sync_xg_to_matches
        us_name, year = args.xg[0], int(args.xg[1])
        n = sync_xg_to_matches(us_name, year)
        print(f"Synced xG for {n} matches ({us_name}/{year})")

    elif args.stats:
        from fetchers.understat import compute_league_draw_stats
        import json
        stats = compute_league_draw_stats(args.stats)
        print(json.dumps(stats, indent=2))

    elif args.score:
        from models.pre_game_scorer import score_upcoming_matches
        results = score_upcoming_matches(days_ahead=3)
        if not results:
            print("No upcoming scheduled matches found in DB.")
        for r in results:
            print(f"pre_draw_prob={r.pre_draw_prob:.3f}  confidence={r.confidence:.2f}")

    elif args.live:
        from fetchers.football_data import update_live_scores
        n = update_live_scores(args.live)
        print(f"Updated {n} live goal events for {args.live}")

    elif args.eval:
        match_id, minute, h_score, a_score = (
            int(args.eval[0]), int(args.eval[1]),
            int(args.eval[2]), int(args.eval[3])
        )
        from engine.trade_engine import TradeEngine
        import json
        from dataclasses import asdict
        engine = TradeEngine()
        alert  = engine.evaluate_checkpoint(
            match_id   = match_id,
            minute     = minute,
            home_score = h_score,
            away_score = a_score,
        )
        if alert:
            print(json.dumps(asdict(alert), indent=2))
        else:
            print("No alert generated (low confidence or minute not a checkpoint).")

    elif args.monitor:
        match_id, fd_code = int(args.monitor[0]), args.monitor[1]
        from engine.trade_engine import MatchMonitor
        monitor = MatchMonitor(match_id, fd_code)
        monitor.run()

    elif args.backtest:
        fd_code, limit = args.backtest[0], int(args.backtest[1])
        from engine.trade_engine import backtest_alerts
        import json
        stats = backtest_alerts(fd_code, limit)
        print(json.dumps(stats, indent=2))

    elif args.tickers is not None:
        from fetchers.kalshi_markets import sync_tickers
        import json
        dates = args.tickers  # [] or [from] or [from, to]
        date_from = dates[0] if len(dates) > 0 else None
        date_to   = dates[1] if len(dates) > 1 else None
        result = sync_tickers(date_from=date_from, date_to=date_to)
        print(f"Fetched {result['fetched']} markets  |  "
              f"Matched {result['matched']}  |  "
              f"Unmatched {result['unmatched']}")
        if result["unmatched_titles"]:
            print("\nUnmatched markets (add aliases or fetch missing teams):")
            for t in result["unmatched_titles"]:
                print(f"  {t}")

    elif args.watchlist is not None:
        from fetchers.kalshi_markets import get_watchlist
        days  = args.watchlist
        rows  = get_watchlist(days_ahead=days)
        if not rows:
            print(f"No upcoming matches with Kalshi tickers in the next {days} days.")
            print("Run  python main.py --tickers  first to match markets.")
        else:
            _print_watchlist(rows)

    else:
        p.print_help()


def _print_watchlist(rows: list) -> None:
    """Pretty-print the watchlist table."""
    col_w = {"date": 16, "match": 40, "ticker": 32, "yes": 8, "oi": 8, "prob": 8}
    header = (
        f"{'DATE':<{col_w['date']}}"
        f"{'MATCH':<{col_w['match']}}"
        f"{'KALSHI TICKER':<{col_w['ticker']}}"
        f"{'YES':>{col_w['yes']}}"
        f"{'OI':>{col_w['oi']}}"
        f"{'PRE-PROB':>{col_w['prob']}}"
    )
    sep = "─" * len(header)
    print(sep)
    print(header)
    print(sep)

    current_date = None
    for r in rows:
        match_date = (r["match_date"] or "")[:16].replace("T", " ")
        if match_date[:10] != current_date:
            if current_date is not None:
                print()
            current_date = match_date[:10]

        match_str   = f"{r['home']} vs {r['away']}"
        ticker      = r["ticker"] or ""
        yes_price   = r["yes_price"]
        oi          = r["open_interest"] or 0
        pre_prob    = r["pre_draw_prob"]

        yes_str  = f"{yes_price:.1f}¢" if yes_price is not None else "  —  "
        oi_str   = str(oi) if oi else "—"
        prob_str = f"{pre_prob*100:.1f}%" if pre_prob is not None else "  —  "

        # Highlight edge if model probability meaningfully differs from market
        edge_flag = ""
        if yes_price is not None and pre_prob is not None:
            edge = pre_prob - yes_price / 100
            if edge > 0.06:
                edge_flag = f"  <-- +{edge*100:.1f}pp EDGE"
            elif edge < -0.06:
                edge_flag = f"  <-- {edge*100:.1f}pp"

        print(
            f"{match_date:<{col_w['date']}}"
            f"{match_str[:col_w['match']-1]:<{col_w['match']}}"
            f"{ticker:<{col_w['ticker']}}"
            f"{yes_str:>{col_w['yes']}}"
            f"{oi_str:>{col_w['oi']}}"
            f"{prob_str:>{col_w['prob']}}"
            f"{edge_flag}"
        )
    print(sep)
    print(f"  {len(rows)} match(es) with Kalshi coverage")


if __name__ == "__main__":
    main()
