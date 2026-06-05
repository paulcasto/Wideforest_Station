"""
Soccer Draw Probability Model – Main Runner
============================================
Usage:
  python main.py --setup              # Create DB schema, seed leagues
  python main.py --fetch PL 2024      # Pull Football-Data.org matches + standings
  python main.py --xg EPL 2024        # Sync Understat xG data
  python main.py --stats PL           # Compute league draw stats
  python main.py --score              # Pre-game score upcoming matches (next 3 days)
  python main.py --live PL            # Update live scores
  python main.py --eval 1 70 1 1      # Evaluate match_id=1 at 70' (1-1 score)
  python main.py --monitor 1 PL       # Start live match monitor for match_id=1
  python main.py --backtest PL 50     # Backtest on last 50 finished PL matches

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

    else:
        p.print_help()


if __name__ == "__main__":
    main()
