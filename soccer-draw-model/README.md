# Soccer Draw Probability Model for Kalshi Trading

A multi-layer quantitative model that estimates the probability of a soccer match ending in a draw, combining pre-game signals (xG similarity, head-to-head history, league table position, team form) with a live in-game Poisson model, then applies a sigmoid blending schedule and quarter-Kelly sizing to identify edge against Kalshi draw contracts at the 60', 70', and 80' checkpoints.

---

## Installation

```bash
git clone <repo-url>
cd soccer-draw-model

python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env with your API keys
```

---

## API Key Setup

### Football-Data.org
1. Register for a free account at https://www.football-data.org/
2. The free tier covers PL, PD, BL1, SA, FL1 with ~10 requests/minute
3. Copy your API token into `.env` as `FD_API_KEY`

### Kalshi
1. Create an account at https://kalshi.com/
2. Generate API credentials in your account settings (API section)
3. Copy your API key and secret into `.env` as `KALSHI_API_KEY` and `KALSHI_API_SECRET`

> **Note on Kalshi contract tickers:** Kalshi does not provide automatic match-to-ticker mapping. You must manually look up the soccer draw contract ticker for each match on the Kalshi website and insert it into the `kalshi_contracts` table using a direct SQL query or a custom script. Example:
> ```sql
> INSERT INTO kalshi_contracts (match_id, contract_ticker, yes_price, no_price, fetched_at)
> VALUES (42, 'SOCCER-EPL-DRAW-2024-12-15-MCI-LIV', 0.28, 0.72, datetime('now'));
> ```

---

## CLI Usage

### Initialize the database
```bash
python main.py --setup
```
Creates `data/soccer_draws.db` with all tables.

### Fetch match data for a date
```bash
python main.py --fetch 2024-12-15
```
Fetches all matches on 2024-12-15 for PL, PD, BL1, SA, FL1 and stores them in the DB.

### Sync xG data from Understat
```bash
python main.py --xg 2024
```
Scrapes Understat for all five leagues for the 2024 season and upserts xG data.

### Recompute stats for a match
```bash
python main.py --stats 42
```
Recomputes h2h stats, team form, and league table for the teams/league of match 42.

### Pre-game draw probability
```bash
python main.py --score 42
```
Runs the 5-signal pre-game scorer and prints a JSON result with probability, raw score, signal breakdown, and confidence.

### Live model + blender
```bash
python main.py --live 42 65 1 0
```
Runs the live Poisson model at minute 65 with score 1-0, blends with pre-game prob, and prints full result.

### Full evaluation (pre-game + live + Kalshi decision)
```bash
python main.py --eval 42 65 1 0
```
Runs all layers and checks whether to buy YES, buy NO, or pass on the Kalshi draw contract. Saves any trade alert to the DB.

### Live monitoring loop
```bash
python main.py --monitor PL
```
Polls Football-Data.org every 60 seconds for live Premier League matches. At each checkpoint minute (60', 70', 80'), runs a full evaluation and prints/saves any trade alerts. Press Ctrl+C to stop.

### Backtest pre-game model
```bash
python main.py --backtest PL 2024
```
Evaluates the pre-game scorer against all finished PL 2024 matches. Reports Brier score, directional accuracy, and per-bucket calibration.

---

## Model Architecture

### Layer 1: Pre-Game Scorer (`src/models/pre_game.py`)

Five weighted signals produce a calibrated draw probability before kick-off:

| Signal | Weight | Description |
|---|---|---|
| `xg_similarity` | 0.30 | How similar home and away recent xG rates are. Computed as `1 - |avg_home_xg - avg_away_xg| / (avg_home_xg + avg_away_xg + 0.001)`. High similarity means evenly matched teams → higher draw probability. |
| `h2h_draw_rate` | 0.20 | Historical draw rate from head-to-head matches in the `h2h_stats` table. Regressed toward league average (0.27) for small samples. |
| `league_draw_rate` | 0.20 | Overall draw rate for this league and season, computed from `league_table`. Varies by competition (Italian Serie A draws ~28%, Bundesliga ~24%). |
| `table_gap_signal` | 0.15 | Points gap between teams in the league table. Gap 0-3 → 1.0, 4-6 → 0.7, 7-10 → 0.5, >10 → 0.3. Close table positions indicate competitive balance and higher draw chances. |
| `form_draw_rate` | 0.15 | Average of home and away teams' draw rates from their last 5 matches (`team_form` table). |

The weighted sum is logistically calibrated:
```
prob = 1 / (1 + exp(-6 * (raw_score - 0.5)))
```
This maps a raw score of 0.5 to 27% draw probability (European average), with steeper changes as signals align or diverge.

---

### Layer 2: Live Poisson Model (`src/models/live_poisson.py`)

During the match, two independent Poisson processes model remaining goals for home and away teams.

**Rate blending:**
- Pre-game rate: `lambda_pregame = avg_xG / 90` (per minute)
- Live rate: `lambda_live = cumulative_xG_so_far / minutes_elapsed`
- Blended: `rate = alpha * lambda_live + (1 - alpha) * lambda_pregame`
- Alpha: `alpha = min(minute / 90, 1)` — linearly increases from 0 to 1 over the match

**Remaining expected goals:**
```
lambda_remaining = blended_rate * remaining_minutes
```

**Draw probability from current state:**
Given current score (H, A) and remaining Poisson rates, the model sums over all goal additions (h_add, a_add) where the final score is a draw:
```
P(draw) = Σ P(home scores h_add) × P(away scores a_add)
          where (H + h_add) == (A + a_add)
```

---

### Layer 3: Probability Blender (`src/models/blender.py`)

Sigmoid weighting schedule smoothly transitions from pre-game to live model:

```
live_weight = 1 / (1 + exp(-0.15 * (minute - 45)))
pregame_weight = 1 - live_weight
```

| Minute | Pre-game weight | Live weight |
|---|---|---|
| 0 | ~88% | ~12% |
| 45 | 50% | 50% |
| 90 | ~12% | ~88% |

```
blended_prob = pregame_weight × pregame_prob + live_weight × live_prob
```

---

### Layer 4: Kalshi Trade Engine (`src/models/kalshi_engine.py`)

**Checkpoints:** 60', 70', 80'

**Decision logic:**
- `YES_BUY`: `model_prob > yes_price + 0.05` (model says draw more likely than market)
- `NO_BUY`: `(1 - model_prob) > no_price + 0.05` (model says draw less likely than market)
- `PASS`: otherwise (insufficient edge)

**Quarter-Kelly sizing:**
```
# YES bet
edge = model_prob - market_prob
kelly_full = edge / (1 - market_prob)

# NO bet
edge = (1 - model_prob) - no_price
kelly_full = edge / no_price

kelly_quarter = kelly_full / 4
suggested_contracts = floor(kelly_quarter × bankroll / contract_value)
```

Default: `bankroll = $1,000`, `contract_value = $1.00`. Quarter-Kelly reduces full Kelly by 75% to account for model uncertainty.

---

## Database Schema

| Table | Purpose |
|---|---|
| `leagues` | Competition metadata (name, country, fd_code, understat_name) |
| `teams` | Team records with Football-Data and Understat IDs |
| `matches` | Match schedule and results |
| `xg_data` | Pre-match and post-match xG from Understat |
| `live_events` | In-game events with cumulative xG tracking |
| `h2h_stats` | Head-to-head draw rates and xG averages |
| `team_form` | Last 5 match results, draw rates, average xG |
| `league_table` | Live standings with draw rates per team |
| `kalshi_contracts` | Manually entered contract tickers and prices |
| `trade_alerts` | Generated trade alerts with Kelly sizing |

The database is stored at `data/soccer_draws.db` as a single SQLite file.

---

## Notes

- **Rate limiting:** Football-Data.org free tier allows ~10 requests/minute. The fetcher enforces a 6-second delay between requests automatically.
- **Understat scraping:** Understat embeds JSON as URL-encoded strings in `<script>` tags. The fetcher uses `urllib.parse.unquote` to decode and `BeautifulSoup` to parse the HTML.
- **Kalshi tickers:** Must be manually mapped — Kalshi does not offer a public soccer match API. Insert prices directly into `kalshi_contracts` before running `--eval` or `--monitor`.
- **Confidence score:** The pre-game scorer reports a `confidence` value (0.0–1.0) representing the fraction of signals backed by real data. A confidence below 0.6 suggests treating the probability estimate with caution.
