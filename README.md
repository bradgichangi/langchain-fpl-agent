# LangChain FPL Agent

Fantasy Premier League assistant powered by:
- `uAgents` for transport/chat protocol
- `deepagents` + LangChain for tool-using reasoning
- Live FPL API data + local scoring/ranking modules

It can answer manager-specific questions, rank players with a composite scoring model, evaluate upcoming gameweek blanks, and suggest chip timing opportunities.

## Features

- Manager analysis (team, picks, bank, value, free transfers)
- Upcoming gameweek awareness (fixtures, deadline, blanking players)
- Composite player ranking engine (`scoring_module.py`) with:
  - fixture difficulty
  - form/value
  - xG/xA
  - availability
  - transfer momentum
  - clean sheet / bonus rate
  - set-piece duties
- Chip timing evaluator (`chip_opportunity_module.py`) for:
  - Wildcard
  - Free Hit
  - Bench Boost
  - Triple Captain
- Session memory for `manager_id` so users do not need to repeat it every message

## Project Structure

- `agent.py` - main runtime, chat protocol, deep agent wiring, session memory
- `tools/agent_tools.py` - tool layer for manager, fixture, ranking, chip analysis
- `tools/fixtures.py` - fixture fetch + team-name enrichment
- `tools/fpl_static.py` - local data loaders
- `scoring_module.py` - composite scoring engine + data mapper
- `chip_opportunity_module.py` - chip timing engine
- `scripts/fetch_fpl_static.py` - build local static snapshot
- `scripts/rank_all_players.py` - generate ranked player JSON
- `data/fpl_static.json` - local FPL static snapshot
- `data/fpl_player_rankings.json` - generated ranked player dataset
- `data/session_manager_ids.json` - persisted sender -> manager ID mapping

## Requirements

- Python 3.11+
- FPL API network access
- ASI1 API key for the LLM backend used by this project

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment Variables

Create `.env.local` in the project root:

```bash
ASI1_API_KEY=your_key_here
```

## Quick Start

### 1) Refresh static data (optional but recommended)

```bash
.venv/bin/python scripts/fetch_fpl_static.py
```

### 2) Build player rankings

```bash
.venv/bin/python scripts/rank_all_players.py
```

This writes `data/fpl_player_rankings.json`.

### 3) Run the agent

```bash
.venv/bin/python agent.py
```

You should see your agent address in stdout.

## Core Tools Exposed to the Agent

- `get_fpl_manager_data(manager_id)`
- `get_fpl_manager_current_team(manager_id, gameweek=None)`
- `get_fpl_upcoming_gameweek()`
- `get_fpl_fixtures(gameweek=None)`
- `get_fpl_player(player_id)`
- `search_fpl_players(name_query)`
- `get_fpl_top_players(metric, limit, position=None)`
- `get_fpl_scored_rankings(...)`
- `get_fpl_chip_opportunities(manager_id, gameweek=None, horizon_gws=6)`

## Session Manager ID Memory

The agent stores manager IDs per sender/session in `data/session_manager_ids.json`.

- If a user says `my id is 9793856`, future manager-specific requests reuse it.
- Users can clear it with phrases like `forget manager id` or `clear my id`.

## Scoring Model Notes

`scoring_module.py` outputs a normalized composite score (`0.0` to `1.0`) and factor breakdown.  
Tiers are applied by percentile over the full ranked population:

- MUST START
- STRONG PICK
- VIABLE OPTION
- RISKY PICK
- AVOID

Set-piece responsibility is included using FPL taker-order fields:
- `penalties_order`
- `direct_freekicks_order`
- `corners_and_indirect_freekicks_order`

## Chip Opportunity Module

`chip_opportunity_module.py` evaluates available chips for the target gameweek and returns:

- score
- confidence
- triggers met
- blockers
- recommended gameweek
- wait reason

The tool integration in `get_fpl_chip_opportunities` automatically prepares the required squad/bench/fixture/ranking context.

## Notes

- FPL endpoints can return `404` for certain future pick states; chip tool includes fallback handling.
- Ranked player data is static until `scripts/rank_all_players.py` is rerun.
- Live manager/team/fixture tools use real-time FPL endpoints.
