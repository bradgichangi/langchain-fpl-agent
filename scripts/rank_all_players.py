#!/usr/bin/env python3
"""Rank every FPL player with scoring_module.FPLScorer and write a static JSON.

Fetches the full bootstrap-static + fixtures from the FPL API (needed because
data/fpl_static.json is a slim snapshot missing stats like xG/xA, bonus,
minutes, etc. that the scorer consumes), scores every player, and writes
data/fpl_player_rankings.json with players ranked descending by composite score.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scoring_module import (  # noqa: E402
    FPLDataMapper,
    FPLScorer,
    FixtureInfo,
    Position,
    apply_percentile_tiers,
)

OUT = ROOT / "data" / "fpl_player_rankings.json"

BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"

N_FIXTURES = 5


def fetch_json(url: str):
    with urlopen(url, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def build_team_fixtures(
    fixtures: list[dict],
    team_ids: list[int],
    n: int,
) -> dict[int, list[FixtureInfo]]:
    """Map team_id -> next n unfinished FixtureInfo objects, kickoff-ordered."""
    unfinished = [fx for fx in fixtures if not fx.get("finished")]
    unfinished.sort(key=lambda fx: (fx.get("kickoff_time") or "", fx.get("event") or 0))
    out: dict[int, list[FixtureInfo]] = {tid: [] for tid in team_ids}
    for fx in unfinished:
        for tid, key_diff, is_home in (
            (fx.get("team_h"), "team_h_difficulty", True),
            (fx.get("team_a"), "team_a_difficulty", False),
        ):
            if tid in out and len(out[tid]) < n:
                out[tid].append(
                    FixtureInfo(
                        difficulty=fx.get(key_diff) or 3,
                        is_home=is_home,
                        gameweek=fx.get("event") or 0,
                    )
                )
    return out


def main() -> None:
    print("Fetching bootstrap-static ...")
    bootstrap = fetch_json(BOOTSTRAP_URL)
    print("Fetching fixtures ...")
    fixtures = fetch_json(FIXTURES_URL)

    teams = {t["id"]: t for t in bootstrap.get("teams", [])}
    team_fixtures = build_team_fixtures(fixtures, list(teams.keys()), N_FIXTURES)

    position_name = {
        1: "GOALKEEPER",
        2: "DEFENDER",
        3: "MIDFIELDER",
        4: "FORWARD",
    }

    scorer = FPLScorer()
    # Score first, sort, then apply percentile-based tiers across the population
    # (the per-player .tier from score_player uses absolute TIERS which nobody hits).
    scored: list[tuple] = []
    for raw in bootstrap.get("elements", []):
        player = FPLDataMapper.player_from_api(raw)
        fxs = team_fixtures.get(raw.get("team"), [])
        score, breakdown = scorer.score_player(player, fxs, sentiment=0.5)
        scored.append((player, score, breakdown, raw))

    scored.sort(key=lambda row: row[1], reverse=True)
    apply_percentile_tiers([(p, s, b) for p, s, b, _ in scored])

    ranked: list[dict] = []
    n = len(scored)
    for i, (player, score, breakdown, raw) in enumerate(scored):
        percentile = 100 * (n - 1 - i) / (n - 1) if n > 1 else 100.0
        team = teams.get(raw.get("team"), {})
        ranked.append(
            {
                "rank": i + 1,
                "percentile": round(percentile, 2),
                "id": player.id,
                "web_name": player.web_name,
                "full_name": f"{raw.get('first_name', '')} {raw.get('second_name', '')}".strip(),
                "team": team.get("short_name") or team.get("name"),
                "team_id": raw.get("team"),
                "position": position_name.get(player.element_type, "UNKNOWN"),
                "position_id": player.element_type,
                "price": round(player.now_cost / 10, 1),
                "total_points": player.total_points,
                "form": player.form,
                "minutes": player.minutes,
                "score": score,
                "tier": breakdown.tier,
                "factors": {
                    "fixture": breakdown.fixture,
                    "form": breakdown.form,
                    "value": breakdown.value,
                    "home_away": breakdown.home_away,
                    "xg_xa": breakdown.xg_xa,
                    "availability": breakdown.availability,
                    "momentum": breakdown.momentum,
                    "clean_sheet": breakdown.clean_sheet,
                    "bonus_rate": breakdown.bonus_rate,
                },
            }
        )

    payload = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "n_fixtures_evaluated": N_FIXTURES,
        "total_players": len(ranked),
        "positions": {str(p.value): p.name for p in Position},
        "players": ranked,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Wrote {OUT} ({len(ranked)} players)")

    print("\nTop 20 overall:")
    for row in ranked[:20]:
        print(
            f"  {row['rank']:>3}. {row['web_name']:<18} "
            f"{row['position']:<11} {row['team']:<4} "
            f"£{row['price']:>4.1f}m  score={row['score']:.4f}  {row['tier']}"
        )


if __name__ == "__main__":
    main()
