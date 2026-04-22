#!/usr/bin/env python3
"""Download FPL data into a compact data/fpl_static.json (no rules bloat, slim teams/elements)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "fpl_static.json"

BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"

TEAM_KEYS = ("id", "strength", "name", "short_name")
EVENT_KEYS = (
    "id",
    "name",
    "deadline_time",
    "finished",
    "is_previous",
    "is_current",
    "is_next",
)
ELEMENT_KEYS = (
    "id",
    "web_name",
    "first_name",
    "second_name",
    "team",
    "element_type",
    "now_cost",
    "status",
    "news",
    "form",
    "total_points",
    "points_per_game",
)
FIXTURE_KEYS = (
    "id",
    "code",
    "event",
    "finished",
    "finished_provisional",
    "kickoff_time",
    "provisional_start_time",
    "started",
    "minutes",
    "team_h",
    "team_a",
    "team_h_difficulty",
    "team_a_difficulty",
    "team_h_score",
    "team_a_score",
    "pulse_id",
)


def fetch_json(url: str) -> dict | list:
    with urlopen(url, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def pick_keys(obj: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {k: obj[k] for k in keys if k in obj}


def slim_bootstrap(raw: dict[str, Any]) -> dict[str, Any]:
    """Drop rules/config noise; keep only catalog fields we use."""
    out: dict[str, Any] = {}
    if "total_players" in raw:
        out["total_players"] = raw["total_players"]
    if raw.get("teams"):
        out["teams"] = [pick_keys(t, TEAM_KEYS) for t in raw["teams"]]
    if raw.get("events"):
        out["gameweeks"] = [pick_keys(ev, EVENT_KEYS) for ev in raw["events"]]
    if raw.get("elements"):
        out["players"] = [pick_keys(el, ELEMENT_KEYS) for el in raw["elements"]]
    if raw.get("element_types"):
        out["positions"] = raw["element_types"]
    return out


def slim_fixtures(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fixture rows without `stats` (large) or other unused fields."""
    return [pick_keys(fx, FIXTURE_KEYS) for fx in raw]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    bootstrap_raw = fetch_json(BOOTSTRAP_URL)
    if not isinstance(bootstrap_raw, dict):
        raise SystemExit("Unexpected bootstrap-static response")
    fixtures_raw = fetch_json(FIXTURES_URL)
    if not isinstance(fixtures_raw, list):
        raise SystemExit("Unexpected fixtures response")

    payload = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "bootstrap": slim_bootstrap(bootstrap_raw),
        "fixtures": slim_fixtures(fixtures_raw),
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
