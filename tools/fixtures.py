"""Gameweek fixtures with team names resolved from local FPL static JSON."""

from __future__ import annotations

from json import loads
from urllib.request import urlopen

from tools.fpl_static import fixtures_for_event, team_id_to_name


def _fetch_fixtures_remote(gameweek: int) -> list:
    url = f"https://fantasy.premierleague.com/api/fixtures/?event={gameweek}"
    with urlopen(url, timeout=10) as response:
        return loads(response.read().decode("utf-8"))


def get_fixtures_by_gameweek(gameweek: int):
    """
    Return fixtures for a gameweek with team_h_name / team_a_name filled in.

    Team names come from data/fpl_static.json (bootstrap) — no extra request
    to bootstrap-static at runtime. Fixture rows use the snapshot's fixtures
    list when present; otherwise one request to /api/fixtures/?event=…
    """
    names = team_id_to_name()
    fixtures = fixtures_for_event(gameweek)
    if not fixtures:
        fixtures = _fetch_fixtures_remote(gameweek)
    for fixture in fixtures:
        th = fixture.get("team_h")
        ta = fixture.get("team_a")
        fixture["team_h_name"] = names.get(th, str(th))
        fixture["team_a_name"] = names.get(ta, str(ta))
    return fixtures


if __name__ == "__main__":
    sample = get_fixtures_by_gameweek(1)
    print(len(sample), "fixtures")
    print(sample)