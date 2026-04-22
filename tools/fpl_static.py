"""Load FPL catalog data from a local JSON snapshot (no runtime HTTP to bootstrap-static)."""

from __future__ import annotations

from functools import lru_cache
from json import loads
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_PATH = _REPO_ROOT / "data" / "fpl_static.json"
_RANKINGS_PATH = _REPO_ROOT / "data" / "fpl_player_rankings.json"


@lru_cache(maxsize=1)
def _raw_payload(path: str | None = None) -> dict:
    p = Path(path) if path else _DEFAULT_PATH
    if not p.is_file():
        raise FileNotFoundError(
            f"Missing FPL static data file: {p}. "
            "Run: python scripts/fetch_fpl_static.py"
        )
    with open(p, encoding="utf-8") as f:
        return loads(f.read())


def load_fpl_static(path: str | None = None) -> dict:
    """Return the full snapshot dict (may include fetched_at, bootstrap, fixtures)."""
    return _raw_payload(path)


def load_bootstrap(path: str | None = None) -> dict:
    """Return bootstrap-static payload (teams, elements, events, …)."""
    data = _raw_payload(path)
    if "bootstrap" in data:
        return data["bootstrap"]
    return data


def team_id_to_name(path: str | None = None) -> dict[int, str]:
    """Map FPL team id -> display name (prefers short_name)."""
    bootstrap = load_bootstrap(path)
    teams = bootstrap.get("teams") or []
    out: dict[int, str] = {}
    for t in teams:
        tid = t.get("id")
        if tid is None:
            continue
        name = t.get("name", "").strip()
        if name:
            out[int(tid)] = name
    return out


def fixtures_for_event(event_id: int, path: str | None = None) -> list[dict]:
    """Return fixtures for a gameweek from the snapshot if present; else []."""
    data = _raw_payload(path)
    all_fixtures = data.get("fixtures")
    if not isinstance(all_fixtures, list):
        return []
    return [fx for fx in all_fixtures if fx.get("event") == event_id]


@lru_cache(maxsize=1)
def _raw_rankings(path: str | None = None) -> dict:
    p = Path(path) if path else _RANKINGS_PATH
    if not p.is_file():
        raise FileNotFoundError(
            f"Missing FPL rankings file: {p}. "
            "Run: python scripts/rank_all_players.py"
        )
    with open(p, encoding="utf-8") as f:
        return loads(f.read())


def load_player_rankings(path: str | None = None) -> dict:
    """Return the composite ranking snapshot produced by `scripts/rank_all_players.py`.

    Structure:
        {
          "fetched_at": str, "n_fixtures_evaluated": int,
          "total_players": int, "positions": {...},
          "players": [ {rank, percentile, id, web_name, team, position,
                        price, total_points, form, minutes, score, tier,
                        factors: {...}}, ... ]
        }
    """
    return _raw_rankings(path)
