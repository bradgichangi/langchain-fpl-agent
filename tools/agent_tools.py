import logging
from functools import lru_cache
from json import dumps, loads
from urllib.request import urlopen

from langchain_core.tools import tool

from tools.fixtures import get_fixtures_by_gameweek
from tools.fpl_static import load_bootstrap

logger = logging.getLogger("fpl-agent")


def fetch_fpl_manager_data(manager_id: int) -> dict:
    logger.info("Fetching manager data from FPL API | manager_id=%s", manager_id)
    url = f"https://fantasy.premierleague.com/api/entry/{manager_id}/"
    with urlopen(url, timeout=10) as response:
        payload = response.read().decode("utf-8")
    return loads(payload)


@lru_cache(maxsize=1)
def fetch_fpl_bootstrap_live() -> dict:
    """Fallback live bootstrap-static data for ID resolution mismatches."""
    logger.info("Fetching live bootstrap-static fallback from FPL API")
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    with urlopen(url, timeout=20) as response:
        payload = response.read().decode("utf-8")
    return loads(payload)


def get_current_gameweek() -> int | None:
    """Read current GW from local static bootstrap snapshot."""
    bootstrap = load_bootstrap()
    events = bootstrap.get("events") or []
    for ev in events:
        if ev.get("is_current") and isinstance(ev.get("id"), int):
            return ev["id"]
    # Fallback to live bootstrap when local snapshot is stale or offseason-marked.
    live_bootstrap = fetch_fpl_bootstrap_live()
    live_events = live_bootstrap.get("events") or []
    for ev in live_events:
        if ev.get("is_current") and isinstance(ev.get("id"), int):
            return ev["id"]
    return None


def fetch_fpl_manager_team_data(manager_id: int, gameweek: int) -> dict:
    logger.info(
        "Fetching manager picks from FPL API | manager_id=%s | gameweek=%s",
        manager_id,
        gameweek,
    )
    url = f"https://fantasy.premierleague.com/api/entry/{manager_id}/event/{gameweek}/picks/"
    with urlopen(url, timeout=10) as response:
        payload = response.read().decode("utf-8")
    return loads(payload)


def fetch_fpl_event_live_data(gameweek: int) -> dict:
    logger.info("Fetching FPL event live data | gameweek=%s", gameweek)
    url = f"https://fantasy.premierleague.com/api/event/{gameweek}/live/"
    with urlopen(url, timeout=20) as response:
        payload = response.read().decode("utf-8")
    return loads(payload)


@tool
def get_fpl_manager_data(manager_id: int) -> str:
    """Get FPL manager profile data by manager ID."""
    logger.info("Tool call start | get_fpl_manager_data | manager_id=%s", manager_id)
    manager_data = fetch_fpl_manager_data(manager_id)
    manager_snapshot = {
        "manager_name": (
            f"{manager_data.get('player_first_name', '')} "
            f"{manager_data.get('player_last_name', '')}"
        ).strip(),
        "team_name": manager_data.get("name"),
        "overall_rank": manager_data.get("summary_overall_rank"),
        "overall_points": manager_data.get("summary_overall_points"),
        "current_gw_points": manager_data.get("summary_event_points"),
        "bank": manager_data.get("last_deadline_bank"),
        "value": manager_data.get("last_deadline_value"),
    }
    return dumps(
        {"manager_snapshot": manager_snapshot, "manager_payload": manager_data}, indent=2
    )


@tool
def get_fpl_manager_current_team(manager_id: int, gameweek: int | None = None) -> str:
    """Get a manager's current team picks (or specific gameweek if provided).

    This already includes player identity + core metadata per pick, so prefer this
    over per-player follow-up calls unless the user explicitly asks for extra fields.
    """
    gw = gameweek or get_current_gameweek()
    if gw is None:
        manager_data = fetch_fpl_manager_data(manager_id)
        fallback_gw = manager_data.get("current_event")
        if isinstance(fallback_gw, int):
            gw = fallback_gw
            logger.info(
                "Resolved current GW from manager profile | manager_id=%s | gameweek=%s",
                manager_id,
                gw,
            )
        else:
            return dumps(
                {
                    "error": (
                        "Could not determine current gameweek from static data "
                        "or manager profile. Pass gameweek explicitly."
                    )
                }
            )

    logger.info(
        "Tool call start | get_fpl_manager_current_team | manager_id=%s | gameweek=%s",
        manager_id,
        gw,
    )
    team_payload = fetch_fpl_manager_team_data(manager_id, gw)
    # Always prefer live bootstrap for current team to avoid season mismatch.
    bootstrap = fetch_fpl_bootstrap_live()
    elements = bootstrap.get("elements") or []
    teams = bootstrap.get("teams") or []
    element_types = bootstrap.get("element_types") or []
    elements_by_id = {p.get("id"): p for p in elements if p.get("id") is not None}
    teams_by_id = {t.get("id"): t for t in teams if t.get("id") is not None}
    element_types_by_id = {
        et.get("id"): et for et in element_types if et.get("id") is not None
    }
    event_live = fetch_fpl_event_live_data(gw)
    live_elements = event_live.get("elements") or []
    live_stats_by_id = {
        el.get("id"): (el.get("stats") or {})
        for el in live_elements
        if el.get("id") is not None
    }

    picks = team_payload.get("picks") or []
    enriched_picks = []
    for pick in picks:
        element_id = pick.get("element")
        player = elements_by_id.get(element_id, {})
        team = teams_by_id.get(player.get("team"), {})
        element_type = element_types_by_id.get(player.get("element_type"), {})
        live_stats = live_stats_by_id.get(element_id, {})
        enriched_picks.append(
            {
                **pick,
                "player_name": player.get("web_name"),
                "first_name": player.get("first_name"),
                "second_name": player.get("second_name"),
                "position_type": player.get("element_type"),
                "position": element_type.get("singular_name_short"),
                "team_name": team.get("name"),
                "team_short_name": team.get("short_name"),
                "now_cost": player.get("now_cost"),
                "status": player.get("status"),
                "injury_news": player.get("news"),
                "chance_of_playing_next_round": player.get("chance_of_playing_next_round"),
                "form": player.get("form"),
                "points_per_game": player.get("points_per_game"),
                "total_points": player.get("total_points"),
                "gw_points": live_stats.get("total_points"),
                "gw_minutes": live_stats.get("minutes"),
                "gw_goals": live_stats.get("goals_scored"),
                "gw_assists": live_stats.get("assists"),
                "gw_clean_sheets": live_stats.get("clean_sheets"),
                "gw_bonus": live_stats.get("bonus"),
                "gw_bps": live_stats.get("bps"),
                "gw_yellow_cards": live_stats.get("yellow_cards"),
                "gw_red_cards": live_stats.get("red_cards"),
            }
        )

    output = {
        "manager_id": manager_id,
        "gameweek": gw,
        "active_chip": team_payload.get("active_chip"),
        "entry_history": team_payload.get("entry_history"),
        "picks": enriched_picks,
        "automatic_subs": team_payload.get("automatic_subs"),
    }
    logger.info(
        "Tool call done | get_fpl_manager_current_team | manager_id=%s | picks=%s",
        manager_id,
        len(enriched_picks),
    )
    return dumps(output, indent=2)


@tool
def get_fpl_fixtures(gameweek: int) -> str:
    """Get FPL fixtures for a gameweek, including team names."""
    logger.info("Tool call start | get_fpl_fixtures | gameweek=%s", gameweek)
    fixtures = get_fixtures_by_gameweek(gameweek)
    logger.info("Tool call done | get_fpl_fixtures | fixtures=%s", len(fixtures))
    return dumps(fixtures, indent=2)


@tool
def get_fpl_player(player_id: int) -> str:
    """Get detailed FPL player data by player id from local static data.

    Use this only for additional player detail not present in
    get_fpl_manager_current_team output.
    """
    logger.info("Tool call start | get_fpl_player | player_id=%s", player_id)
    bootstrap = load_bootstrap()
    elements = bootstrap.get("elements") or []
    for p in elements:
        if p.get("id") == player_id:
            return dumps(p, indent=2)
    logger.warning(
        "Local static miss in get_fpl_player | player_id=%s | trying live bootstrap",
        player_id,
    )
    live_bootstrap = fetch_fpl_bootstrap_live()
    live_elements = live_bootstrap.get("elements") or []
    for p in live_elements:
        if p.get("id") == player_id:
            return dumps(p, indent=2)
    return dumps({"error": f"Player id {player_id} not found in local or live bootstrap."})


@tool
def search_fpl_players(name_query: str) -> str:
    """Search local FPL players by name fragment."""
    logger.info("Tool call start | search_fpl_players | query=%s", name_query)
    q = name_query.strip().lower()
    if not q:
        return dumps({"error": "name_query is required."})
    bootstrap = load_bootstrap()
    elements = bootstrap.get("elements") or []
    matches = []
    for p in elements:
        haystack = " ".join(
            [
                str(p.get("web_name", "")),
                str(p.get("first_name", "")),
                str(p.get("second_name", "")),
            ]
        ).lower()
        if q in haystack:
            matches.append(p)
    logger.info("Tool call done | search_fpl_players | matches=%s", len(matches))
    return dumps(matches[:25], indent=2)

