import logging
from functools import lru_cache
from json import dumps, loads
from urllib.error import HTTPError
from urllib.request import urlopen

from langchain_core.tools import tool

from modules.chip_opportunity_module import ChipTimingEngine, serialize_chip_scores
from tools.fixtures import get_fixtures_by_gameweek
from tools.fpl_static import load_bootstrap, load_player_rankings

logger = logging.getLogger("fpl-agent")


def _normalize_chip_name(raw: str | None) -> str | None:
    if not raw:
        return None
    key = str(raw).strip().lower()
    aliases = {
        "wildcard": "wildcard",
        "freehit": "free_hit",
        "free_hit": "free_hit",
        "bboost": "bench_boost",
        "bench_boost": "bench_boost",
        "bench boost": "bench_boost",
        "3xc": "triple_captain",
        "triple_captain": "triple_captain",
        "triple captain": "triple_captain",
    }
    return aliases.get(key)


def _available_chips_from_history(chips_used: list[dict]) -> list[str]:
    # Current supported chips in this agent policy.
    all_supported = ["wildcard", "free_hit", "bench_boost", "triple_captain"]
    used = set()
    for chip in chips_used:
        norm = _normalize_chip_name(chip.get("name"))
        if norm:
            used.add(norm)
    return [chip for chip in all_supported if chip not in used]


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


def _find_event(events: list[dict], flag: str) -> dict | None:
    for ev in events:
        if ev.get(flag) and isinstance(ev.get("id"), int):
            return ev
    return None


def get_current_gameweek() -> int | None:
    """Read current GW (`is_current`) from local static bootstrap, live fallback."""
    bootstrap = load_bootstrap()
    ev = _find_event(bootstrap.get("events") or [], "is_current")
    if ev is None:
        ev = _find_event(fetch_fpl_bootstrap_live().get("events") or [], "is_current")
    return ev["id"] if ev else None


def get_next_gameweek_event() -> dict | None:
    """Return the upcoming GW event dict (the one with `is_next: True`).

    Prefers the live bootstrap-static so the deadline_time and flags reflect
    the FPL state at query time rather than the snapshot's fetched_at.
    """
    live_events = fetch_fpl_bootstrap_live().get("events") or []
    ev = _find_event(live_events, "is_next")
    if ev is None:
        ev = _find_event(load_bootstrap().get("events") or [], "is_next")
    return ev


def get_upcoming_fixture_map() -> tuple[int | None, dict[int, list[dict]]]:
    """Return (upcoming_gw_id, {team_id: [fixture_dict, ...]}).

    Each fixture_dict carries opponent metadata + home/away. Empty list = blank
    GW for that team. Used to flag players who don't play in the upcoming GW.
    """
    next_ev = get_next_gameweek_event()
    if not next_ev:
        return None, {}
    gw_id = next_ev["id"]
    fixtures = get_fixtures_by_gameweek(gw_id)
    bootstrap = fetch_fpl_bootstrap_live()
    teams = bootstrap.get("teams") or []
    short_by_id = {t.get("id"): t.get("short_name") for t in teams if t.get("id")}
    name_by_id = {t.get("id"): t.get("name") for t in teams if t.get("id")}
    by_team: dict[int, list[dict]] = {tid: [] for tid in short_by_id}
    for fx in fixtures:
        th, ta, kickoff = fx.get("team_h"), fx.get("team_a"), fx.get("kickoff_time")
        if isinstance(th, int):
            by_team.setdefault(th, []).append({
                "opponent_id": ta,
                "opponent_short": short_by_id.get(ta),
                "opponent_name": name_by_id.get(ta),
                "is_home": True,
                "kickoff_time": kickoff,
                "difficulty": fx.get("team_h_difficulty"),
            })
        if isinstance(ta, int):
            by_team.setdefault(ta, []).append({
                "opponent_id": th,
                "opponent_short": short_by_id.get(th),
                "opponent_name": name_by_id.get(th),
                "is_home": False,
                "kickoff_time": kickoff,
                "difficulty": fx.get("team_a_difficulty"),
            })
    return gw_id, by_team


def _summarize_upcoming(team_id: int | None, by_team: dict[int, list[dict]]) -> dict:
    fxs = by_team.get(team_id, []) if isinstance(team_id, int) else []
    return {
        "plays_upcoming": bool(fxs),
        "fixture_count": len(fxs),
        "opponents": [
            f"{fx.get('opponent_short') or '?'} ({'H' if fx.get('is_home') else 'A'})"
            for fx in fxs
        ],
        "fixtures": fxs,
    }


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


def fetch_fpl_manager_history(manager_id: int) -> dict:
    logger.info("Fetching manager history | manager_id=%s", manager_id)
    url = f"https://fantasy.premierleague.com/api/entry/{manager_id}/history/"
    with urlopen(url, timeout=15) as response:
        payload = response.read().decode("utf-8")
    return loads(payload)


# FPL rule (2024/25+): managers can bank up to 5 free transfers. Earlier
# seasons capped at 2; this constant affects the max banked state only.
MAX_FREE_TRANSFERS = 5


def compute_free_transfers(history: dict) -> dict:
    """Derive banked free transfers from a manager's /history/ payload.

    FPL does not expose free-transfer count on public endpoints, so we replay
    GW-by-GW transfer activity with the banking rules:
      * +1 FT earned each completed GW (cap at `MAX_FREE_TRANSFERS`).
      * `event_transfers` consume banked FTs first; remainder becomes hits.
      * Wildcard and Free Hit: transfers are free; the FT bank is preserved
        and the GW's +1 is NOT earned (the chip replaces the FT economy for
        that GW). Pre-2024/25 Wildcard used to reset bank to 1; that rule
        changed and the new behaviour is verified against live /history/ data.
      * Bench Boost / Triple Captain / Assistant Manager: no effect.
    Returns the state *after* the last completed GW, i.e. the FTs the manager
    carries into the upcoming GW.
    """
    current = history.get("current") or []
    chips = history.get("chips") or []
    chip_by_event: dict[int, str] = {
        c["event"]: (c.get("name") or "").lower()
        for c in chips
        if isinstance(c.get("event"), int)
    }

    bank = 1  # every manager starts the season with 1 FT pre-GW1
    transfers_total = 0
    hits_total = 0
    last_gw: int | None = None
    trajectory: list[dict] = []

    for row in current:
        event = row.get("event")
        if not isinstance(event, int):
            continue
        last_gw = event
        event_transfers = int(row.get("event_transfers") or 0)
        event_hit_cost = int(row.get("event_transfers_cost") or 0)
        transfers_total += event_transfers
        hits_total += event_hit_cost

        bank_before = bank
        chip = chip_by_event.get(event)
        if chip in ("wildcard", "freehit"):
            # Both chips freeze the FT economy under 2024/25+ rules:
            # chip-GW transfers are free, bank is preserved, no +1 earned.
            # (Pre-2024/25 Wildcard reset bank to 1; that no longer applies —
            # verified against real /history/ data.)
            pass
        else:
            used_free = min(event_transfers, bank)
            bank = min(bank - used_free + 1, MAX_FREE_TRANSFERS)

        trajectory.append({
            "event": event,
            "transfers_made": event_transfers,
            "hit_cost": event_hit_cost,
            "chip": chip,
            "bank_before": bank_before,
            "bank_after": bank,
        })

    last_row = current[-1] if current else {}
    return {
        "free_transfers_available": bank,
        "max_banked": MAX_FREE_TRANSFERS,
        "last_completed_gw": last_gw,
        "last_gw_transfers": int(last_row.get("event_transfers") or 0) if last_row else 0,
        "last_gw_hit_cost": int(last_row.get("event_transfers_cost") or 0) if last_row else 0,
        "season_totals": {
            "transfers": transfers_total,
            "hit_cost": hits_total,
            "gameweeks_played": len(current),
        },
        "chips_used": [
            {"name": c.get("name"), "event": c.get("event"), "time": c.get("time")}
            for c in chips
        ],
        # Full per-GW FT trajectory — useful for diagnosing off-by-one disputes
        # against the FPL UI. `bank_after` = FTs heading into the next GW.
        "ft_trajectory": trajectory,
        "notes": [
            "Derived from /history/ using FPL 2024/25+ banking rules (cap = 5).",
            "Wildcard and Free Hit GWs freeze the FT bank — no transfers "
            "consumed, no +1 earned.",
            "`free_transfers_available` = FTs heading into the GW AFTER "
            "`last_completed_gw`. Does not reflect transfers made between that "
            "deadline and the next /history/ refresh.",
        ],
    }


def _tenths_to_millions(raw: int | None) -> float | None:
    """FPL stores prices / bank / team value in tenths of £m (e.g. 1053 → £105.3m)."""
    if raw is None:
        return None
    try:
        return round(int(raw) / 10, 1)
    except (TypeError, ValueError):
        return None


@tool
def get_fpl_manager_data(manager_id: int) -> str:
    """Get FPL manager profile data by manager ID.

    Money fields are converted from FPL's raw tenths-of-millions ints to £m
    floats, and the team's total value is split into `squad_value_m` (excluding
    bank) vs `team_value_m` (including bank, matches FPL's "Team Value" UI).
    """
    logger.info("Tool call start | get_fpl_manager_data | manager_id=%s", manager_id)
    manager_data = fetch_fpl_manager_data(manager_id)
    bank_raw = manager_data.get("last_deadline_bank")
    value_raw = manager_data.get("last_deadline_value")
    bank_m = _tenths_to_millions(bank_raw)
    team_value_m = _tenths_to_millions(value_raw)
    squad_value_m = None
    if isinstance(bank_raw, int) and isinstance(value_raw, int):
        squad_value_m = round((value_raw - bank_raw) / 10, 1)

    manager_snapshot = {
        "manager_name": (
            f"{manager_data.get('player_first_name', '')} "
            f"{manager_data.get('player_last_name', '')}"
        ).strip(),
        "team_name": manager_data.get("name"),
        "overall_rank": manager_data.get("summary_overall_rank"),
        "overall_points": manager_data.get("summary_overall_points"),
        "current_gw_points": manager_data.get("summary_event_points"),
        "bank_m": bank_m,
        "squad_value_m": squad_value_m,
        "team_value_m": team_value_m,
        "total_transfers_season": manager_data.get("last_deadline_total_transfers"),
        "_units": (
            "All money values are in £m (e.g. 4.9 = £4.9m). "
            "squad_value_m excludes bank; team_value_m includes bank."
        ),
    }
    return dumps(
        {"manager_snapshot": manager_snapshot, "manager_payload": manager_data}, indent=2
    )


@tool
def get_fpl_manager_current_team(manager_id: int, gameweek: int | None = None) -> str:
    """Get a manager's current team picks (or specific gameweek if provided).

    This already includes player identity + core metadata per pick, so prefer this
    over per-player follow-up calls unless the user explicitly asks for extra fields.
    The output also includes a derived `free_transfers` block (banked FTs for the
    upcoming GW plus season transfer/hit totals), so use this tool for questions
    about free transfers, hits, or how many transfers a manager has made.
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

    upcoming_gw_id, upcoming_by_team = get_upcoming_fixture_map()

    picks = team_payload.get("picks") or []
    enriched_picks = []
    blanks: list[dict] = []
    for pick in picks:
        element_id = pick.get("element")
        player = elements_by_id.get(element_id, {})
        team = teams_by_id.get(player.get("team"), {})
        element_type = element_types_by_id.get(player.get("element_type"), {})
        live_stats = live_stats_by_id.get(element_id, {})
        upcoming = _summarize_upcoming(player.get("team"), upcoming_by_team)
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
                "upcoming_plays": upcoming["plays_upcoming"],
                "upcoming_fixture_count": upcoming["fixture_count"],
                "upcoming_opponents": upcoming["opponents"],
                "upcoming_fixtures": upcoming["fixtures"],
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
        if not upcoming["plays_upcoming"]:
            blanks.append({
                "element_id": element_id,
                "player_name": player.get("web_name"),
                "team_short_name": team.get("short_name"),
                "position": element_type.get("singular_name_short"),
            })

    free_transfers: dict | None
    try:
        history_payload = fetch_fpl_manager_history(manager_id)
        free_transfers = compute_free_transfers(history_payload)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Could not compute free transfers | manager_id=%s | err=%s",
            manager_id, exc,
        )
        free_transfers = {"error": f"Could not derive free transfers: {exc}"}

    entry_history = team_payload.get("entry_history") or {}
    eh_bank = entry_history.get("bank")
    eh_value = entry_history.get("value")
    squad_value_m = None
    if isinstance(eh_bank, int) and isinstance(eh_value, int):
        squad_value_m = round((eh_value - eh_bank) / 10, 1)
    money_snapshot = {
        "bank_m": _tenths_to_millions(eh_bank),
        "squad_value_m": squad_value_m,
        "team_value_m": _tenths_to_millions(eh_value),
        "_units": (
            "All money values are in £m (e.g. 4.9 = £4.9m). "
            "squad_value_m excludes bank; team_value_m includes bank."
        ),
    }

    upcoming_squad_summary = {
        "upcoming_gameweek": upcoming_gw_id,
        "blank_count": len(blanks),
        "blanking_players": blanks,
        "_note": (
            "Players in `blanking_players` have no fixture in the upcoming GW. "
            "Flag them when discussing the squad and consider transfers/bench order."
        ),
    }

    output = {
        "manager_id": manager_id,
        "gameweek": gw,
        "active_chip": team_payload.get("active_chip"),
        "money": money_snapshot,
        "entry_history": entry_history,
        "free_transfers": free_transfers,
        "upcoming_squad": upcoming_squad_summary,
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
def get_fpl_upcoming_gameweek() -> str:
    """Return the upcoming gameweek context: id, name, deadline, fixtures.

    Call this BEFORE answering any time-sensitive question (transfers, captain,
    'this week', 'next match', squad changes) so your recommendation is
    anchored to the correct GW and the user knows the deadline. Output also
    includes the current GW and fixtures already enriched with team names.
    """
    logger.info("Tool call start | get_fpl_upcoming_gameweek")
    next_ev = get_next_gameweek_event()
    if not next_ev:
        return dumps({"error": "Could not determine the upcoming gameweek."})
    gw_id = next_ev["id"]
    fixtures = get_fixtures_by_gameweek(gw_id)
    current_gw = get_current_gameweek()
    payload = {
        "upcoming_gameweek": gw_id,
        "name": next_ev.get("name"),
        "deadline_time": next_ev.get("deadline_time"),
        "is_current": bool(next_ev.get("is_current")),
        "is_next": bool(next_ev.get("is_next")),
        "current_gameweek": current_gw,
        "fixture_count": len(fixtures),
        "fixtures": fixtures,
    }
    logger.info(
        "Tool call done | get_fpl_upcoming_gameweek | upcoming=%s | fixtures=%s",
        gw_id, len(fixtures),
    )
    return dumps(payload, indent=2)


@tool
def get_fpl_fixtures(gameweek: int | None = None) -> str:
    """Get FPL fixtures for a gameweek, including team names.

    If `gameweek` is omitted, defaults to the upcoming GW (`is_next`), so the
    agent can call this with no args for "this week's fixtures" questions.
    """
    if gameweek is None:
        next_ev = get_next_gameweek_event()
        if not next_ev:
            return dumps({"error": "Could not determine the upcoming gameweek."})
        gameweek = next_ev["id"]
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


@tool
def get_fpl_top_players(
    metric: str = "total_points", limit: int = 10, position: str | None = None
) -> str:
    """Get top FPL players by a metric.

    Args:
        metric: one of total_points, points_per_game, form, now_cost, selected_by_percent
        limit: number of players to return (1-50)
        position: optional filter GK, DEF, MID, FWD
    """
    allowed_metrics = {
        "total_points",
        "points_per_game",
        "form",
        "now_cost",
        "selected_by_percent",
    }
    metric = metric.strip()
    if metric not in allowed_metrics:
        return dumps(
            {
                "error": f"Unsupported metric '{metric}'.",
                "allowed_metrics": sorted(allowed_metrics),
            }
        )

    limit = max(1, min(50, int(limit)))
    bootstrap = fetch_fpl_bootstrap_live()
    elements = bootstrap.get("elements") or []
    teams = bootstrap.get("teams") or []
    element_types = bootstrap.get("element_types") or []
    teams_by_id = {t.get("id"): t for t in teams if t.get("id") is not None}
    pos_by_id = {
        et.get("id"): et.get("singular_name_short")
        for et in element_types
        if et.get("id") is not None
    }

    pos_filter = position.upper().strip() if position else None
    if pos_filter and pos_filter not in {"GK", "DEF", "MID", "FWD"}:
        return dumps({"error": "position must be one of GK, DEF, MID, FWD"})

    def metric_value(player: dict) -> float:
        value = player.get(metric)
        if value is None:
            return float("-inf")
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(str(value))
        except ValueError:
            return float("-inf")

    filtered = []
    for p in elements:
        pos = pos_by_id.get(p.get("element_type"))
        if pos_filter and pos != pos_filter:
            continue
        filtered.append(p)

    top_players = sorted(filtered, key=metric_value, reverse=True)[:limit]
    out = []
    for p in top_players:
        team = teams_by_id.get(p.get("team"), {})
        out.append(
            {
                "id": p.get("id"),
                "name": p.get("web_name"),
                "position": pos_by_id.get(p.get("element_type")),
                "team_name": team.get("name"),
                "team_short_name": team.get("short_name"),
                "metric": metric,
                "metric_value": p.get(metric),
                "total_points": p.get("total_points"),
                "points_per_game": p.get("points_per_game"),
                "form": p.get("form"),
                "now_cost": p.get("now_cost"),
                "status": p.get("status"),
                "injury_news": p.get("news"),
            }
        )

    logger.info(
        "Tool call done | get_fpl_top_players | metric=%s | position=%s | count=%s",
        metric,
        pos_filter,
        len(out),
    )
    return dumps(
        {
            "metric": metric,
            "position_filter": pos_filter,
            "count": len(out),
            "players": out,
        },
        indent=2,
    )


_POSITION_ALIASES = {
    "GK": "GOALKEEPER", "GKP": "GOALKEEPER", "GOALKEEPER": "GOALKEEPER",
    "DEF": "DEFENDER", "DEFENDER": "DEFENDER",
    "MID": "MIDFIELDER", "MIDFIELDER": "MIDFIELDER",
    "FWD": "FORWARD", "FORWARD": "FORWARD",
}


@tool
def get_fpl_scored_rankings(
    position: str | None = None,
    tier: str | None = None,
    min_price: float | None = None,
    max_price: float | None = None,
    min_minutes: int | None = None,
    must_play_upcoming: bool = True,
    limit: int = 20,
) -> str:
    """Return players ranked by the composite `scoring_module` score.

    This is the recommended tool for recommendation questions (transfers, captaincy,
    best-in-position picks). Scores blend fixture difficulty, form, value, xG/xA,
    availability, clean sheets, bonus rate and transfer momentum with
    position-specific weights. Tiers are percentile-based within the full player pool.

    Each returned player is annotated with `upcoming_plays`, `upcoming_opponents`,
    and `upcoming_fixtures` for the next GW so blanks are obvious. By default
    blanking players are filtered out. A `set_pieces` block is also included
    per row (penalties / direct FK / corners taker order, plus a human-readable
    `role` like "primary pen, secondary corners"), and the composite score
    factors a `set_piece` term in.

    Args:
        position:           Optional filter — GK/DEF/MID/FWD or full name.
        tier:               Optional filter — "MUST START", "STRONG PICK",
                            "VIABLE OPTION", "RISKY PICK", "AVOID"
                            (substring match, case-insensitive).
        min_price:          Optional lower bound on price in £m (e.g. 4.5).
        max_price:          Optional upper bound on price in £m (e.g. 6.0).
        min_minutes:        Optional minimum minutes played (filters rotation risks).
        must_play_upcoming: If True (default), exclude players whose club has no
                            fixture in the upcoming GW. Set False only when the
                            user explicitly wants longer-horizon analysis.
        limit:              Max rows to return (1-100, default 20).
    """
    logger.info(
        "Tool call start | get_fpl_scored_rankings | pos=%s | tier=%s | "
        "price=%s-%s | min_minutes=%s | must_play_upcoming=%s | limit=%s",
        position, tier, min_price, max_price, min_minutes,
        must_play_upcoming, limit,
    )

    try:
        data = load_player_rankings()
    except FileNotFoundError as exc:
        return dumps({"error": str(exc)})

    pos_filter: str | None = None
    if position:
        key = position.strip().upper()
        pos_filter = _POSITION_ALIASES.get(key)
        if pos_filter is None:
            return dumps({
                "error": f"Unsupported position '{position}'.",
                "allowed_positions": sorted(set(_POSITION_ALIASES)),
            })

    tier_filter = tier.strip().lower() if tier else None
    limit = max(1, min(100, int(limit)))

    upcoming_gw_id, upcoming_by_team = get_upcoming_fixture_map()

    players = data.get("players") or []
    filtered: list[dict] = []
    excluded_blanks = 0
    for p in players:
        if pos_filter and p.get("position") != pos_filter:
            continue
        if tier_filter and tier_filter not in (p.get("tier") or "").lower():
            continue
        price = p.get("price")
        if min_price is not None and (price is None or price < min_price):
            continue
        if max_price is not None and (price is None or price > max_price):
            continue
        if min_minutes is not None and (p.get("minutes") or 0) < min_minutes:
            continue
        upcoming = _summarize_upcoming(p.get("team_id"), upcoming_by_team)
        if must_play_upcoming and not upcoming["plays_upcoming"]:
            excluded_blanks += 1
            continue
        filtered.append({
            "rank": p.get("rank"),
            "percentile": p.get("percentile"),
            "id": p.get("id"),
            "name": p.get("web_name"),
            "full_name": p.get("full_name"),
            "team": p.get("team"),
            "team_id": p.get("team_id"),
            "position": p.get("position"),
            "price": p.get("price"),
            "total_points": p.get("total_points"),
            "form": p.get("form"),
            "minutes": p.get("minutes"),
            "score": p.get("score"),
            "tier": p.get("tier"),
            "factors": p.get("factors"),
            "set_pieces": p.get("set_pieces"),
            "upcoming_plays": upcoming["plays_upcoming"],
            "upcoming_fixture_count": upcoming["fixture_count"],
            "upcoming_opponents": upcoming["opponents"],
            "upcoming_fixtures": upcoming["fixtures"],
        })
        if len(filtered) >= limit:
            break

    logger.info(
        "Tool call done | get_fpl_scored_rankings | returned=%s | total_pool=%s "
        "| excluded_blanks=%s | upcoming_gw=%s",
        len(filtered), len(players), excluded_blanks, upcoming_gw_id,
    )
    return dumps(
        {
            "fetched_at": data.get("fetched_at"),
            "n_fixtures_evaluated": data.get("n_fixtures_evaluated"),
            "total_players_in_pool": data.get("total_players"),
            "upcoming_gameweek": upcoming_gw_id,
            "filters": {
                "position": pos_filter,
                "tier": tier_filter,
                "min_price": min_price,
                "max_price": max_price,
                "min_minutes": min_minutes,
                "must_play_upcoming": must_play_upcoming,
                "limit": limit,
            },
            "excluded_blanks": excluded_blanks,
            "count": len(filtered),
            "players": filtered,
        },
        indent=2,
    )


@tool
def get_fpl_chip_opportunities(
    manager_id: int,
    gameweek: int | None = None,
    horizon_gws: int = 6,
) -> str:
    """Evaluate best chip timing for a manager (WC/FH/BB/TC).

    Uses the `chip_opportunity_module` engine with:
    - live manager picks for the target GW
    - manager chip history (to infer available chips)
    - composite player rankings + fixture context
    - fixture map for current + near-term horizon
    """
    logger.info(
        "Tool call start | get_fpl_chip_opportunities | manager_id=%s | gameweek=%s | horizon=%s",
        manager_id,
        gameweek,
        horizon_gws,
    )
    manager_data = fetch_fpl_manager_data(manager_id)

    gw = gameweek
    if gw is None:
        next_ev = get_next_gameweek_event()
        gw = next_ev.get("id") if next_ev else None
    if gw is None:
        gw = get_current_gameweek()
    if gw is None:
        fallback_gw = manager_data.get("current_event")
        if isinstance(fallback_gw, int):
            gw = fallback_gw
    if gw is None:
        return dumps({"error": "Could not determine target gameweek."})

    # Future GW picks may 404 before the deadline is locked; fallback to current.
    team_candidates: list[int] = []
    for candidate in (
        gw,
        manager_data.get("current_event"),
        gw - 1 if isinstance(gw, int) else None,
    ):
        if isinstance(candidate, int) and candidate not in team_candidates and candidate >= 1:
            team_candidates.append(candidate)

    try:
        team_payload = None
        team_payload_gw = None
        for candidate in team_candidates:
            try:
                team_payload = fetch_fpl_manager_team_data(manager_id, candidate)
                team_payload_gw = candidate
                break
            except HTTPError as exc:
                if exc.code == 404:
                    continue
                raise
        if team_payload is None:
            return dumps(
                {
                    "error": (
                        "Could not fetch manager picks for target or fallback gameweeks."
                    ),
                    "target_gameweek": gw,
                    "attempted_gameweeks": team_candidates,
                }
            )
        history_payload = fetch_fpl_manager_history(manager_id)
        rankings = load_player_rankings().get("players") or []
    except Exception as exc:  # noqa: BLE001
        logger.exception("Chip opportunity data fetch failed | manager_id=%s", manager_id)
        return dumps({"error": f"Could not fetch required inputs: {exc}"})

    # Build fixture map for current + near-term horizon.
    horizon = max(1, min(int(horizon_gws), 12))
    gw_fixture_map: dict[int, list[dict]] = {}
    for event in range(gw, gw + horizon):
        gw_fixture_map[event] = get_fixtures_by_gameweek(event)

    fixture_count_by_team: dict[int, int] = {}
    for fx in gw_fixture_map.get(gw, []):
        th = fx.get("team_h")
        ta = fx.get("team_a")
        if isinstance(th, int):
            fixture_count_by_team[th] = fixture_count_by_team.get(th, 0) + 1
        if isinstance(ta, int):
            fixture_count_by_team[ta] = fixture_count_by_team.get(ta, 0) + 1

    chips_used = history_payload.get("chips") or []
    available_chips = _available_chips_from_history(chips_used)

    ranked_by_id = {p.get("id"): p for p in rankings if p.get("id") is not None}
    ranked_players: list[dict] = []
    for row in rankings:
        team_id = row.get("team_id")
        fixtures_this_gw = fixture_count_by_team.get(team_id, 0) if isinstance(team_id, int) else 0
        ranked_players.append(
            {
                **row,
                "web_name": row.get("web_name") or row.get("name"),
                "breakdown": row.get("factors") or {},
                "fixtures_this_gw": fixtures_this_gw,
            }
        )

    picks = team_payload.get("picks") or []
    squad_ids = [p.get("element") for p in picks if isinstance(p.get("element"), int)]
    bench_picks = [p for p in picks if isinstance(p.get("position"), int) and p["position"] > 11]
    bench_player_data: list[dict] = []
    for p in bench_picks:
        pid = p.get("element")
        ranked = ranked_by_id.get(pid) or {}
        fixture_count = 1
        if isinstance(ranked.get("team_id"), int):
            fixture_count = fixture_count_by_team.get(ranked["team_id"], 0)
        bench_player_data.append(
            {
                "id": pid,
                "score": ranked.get("score", 0.4),
                "has_fixture": fixture_count >= 1,
                "fixture_count": fixture_count,
            }
        )

    remaining_gws = max(0, 38 - gw)
    squad_context = {
        "chips_available": available_chips,
        "squad_ids": squad_ids,
        "bench_ids": [p.get("element") for p in bench_picks if p.get("element") is not None],
        "bench_player_data": bench_player_data,
        "remaining_gws": remaining_gws,
    }

    engine = ChipTimingEngine()
    scores = engine.evaluate_all_chips(
        squad_context=squad_context,
        current_gw=gw,
        gw_fixture_map=gw_fixture_map,
        ranked_players=ranked_players,
        remaining_gws=remaining_gws,
    )

    payload = {
        "manager_id": manager_id,
        "target_gameweek": gw,
        "manager_team_source_gameweek": team_payload_gw,
        "remaining_gws": remaining_gws,
        "chips_available": available_chips,
        "chips_used": chips_used,
        "best_chip_now": scores[0].chip if scores else None,
        "chip_scores": serialize_chip_scores(scores),
    }
    logger.info(
        "Tool call done | get_fpl_chip_opportunities | manager_id=%s | chips_evaluated=%s",
        manager_id,
        len(scores),
    )
    return dumps(payload, indent=2)

