import logging
from functools import lru_cache
from json import dumps, loads
from urllib.request import urlopen

from langchain_core.tools import tool

from tools.fixtures import get_fixtures_by_gameweek
from tools.fpl_static import load_bootstrap, load_player_rankings

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

    output = {
        "manager_id": manager_id,
        "gameweek": gw,
        "active_chip": team_payload.get("active_chip"),
        "money": money_snapshot,
        "entry_history": entry_history,
        "free_transfers": free_transfers,
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
    limit: int = 20,
) -> str:
    """Return players ranked by the composite `scoring_module` score.

    This is the recommended tool for recommendation questions (transfers, captaincy,
    best-in-position picks). Scores blend fixture difficulty, form, value, xG/xA,
    availability, clean sheets, bonus rate and transfer momentum with
    position-specific weights. Tiers are percentile-based within the full player pool.

    Args:
        position:    Optional filter — GK/DEF/MID/FWD or full name.
        tier:        Optional filter — "MUST START", "STRONG PICK", "VIABLE OPTION",
                     "RISKY PICK", "AVOID" (substring match, case-insensitive).
        min_price:   Optional lower bound on price in £m (e.g. 4.5).
        max_price:   Optional upper bound on price in £m (e.g. 6.0).
        min_minutes: Optional minimum minutes played (filters rotation risks).
        limit:       Max rows to return (1-100, default 20).
    """
    logger.info(
        "Tool call start | get_fpl_scored_rankings | pos=%s | tier=%s | "
        "price=%s-%s | min_minutes=%s | limit=%s",
        position, tier, min_price, max_price, min_minutes, limit,
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

    players = data.get("players") or []
    filtered: list[dict] = []
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
        filtered.append({
            "rank": p.get("rank"),
            "percentile": p.get("percentile"),
            "id": p.get("id"),
            "name": p.get("web_name"),
            "full_name": p.get("full_name"),
            "team": p.get("team"),
            "position": p.get("position"),
            "price": p.get("price"),
            "total_points": p.get("total_points"),
            "form": p.get("form"),
            "minutes": p.get("minutes"),
            "score": p.get("score"),
            "tier": p.get("tier"),
            "factors": p.get("factors"),
        })
        if len(filtered) >= limit:
            break

    logger.info(
        "Tool call done | get_fpl_scored_rankings | returned=%s | total_pool=%s",
        len(filtered), len(players),
    )
    return dumps(
        {
            "fetched_at": data.get("fetched_at"),
            "n_fixtures_evaluated": data.get("n_fixtures_evaluated"),
            "total_players_in_pool": data.get("total_players"),
            "filters": {
                "position": pos_filter,
                "tier": tier_filter,
                "min_price": min_price,
                "max_price": max_price,
                "min_minutes": min_minutes,
                "limit": limit,
            },
            "count": len(filtered),
            "players": filtered,
        },
        indent=2,
    )

