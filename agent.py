import logging
import os
import re
from json import JSONDecodeError, dump, load
from pathlib import Path
from datetime import datetime, timezone
from urllib.error import URLError
from uuid import uuid4

from deepagents import create_deep_agent
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from uagents import Agent, Context, Protocol
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    TextContent,
    chat_protocol_spec,
)

from tools.agent_tools import (
    get_fpl_chip_opportunities,
    get_fpl_fixtures,
    get_fpl_manager_current_team,
    get_fpl_manager_data,
    get_fpl_player,
    get_fpl_scored_rankings,
    get_fpl_top_players,
    get_fpl_upcoming_gameweek,
    search_fpl_players,
    search_web,
)

load_dotenv(".env.local")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("fpl-agent")

_SESSION_STATE_PATH = (
    Path(__file__).resolve().parent / "data" / "session_manager_ids.json"
)
_MANAGER_ID_RE = re.compile(
    r"(?:manager\s*id|my\s*id|entry\s*id)\D{0,12}(\d{4,12})",
    flags=re.IGNORECASE,
)
_BARE_ID_RE = re.compile(r"\s*(\d{4,12})\s*")
_CLEAR_ID_RE = re.compile(
    r"\b(forget|clear|remove|reset)\b.{0,24}\b(manager\s*id|my\s*id|id)\b",
    flags=re.IGNORECASE,
)
_CLEAR_HISTORY_RE = re.compile(
    r"\b(forget|clear|reset|wipe)\b.{0,24}"
    r"\b(history|chat|messages?|conversation|context|memory)\b",
    flags=re.IGNORECASE,
)

# Session chat history (in-memory only; cleared on process restart).
# `_MAX_HISTORY_TURNS` counts user/assistant pairs (16 messages total at cap).
_MAX_HISTORY_TURNS = 8
_MAX_HISTORY_CHARS = 8000
# Char budget for the compact history snippet that feeds the reasoning pass.
_REASONING_HISTORY_CHARS = 2000
CHAT_HISTORY_BY_SENDER: dict[str, list[dict[str, str]]] = {}


def _send_text_reply(sender: str, text: str) -> ChatMessage:
    return ChatMessage(
        timestamp=datetime.now(timezone.utc),
        msg_id=uuid4(),
        content=[TextContent(type="text", text=text)],
    )


def _load_manager_id_store() -> dict[str, int]:
    if not _SESSION_STATE_PATH.is_file():
        return {}
    try:
        with open(_SESSION_STATE_PATH, encoding="utf-8") as f:
            raw = load(f)
    except (OSError, JSONDecodeError):
        logger.warning(
            "Could not read session manager-id store | path=%s",
            _SESSION_STATE_PATH,
        )
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, int] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            continue
        if isinstance(value, int):
            out[key] = value
            continue
        try:
            out[key] = int(str(value))
        except (TypeError, ValueError):
            continue
    return out


def _persist_manager_id_store(store: dict[str, int]) -> None:
    try:
        _SESSION_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_SESSION_STATE_PATH, "w", encoding="utf-8") as f:
            dump(store, f, indent=2)
    except OSError as exc:
        logger.warning(
            "Could not persist session manager-id store | path=%s | err=%s",
            _SESSION_STATE_PATH,
            exc,
        )


MANAGER_ID_BY_SENDER: dict[str, int] = _load_manager_id_store()


def _extract_manager_id_from_text(text: str) -> int | None:
    # Prefer explicit mention to avoid confusing GW numbers or prices with IDs.
    explicit = _MANAGER_ID_RE.search(text)
    if explicit:
        return int(explicit.group(1))
    # Fallback: if user sends only a number, treat it as manager id.
    bare = _BARE_ID_RE.fullmatch(text)
    if bare:
        return int(bare.group(1))
    return None


def _wants_to_clear_manager_id(text: str) -> bool:
    return bool(_CLEAR_ID_RE.search(text))


def _inject_manager_context(user_text: str, sender: str) -> str:
    manager_id = MANAGER_ID_BY_SENDER.get(sender)
    if manager_id is None:
        return user_text
    return (
        f"{user_text}\n\n"
        f"[Session context: default manager_id for this sender is {manager_id}. "
        "Use it for manager-specific analysis unless the user provides a new manager id.]"
    )


def _handle_session_manager_id(sender: str, user_text: str) -> tuple[str | None, str]:
    """Return (immediate_reply, augmented_user_text)."""
    if _wants_to_clear_manager_id(user_text):
        removed = MANAGER_ID_BY_SENDER.pop(sender, None)
        _persist_manager_id_store(MANAGER_ID_BY_SENDER)
        if removed is None:
            return "No saved manager ID found for this session.", user_text
        return "Saved manager ID cleared for this session.", user_text

    extracted_manager_id = _extract_manager_id_from_text(user_text)
    if extracted_manager_id is not None:
        previous = MANAGER_ID_BY_SENDER.get(sender)
        MANAGER_ID_BY_SENDER[sender] = extracted_manager_id
        _persist_manager_id_store(MANAGER_ID_BY_SENDER)
        logger.info(
            "Stored manager id for sender | sender=%s | manager_id=%s | previous=%s",
            sender,
            extracted_manager_id,
            previous,
        )

    return None, _inject_manager_context(user_text, sender)


def _append_history(sender: str, role: str, content: str) -> None:
    """Append one message and enforce per-sender size/char caps."""
    if not content:
        return
    history = CHAT_HISTORY_BY_SENDER.setdefault(sender, [])
    history.append({"role": role, "content": content})
    max_messages = _MAX_HISTORY_TURNS * 2
    if len(history) > max_messages:
        del history[: len(history) - max_messages]
    total = sum(len(m["content"]) for m in history)
    while total > _MAX_HISTORY_CHARS and len(history) > 1:
        dropped = history.pop(0)
        total -= len(dropped["content"])


def _clear_chat_history(sender: str) -> bool:
    """Remove all stored messages for a sender; True if anything was dropped."""
    return CHAT_HISTORY_BY_SENDER.pop(sender, None) is not None


def _wants_to_clear_chat_history(text: str) -> bool:
    return bool(_CLEAR_HISTORY_RE.search(text))


def _prior_messages(sender: str) -> list[dict[str, str]]:
    """Return a shallow copy of the sender's history for deep-agent replay."""
    return list(CHAT_HISTORY_BY_SENDER.get(sender) or [])


def _conversation_context(sender: str) -> str:
    """Recent history formatted as a prose block for the reasoning pass.

    Oldest messages are dropped first if the full history exceeds
    `_REASONING_HISTORY_CHARS`.
    """
    history = CHAT_HISTORY_BY_SENDER.get(sender) or []
    if not history:
        return "(none)"
    rendered = "\n".join(
        f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
        for msg in history
    )
    if len(rendered) <= _REASONING_HISTORY_CHARS:
        return rendered
    # Head-truncate so the most recent turns stay in frame.
    excess = len(rendered) - _REASONING_HISTORY_CHARS
    return "…(older history truncated)…\n" + rendered[excess:]


agent = Agent(
    name="FPL Bot",
    port=8000,
    mailbox=True,
    publish_agent_details=True,
    readme_path="README.md",
)

chat_protocol = Protocol(spec=chat_protocol_spec)

llm = ChatOpenAI(
    model="asi1-mini",
    api_key=os.getenv("ASI1_API_KEY"),
    base_url="https://api.asi1.ai/v1",
    temperature=0.2,
)

_REASONING_TODAY_ISO = datetime.now(timezone.utc).strftime("%Y-%m-%d")
_REASONING_TODAY_HUMAN = datetime.now(timezone.utc).strftime("%A %B %d, %Y")

reasoning_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"CURRENT DATE: {_REASONING_TODAY_HUMAN} (ISO {_REASONING_TODAY_ISO}). "
            "Trust this over any internal training-cutoff assumption. "
            "You are an FPL reasoning assistant. Improve clarity and reasoning while "
            "staying faithful to tool-derived facts. Do not invent missing data. "
            "Tool evidence provided below is already fetched and available to you. "
            "Do NOT say you lack access to data, cannot verify, or cannot access tools. "
            "If evidence is missing a requested field, state that specific limitation. "
            "If tool evidence contains web search results with URLs, preserve those URLs "
            "as inline citations (e.g. `(source: <url>)`) in the final answer. "
            "If conversation context is provided below, use it to resolve follow-up "
            "references (pronouns, 'my team', 'that player', 'the same GW') — do NOT "
            "ask the user to repeat info they already shared earlier in the session.",
        ),
        (
            "human",
            "Conversation so far (oldest → newest):\n{conversation_context}\n\n"
            "Current user request:\n{user_text}\n\n"
            "Tool evidence (already available):\n{tool_evidence}\n\n"
            "Draft answer:\n{draft_answer}\n\n"
            "Return only the final improved answer.",
        ),
    ]
)
reasoning_chain = reasoning_prompt | llm | StrOutputParser()

_TODAY = datetime.now(timezone.utc)
_TODAY_ISO = _TODAY.strftime("%Y-%m-%d")
_TODAY_HUMAN = _TODAY.strftime("%A %B %d, %Y")
_CURRENT_YEAR = _TODAY.year

deep_agent = create_deep_agent(
    model=llm,
    tools=[
        get_fpl_manager_data,
        get_fpl_manager_current_team,
        get_fpl_chip_opportunities,
        get_fpl_upcoming_gameweek,
        get_fpl_fixtures,
        get_fpl_player,
        get_fpl_top_players,
        get_fpl_scored_rankings,
        search_fpl_players,
        search_web,
    ],
    system_prompt=(
        f"CURRENT DATE: {_TODAY_HUMAN} (ISO {_TODAY_ISO}). The current year is "
        f"{_CURRENT_YEAR}. Trust this date over any internal training-cutoff assumption. "
        f"When calling search_web for recency-sensitive questions, include "
        f"'{_CURRENT_YEAR}' (or the current month/year) in the query string and set "
        f"time_range='d' or 'w' so results are actually current.\n\n"
        "You are an FPL (Fantasy Premier League) assistant. You already know the FPL "
        "rules below; never ask the user to remind you of them, and always honor them "
        "when building squads, suggesting transfers, or evaluating decisions.\n\n"
        "FPL RULES (assume 2025/26+ season):\n"
        "Squad composition: exactly 15 players — 2 GK, 5 DEF, 5 MID, 3 FWD. "
        "Max 3 players from any one Premier League club. "
        "Starting budget £100.0m; squad cost must fit budget + bank. "
        "Starting XI is 11 of the 15: exactly 1 GK, min 3 DEF, min 2 MID, min 1 FWD; "
        "valid formations include 3-4-3, 3-5-2, 4-3-3, 4-4-2, 4-5-1, 5-3-2, 5-4-1. "
        "Captain scores double points; vice-captain auto-promotes if captain doesn't play. "
        "Bench order matters for auto-subs (positions 12-15, with GK reserve fixed).\n"
        "Transfers: 1 free transfer earned per gameweek, max 5 banked. "
        "Each extra transfer costs -4 points. Wildcard and Free Hit chips make all "
        "transfers in that GW free; both freeze the FT bank (no consume, no +1 earned).\n"
        "Chips: Wildcard, Free Hit, Bench Boost, Triple Captain. "
        "Only one chip can be active per gameweek.\n"
        "Routing policy: "
        "0) For ANY time-sensitive question (transfers, captain choice, 'this week', 'next match', "
        "squad/starting XI changes, deadline, who plays this GW), call get_fpl_upcoming_gameweek "
        "FIRST so your recommendation is anchored to the correct GW and you can quote the deadline. "
        "Do not assume the gameweek; always confirm it via this tool when timing matters. "
        "0a) Players must have a fixture in the upcoming GW to be a valid pick/captain/transfer-in. "
        "get_fpl_scored_rankings already filters out blanks by default (must_play_upcoming=True) and "
        "tags every row with `upcoming_plays`, `upcoming_opponents`. get_fpl_manager_current_team "
        "returns an `upcoming_squad` block listing any blanking players in the current squad — "
        "ALWAYS surface those to the user (warn against captaining/starting them; suggest bench / transfer). "
        "Only set must_play_upcoming=False if the user explicitly asks about long-term value. "
        "1) If manager-specific analysis is requested and manager id is missing, ask for manager ID first. "
        "2) If manager id exists, call get_fpl_manager_data and/or get_fpl_manager_current_team as needed. "
        "2b) For questions like 'how are players in my team performing', use get_fpl_manager_current_team as the primary data source. "
        "2a) Do NOT call get_fpl_player immediately after get_fpl_manager_current_team for the same picks; "
        "that tool already returns player names, team, position_type, now_cost, status, form, points_per_game, and total_points. "
        "Only call get_fpl_player if the user explicitly requests additional fields not in that output. "
        "2c) For questions about free transfers, transfer hits, or how many transfers a manager has made, "
        "call get_fpl_manager_current_team and read its `free_transfers` block "
        "(fields: free_transfers_available, last_gw_transfers, last_gw_hit_cost, season_totals, chips_used). "
        "2d) For bank / squad value / team value questions, use the `money` block returned by "
        "get_fpl_manager_current_team (or `manager_snapshot` from get_fpl_manager_data). "
        "All money fields are already in £m floats — DO NOT divide or multiply them. "
        "`bank_m` is the bank in £m, `squad_value_m` excludes bank, `team_value_m` includes bank. "
        "Do NOT read or interpret raw `bank` / `value` / `last_deadline_*` integers; those are in "
        "tenths of millions and the *_m fields are the correct display values. "
        "2e) For chip timing / chip strategy questions (Wildcard, Free Hit, Bench Boost, Triple Captain), "
        "call get_fpl_chip_opportunities. Use its scored recommendation, triggers, blockers, and wait_reason "
        "instead of generic chip advice. "
        "3) For fixture questions, call get_fpl_fixtures. "
        "4) For player lookup questions, call get_fpl_player or search_fpl_players. "
        "5) For top/best player questions by a single raw FPL metric (points, form, ownership), call get_fpl_top_players. "
        "6) For RECOMMENDATION-style questions — who to captain, who to transfer in, best pick under £6m, "
        "strongest GK/DEF/MID/FWD, differential picks — call get_fpl_scored_rankings. It returns "
        "composite-scored players with tiers (MUST START > STRONG PICK > VIABLE OPTION > RISKY PICK > AVOID), "
        "position, price, and factor breakdowns (fixture, form, value, xg_xa, availability, clean_sheet, "
        "set_piece, etc.) plus a `set_pieces` block (penalty/FK/corner taker order + role label). "
        "Use its filters (position, tier, min_price/max_price, min_minutes) rather than hand-filtering. "
        "When answering recommendation questions, cite the tier and 1-2 dominant factors driving the score, "
        "and call out set-piece duties (especially primary penalty takers) since they materially raise ceiling.\n"
        "6a) For news-shaped questions — injuries, suspensions, press conference quotes, "
        "predicted lineups, rotation rumours, price-change alerts, manager statements, or "
        "anything the FPL API / static data cannot cover — call search_web. Prefer FPL-specific "
        "tools first; escalate to search_web only when the question is explicitly news/rumour-shaped "
        "or prior tool calls returned insufficient data. RECENCY: use the CURRENT DATE noted above "
        "(never assume an older year). For 'this week' / 'latest' / 'right now' queries, include "
        "the current year (and month if helpful) in the query string and pass time_range='d' or 'w'. "
        "For general news use time_range='m'. If the first call returns stale or empty results, retry "
        "with a tighter time_range and/or an explicit date in the query. ALWAYS cite the source URLs "
        "inline (e.g. `(source: <url>)`). Do not call search_web redundantly after get_fpl_player or "
        "get_fpl_scored_rankings for the same player unless the user explicitly asked about news.\n"
        "7) For full-squad-build requests ('build me a 15-man squad', 'pick my team'), "
        "first call get_fpl_upcoming_gameweek so the build is anchored to the next GW "
        "(quote the GW id and deadline in the answer). Then call get_fpl_scored_rankings "
        "once per position with the appropriate price ceilings (keep must_play_upcoming=True so "
        "no blanking players are picked), and assemble exactly 2 GK / 5 DEF / 5 MID / 3 FWD "
        "that fits within the user's budget (default £100.0m if unknown), respects the "
        "max-3-per-club rule, and where possible leaves a usable bank. Always show the squad "
        "split by position with prices and a sum, verify the sum is within budget, "
        "confirm no club has more than 3 selections, and confirm every selected player has "
        "a fixture in the upcoming GW.\n\n"
        "Never invent FPL data; use tool output as source of truth and mention limits."
    ),
)


def run_deep_agent(
    user_text: str,
    thread_id: str,
    prior_messages: list[dict[str, str]] | None = None,
    conversation_context: str = "(none)",
) -> str:
    history_turns = len(prior_messages or [])
    logger.info(
        "Deep agent invoke start | thread_id=%s | history_turns=%s",
        thread_id,
        history_turns,
    )
    deep_messages: list[dict[str, str]] = list(prior_messages or [])
    deep_messages.append({"role": "user", "content": user_text})
    result = deep_agent.invoke(
        {"messages": deep_messages},
        config={"configurable": {"thread_id": thread_id}},
    )
    draft_answer = "No response generated."
    tool_evidence_parts: list[str] = []
    messages = result.get("messages", [])
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            if isinstance(message.content, str):
                draft_answer = message.content
                break
            draft_answer = str(message.content)
            break
    for message in messages:
        if isinstance(message, ToolMessage):
            content = message.content
            text = content if isinstance(content, str) else str(content)
            tool_evidence_parts.append(text)

    logger.info("Deep agent invoke done | thread_id=%s", thread_id)
    logger.info("Reasoning pass start | thread_id=%s", thread_id)
    tool_evidence = "\n\n".join(tool_evidence_parts[-4:]) if tool_evidence_parts else "(none)"
    final_answer = reasoning_chain.invoke(
        {
            "user_text": user_text,
            "draft_answer": draft_answer,
            "tool_evidence": tool_evidence,
            "conversation_context": conversation_context,
        }
    )
    logger.info("Reasoning pass done | thread_id=%s", thread_id)
    return final_answer


@chat_protocol.on_message(model=ChatMessage)
async def on_chat_message(ctx: Context, sender: str, msg: ChatMessage):
    logger.info("Incoming message | sender=%s | msg_id=%s", sender, msg.msg_id)
    await ctx.send(
        sender,
        ChatAcknowledgement(
            timestamp=datetime.now(timezone.utc),
            acknowledged_msg_id=msg.msg_id,
        ),
    )

    user_text = " ".join(
        part.text.strip() for part in msg.content if isinstance(part, TextContent)
    ).strip()

    if not user_text:
        logger.info("Empty message received | sender=%s", sender)
        reply = (
            "I can help with FPL manager, fixtures, and player questions. "
            "Send a request and include manager ID for team-specific analysis."
        )
    elif _wants_to_clear_chat_history(user_text):
        cleared = _clear_chat_history(sender)
        reply = (
            "Chat history cleared for this session."
            if cleared
            else "No chat history to clear for this session."
        )
    else:
        immediate_reply, augmented_user_text = _handle_session_manager_id(sender, user_text)
        if immediate_reply is not None:
            reply = immediate_reply
        else:
            prior = _prior_messages(sender)
            context = _conversation_context(sender)
            try:
                reply = run_deep_agent(
                    augmented_user_text,
                    thread_id=sender,
                    prior_messages=prior,
                    conversation_context=context,
                )
            except URLError as exc:
                logger.exception("FPL API network error | sender=%s", sender)
                reply = f"Could not reach FPL endpoint: {exc}"
            except Exception as exc:
                logger.exception("Unhandled agent error | sender=%s", sender)
                reply = f"LLM error: {exc}"
            else:
                # Only persist the turn when the deep agent produced a reply.
                # Store the raw user text (not the manager-id-augmented form).
                _append_history(sender, "user", user_text)
                _append_history(sender, "assistant", reply)

    await ctx.send(sender, _send_text_reply(sender, reply))
    logger.info(
        "Reply sent | sender=%s | reply_chars=%s | history_msgs=%s",
        sender,
        len(reply),
        len(CHAT_HISTORY_BY_SENDER.get(sender) or []),
    )


@chat_protocol.on_message(model=ChatAcknowledgement)
async def on_chat_ack(_ctx: Context, _sender: str, _msg: ChatAcknowledgement):
    return


agent.include(chat_protocol)


if __name__ == "__main__":
    print(f"Your agent's address is: {agent.address}")
    agent.run()
