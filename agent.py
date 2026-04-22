import logging
import os
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
    get_fpl_fixtures,
    get_fpl_manager_current_team,
    get_fpl_manager_data,
    get_fpl_player,
    get_fpl_scored_rankings,
    get_fpl_top_players,
    get_fpl_upcoming_gameweek,
    search_fpl_players,
)

load_dotenv(".env.local")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("fpl-agent")

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

reasoning_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an FPL reasoning assistant. Improve clarity and reasoning while "
            "staying faithful to tool-derived facts. Do not invent missing data. "
            "Tool evidence provided below is already fetched and available to you. "
            "Do NOT say you lack access to data, cannot verify, or cannot access tools. "
            "If evidence is missing a requested field, state that specific limitation.",
        ),
        (
            "human",
            "User request:\n{user_text}\n\n"
            "Tool evidence (already available):\n{tool_evidence}\n\n"
            "Draft answer:\n{draft_answer}\n\n"
            "Return only the final improved answer.",
        ),
    ]
)
reasoning_chain = reasoning_prompt | llm | StrOutputParser()

deep_agent = create_deep_agent(
    model=llm,
    tools=[
        get_fpl_manager_data,
        get_fpl_manager_current_team,
        get_fpl_upcoming_gameweek,
        get_fpl_fixtures,
        get_fpl_player,
        get_fpl_top_players,
        get_fpl_scored_rankings,
        search_fpl_players,
    ],
    system_prompt=(
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
        "Chips (one activation per chip per half-season window in 2025/26+): "
        "Wildcard, Free Hit, Bench Boost, Triple Captain, Assistant Manager. "
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


def run_deep_agent(user_text: str, thread_id: str) -> str:
    logger.info("Deep agent invoke start | thread_id=%s", thread_id)
    result = deep_agent.invoke(
        {"messages": [{"role": "user", "content": user_text}]},
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
    else:
        try:
            reply = run_deep_agent(user_text, thread_id=sender)
        except URLError as exc:
            logger.exception("FPL API network error | sender=%s", sender)
            reply = f"Could not reach FPL endpoint: {exc}"
        except Exception as exc:
            logger.exception("Unhandled agent error | sender=%s", sender)
            reply = f"LLM error: {exc}"

    await ctx.send(
        sender,
        ChatMessage(
            timestamp=datetime.now(timezone.utc),
            msg_id=uuid4(),
            content=[TextContent(type="text", text=reply)],
        ),
    )
    logger.info("Reply sent | sender=%s | reply_chars=%s", sender, len(reply))


@chat_protocol.on_message(model=ChatAcknowledgement)
async def on_chat_ack(_ctx: Context, _sender: str, _msg: ChatAcknowledgement):
    return


agent.include(chat_protocol)


if __name__ == "__main__":
    print(f"Your agent's address is: {agent.address}")
    agent.run()
