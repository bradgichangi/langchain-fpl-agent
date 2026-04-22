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
        get_fpl_fixtures,
        get_fpl_player,
        get_fpl_top_players,
        get_fpl_scored_rankings,
        search_fpl_players,
    ],
    system_prompt=(
        "You are an FPL assistant that must route requests and use tools when data is needed. "
        "Routing policy: "
        "1) If manager-specific analysis is requested and manager id is missing, ask for manager ID first. "
        "2) If manager id exists, call get_fpl_manager_data and/or get_fpl_manager_current_team as needed. "
        "2b) For questions like 'how are players in my team performing', use get_fpl_manager_current_team as the primary data source. "
        "2a) Do NOT call get_fpl_player immediately after get_fpl_manager_current_team for the same picks; "
        "that tool already returns player names, team, position_type, now_cost, status, form, points_per_game, and total_points. "
        "Only call get_fpl_player if the user explicitly requests additional fields not in that output. "
        "3) For fixture questions, call get_fpl_fixtures. "
        "4) For player lookup questions, call get_fpl_player or search_fpl_players. "
        "5) For top/best player questions by a single raw FPL metric (points, form, ownership), call get_fpl_top_players. "
        "6) For RECOMMENDATION-style questions — who to captain, who to transfer in, best pick under £6m, "
        "strongest GK/DEF/MID/FWD, differential picks — call get_fpl_scored_rankings. It returns "
        "composite-scored players with tiers (MUST START > STRONG PICK > VIABLE OPTION > RISKY PICK > AVOID), "
        "position, price, and factor breakdowns (fixture, form, value, xg_xa, availability, clean_sheet, etc.). "
        "Use its filters (position, tier, min_price/max_price, min_minutes) rather than hand-filtering. "
        "When answering recommendation questions, cite the tier and 1-2 dominant factors driving the score. "
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
