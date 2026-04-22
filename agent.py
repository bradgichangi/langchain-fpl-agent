import os
from datetime import datetime, timezone
from json import dumps, loads
from urllib.error import URLError
from urllib.request import urlopen
from uuid import uuid4

from deepagents import create_deep_agent
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from uagents import Agent, Context, Protocol
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    TextContent,
    chat_protocol_spec,
)

from tools.fixtures import get_fixtures_by_gameweek
from tools.fpl_static import load_bootstrap

load_dotenv(".env.local")

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


def fetch_fpl_manager_data(manager_id: int) -> dict:
    url = f"https://fantasy.premierleague.com/api/entry/{manager_id}/"
    with urlopen(url, timeout=10) as response:
        payload = response.read().decode("utf-8")
    return loads(payload)


@tool
def get_fpl_manager_data(manager_id: int) -> str:
    """Get FPL manager profile data by manager ID."""
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
def get_fpl_fixtures(gameweek: int) -> str:
    """Get FPL fixtures for a gameweek, including team names."""
    fixtures = get_fixtures_by_gameweek(gameweek)
    return dumps(fixtures, indent=2)


@tool
def get_fpl_player(player_id: int) -> str:
    """Get FPL player data by player id from local static data."""
    bootstrap = load_bootstrap()
    elements = bootstrap.get("elements") or []
    for p in elements:
        if p.get("id") == player_id:
            return dumps(p, indent=2)
    return dumps({"error": f"Player id {player_id} not found."})


@tool
def search_fpl_players(name_query: str) -> str:
    """Search local FPL players by name fragment."""
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
    return dumps(matches[:25], indent=2)


deep_agent = create_deep_agent(
    model=llm,
    tools=[get_fpl_manager_data, get_fpl_fixtures, get_fpl_player, search_fpl_players],
    system_prompt=(
        "You are an FPL assistant that must route requests and use tools when data is needed. "
        "Routing policy: "
        "1) If manager-specific analysis is requested and manager id is missing, ask for manager ID first. "
        "2) If manager id exists, call get_fpl_manager_data. "
        "3) For fixture questions, call get_fpl_fixtures. "
        "4) For player lookup questions, call get_fpl_player or search_fpl_players. "
        "Never invent FPL data; use tool output as source of truth and mention limits."
    ),
)


def run_deep_agent(user_text: str, thread_id: str) -> str:
    result = deep_agent.invoke(
        {"messages": [{"role": "user", "content": user_text}]},
        config={"configurable": {"thread_id": thread_id}},
    )
    messages = result.get("messages", [])
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            if isinstance(message.content, str):
                return message.content
            return str(message.content)
    return "No response generated."


@chat_protocol.on_message(model=ChatMessage)
async def on_chat_message(ctx: Context, sender: str, msg: ChatMessage):
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
        reply = (
            "I can help with FPL manager, fixtures, and player questions. "
            "Send a request and include manager ID for team-specific analysis."
        )
    else:
        try:
            reply = run_deep_agent(user_text, thread_id=sender)
        except URLError as exc:
            reply = f"Could not reach FPL endpoint: {exc}"
        except Exception as exc:
            reply = f"LLM error: {exc}"

    await ctx.send(
        sender,
        ChatMessage(
            timestamp=datetime.now(timezone.utc),
            msg_id=uuid4(),
            content=[TextContent(type="text", text=reply)],
        ),
    )


@chat_protocol.on_message(model=ChatAcknowledgement)
async def on_chat_ack(_ctx: Context, _sender: str, _msg: ChatAcknowledgement):
    return


agent.include(chat_protocol)


if __name__ == "__main__":
    print(f"Your agent's address is: {agent.address}")
    agent.run()
