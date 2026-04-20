import os
import re
from datetime import datetime, timezone
from json import dumps, loads
from urllib.error import URLError
from urllib.request import urlopen
from uuid import uuid4

from dotenv import load_dotenv
from openai import OpenAI
from uagents import Agent, Context, Protocol
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    TextContent,
    chat_protocol_spec,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv(".env.local")

client = OpenAI(
    api_key=os.getenv("ASI1_API_KEY"), 
    base_url='https://api.asi1.ai/v1'
)

agent = Agent(
    name="FPL Bot",
    port=8000,
    mailbox=True,
    publish_agent_details=True,
    readme_path = "README.md"
)

chat_protocol = Protocol(spec=chat_protocol_spec)

llm = ChatOpenAI(
    model="asi1-mini",
    api_key=os.getenv("ASI1_API_KEY"),
    base_url="https://api.asi1.ai/v1",
    temperature=0.2,
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "{input}"),
    ]
)

chain = prompt | llm

def ask_llm(user_text: str) -> str:
    result = chain.invoke({"input": user_text})
    return result.content if result and result.content else "No response generated."


def extract_manager_id(user_text: str) -> int | None:
    match = re.search(r"\bmanager\s*id\b\D*(\d+)", user_text, re.IGNORECASE)
    if match:
        return int(match.group(1))

    # If the user only sends a number, treat it as manager id.
    user_text = user_text.strip()
    if user_text.isdigit():
        return int(user_text)

    return None


def fetch_fpl_manager_data(manager_id: int) -> dict:
    url = f"https://fantasy.premierleague.com/api/entry/{manager_id}/"
    with urlopen(url, timeout=10) as response:
        payload = response.read().decode("utf-8")

    return loads(payload)


def format_manager_reply(manager_data: dict) -> str:
    player_first_name = manager_data.get("player_first_name", "")
    player_last_name = manager_data.get("player_last_name", "")
    manager_name = f"{player_first_name} {player_last_name}".strip() or "Unknown"
    team_name = manager_data.get("name", "Unknown")
    overall_rank = manager_data.get("summary_overall_rank", "N/A")
    total_points = manager_data.get("summary_overall_points", "N/A")

    return (
        f"Manager: {manager_name}\n"
        f"Team: {team_name}\n"
        f"Overall rank: {overall_rank}\n"
        f"Total points: {total_points}\n\n"
        f"Raw data:\n{dumps(manager_data, indent=2)}"
    )


def process_manager_data(user_text: str, manager_data: dict) -> str:
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

    reasoning_prompt = (
        "You are an FPL reasoning assistant.\n"
        "Use the provided manager data to answer the user.\n"
        "Do not invent data that is not in the payload.\n"
        "Provide concise reasoning and mention any limitations.\n\n"
        f"User request: {user_text}\n\n"
        "Manager snapshot:\n"
        f"{dumps(manager_snapshot, indent=2)}\n\n"
        "Full manager payload:\n"
        f"{dumps(manager_data, indent=2)}"
    )

    return ask_llm(reasoning_prompt)

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
        reply = "Send me a text message and I will ask ASI1."
    else:
        try:
            manager_id = extract_manager_id(user_text)
            if manager_id is not None:
                manager_data = fetch_fpl_manager_data(manager_id)
                reply = process_manager_data(user_text, manager_data)
            else:
                reply = ask_llm(user_text)
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
