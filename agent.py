import os
import re
from datetime import datetime, timezone
from json import JSONDecodeError, dumps, loads
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

# Step 1: router — FPL only: state the goal and what manager data to fetch (JSON only).
ROUTER_INSTRUCTIONS = """You are a routing layer for a Fantasy Premier League (FPL) assistant.
Every message is treated as an FPL question. Output a single JSON object (no markdown, no prose):
{
  "goal": "<short restatement of what the user wants>",
  "required_data": ["<facts needed, e.g. manager_profile, overall_rank>"],
  "tool_plan": [
    {"tool": "get_fpl_manager_data", "manager_id": <integer or null>}
  ],
  "missing_inputs": ["<only if manager_id unknown: what to ask the user>"]
}
Rules:
- If the message contains an FPL manager id (digits alone or e.g. "manager id 12345"), set tool_plan[0].manager_id to that integer.
- If no manager id can be inferred, set manager_id to null and put clear asks in missing_inputs (e.g. "Send your FPL manager ID (the number in your team URL)").
- required_data should list which parts of the manager API payload matter for answering the goal.
"""


def parse_router_json(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        return loads(text)
    except JSONDecodeError:
        return {
            "goal": "",
            "required_data": [],
            "tool_plan": [],
            "missing_inputs": ["Send your FPL manager ID so I can load your team."],
        }


def run_router(user_text: str) -> dict:
    completion = client.chat.completions.create(
        model="asi1-mini",
        messages=[
            {"role": "system", "content": ROUTER_INSTRUCTIONS},
            {"role": "user", "content": user_text},
        ],
        temperature=0,
    )
    content = completion.choices[0].message.content or "{}"
    return parse_router_json(content)


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
        ("system", "You are an FPL (Fantasy Premier League) assistant."),
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
        reply = "I'm here to help you with your FPL team. Please ask me a question about your team."
    else:
        try:
            plan = run_router(user_text)
            manager_id = None
            for step in plan.get("tool_plan") or []:
                if step.get("tool") == "get_fpl_manager_data":
                    mid = step.get("manager_id")
                    if isinstance(mid, int):
                        manager_id = mid
                        break
            if manager_id is None:
                manager_id = extract_manager_id(user_text)

            if manager_id is None:
                missing = plan.get("missing_inputs") or []
                reply = (
                    "I need your FPL manager ID to look up your team (the number in your team URL on fantasy.premierleague.com). "
                    + (" ".join(missing) if missing else "")
                ).strip()
            else:
                manager_data = fetch_fpl_manager_data(manager_id)
                router_context = dumps(plan, indent=2)
                reply = process_manager_data(
                    f"{user_text}\n\n[Router plan]\n{router_context}",
                    manager_data,
                )
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
