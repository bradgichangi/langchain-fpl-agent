import os
from datetime import datetime, timezone
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


def ask_llm(prompt: str) -> str:
    response = client.chat.completions.create(
        model='asi1-mini',
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=300,
    )
    return response.choices[0].message.content or "No response generated."


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
            reply = ask_llm(user_text)
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
