from uagents import Agent, Context, Model
import os

class Message(Model):
    message: str
 

agent = Agent(
    name="FPL Bot",
    port=8000,
    mailbox=True,
    seed=os.getenv("UAGENT_SEED"),
)
 
# Copy the address shown below
print(f"Your agent's address is: {agent.address}")
 
if __name__ == "__main__":
    agent.run()