import os
import asyncio
from nova_simple_agent import RemoteCommandTool, RemoteSkillTool, RemoteWriteTool, RemoteReadTool
from pi_agent import Agent, AgentEvent
from nova_ai import ThinkingLevel, get_model

os.environ["VOLCENGINE_API_KEY"] = "*"
os.environ["DEEPSEEK_API_KEY"] = "*"

async def main():
    # 1. Initialize Agent (only once)
    print("Initializing Agent...")
    agent = Agent(
        steering_mode="one-at-a-time",
        follow_up_mode="all",
        max_retry_delay_ms=30
    )

    # 2. Set Model
    print("Configuring model and tools...")
    model = get_model("volcengine", "deepseek-r1-250528")
    agent.set_model(model)
    agent.set_system_prompt(
        'You are a helpful assistant that can use tools to answer user questions. Please prioritize querying the skill library.'
        'First, initialize the remote host with IP: 127.0.0.1, port 8899, then search for available skills.'
    )
    agent.set_thinking_level(ThinkingLevel.MEDIUM)

    # 3. Add Tools
    tools = [RemoteCommandTool(), RemoteSkillTool(), RemoteWriteTool(), RemoteReadTool()]
    agent.set_tools(tools)

    def on_event(event: AgentEvent):
        """Handle all Agent events"""
        event_type = event.type

        if event_type == "message_start":
            msg = event.message
            print(f"\n[Message Start] {msg.role}: ...")

        elif event_type == "message_end":
            msg = event.message
            for content in msg.content:
                if content.type == "text" and content.text:
                    print(f"[Answer]: {content.text}")
                elif content.type == "thinking":
                    print(f"[Thinking]: {content.thinking}")
            error = msg.error_message
            if error:
                print(f"[Error]: {error}")
            print(f"[Message End] {msg.role}")

        elif event_type == "tool_execution_start":
            print(f"\n[Tool Start] {event.tool_name}({event.args})")

        elif event_type == "tool_execution_update":
            print(f"  [Tool Update] {event.partialResult}")

        elif event_type == "tool_execution_end":
            status = "✓ Success" if not event.is_error else "✗ Failed"
            print(f"[Tool End] {event.tool_name} {status}")

        elif event_type == "turn_start":
            print("\n--- New Turn Started ---")

        elif event_type == "turn_end":
            print(f"--- Turn Ended (Tool Results: {len(event.toolResults)}) ---")

        elif event_type == "agent_start":
            print("\n=== Agent Started ===")

        elif event_type == "agent_end":
            print(f"\n=== Agent Ended (Total {len(event.messages)} messages) ===")

    # 4. Register Event Listener
    agent.subscribe(on_event)

    # 5. Interactive Loop
    print("\n" + "="*60)
    print("Agent Interactive Chat Started")
    print("Type 'quit' or 'exit' to end the program")
    print("="*60 + "\n")

    turn_count = 0

    while True:
        turn_count += 1

        # Get user input
        try:
            # Use asyncio to handle user input, avoiding blocking the async event loop
            user_input = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: input("\nYou: ").strip()
            )
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break

        # Check exit conditions
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break

        # Handle empty input
        if not user_input:
            print("Please enter a valid question")
            continue

        # Send user message to agent
        print(f"\nProcessing question {turn_count}...")
        try:
            await agent.prompt(user_input)

            # Wait for agent to finish processing
            await agent.wait_for_idle()

        except Exception as e:
            print(f"Error during Agent processing: {str(e)}")
            # Continue to the next turn without interrupting the entire program
            continue

if __name__ == "__main__":
    asyncio.run(main())