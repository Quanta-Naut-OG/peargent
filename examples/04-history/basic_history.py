"""
Example demonstrating persistent conversation history in peargent.

This example shows how to:
1. Create a history manager with file-based storage
2. Use history with single agents
3. Resume conversations from persistent storage
4. Use history with agent pools
5. Query and inspect conversation history
"""

import os
from peargent import (
    create_agent,
    create_history,
    create_pool,
    create_tool,
    File,
    HistoryConfig,
)
from peargent.models import gemini, groq
from peargent.models.openai import OpenAIModel

def example_1_basic_history():
    """Example 1: Basic agent with persistent history."""
    print("\n" + "="*60)
    print("Example 1: Basic Agent with Persistent History")
    print("="*60)

    # Create a history manager with file-based storage
    history = create_history(
        store_type=File(
            storage_dir="./conversation_history"
        )
    )

    # Create a new conversation thread
    thread_id = history.create_thread(metadata={
        "user": "demo_user",
        "topic": "math_help"
    })
    print(f"Created new conversation thread: {thread_id}")

    # Create an agent with history using new HistoryConfig DSL
    math_agent = create_agent(
        name="MathTeacher",
        description="A helpful math teacher",
        persona="You are a patient and encouraging math teacher who helps students understand concepts.",
        model=groq("llama-3.3-70b-versatile"),
        tools=[],
        history=HistoryConfig(
            auto_manage_context=False,  # Manual management for this example
            store=history  # Use the existing history instance
        )
    )

    # First interaction
    print("\n--- First Interaction ---")
    response1 = math_agent.run("What is 15 + 27?")
    print(f"Agent: {response1}")

    # Second interaction - agent has context from first
    print("\n--- Second Interaction ---")
    response2 = math_agent.run("Can you show me how to do that with carrying?")
    print(f"Agent: {response2}")

    # View conversation history
    print("\n--- Conversation History ---")
    messages = history.get_messages()
    for i, msg in enumerate(messages, 1):
        print(f"{i}. [{msg.role}] {msg.content[:100]}...")

    print(f"\nTotal messages in thread: {len(messages)}")
    return thread_id


def example_2_resume_conversation(thread_id: str):
    """Example 2: Resume a conversation from history."""
    print("\n" + "="*60)
    print("Example 2: Resume Conversation from History")
    print("="*60)

    # Create a NEW history manager instance (simulating a new session)
    history = create_history(
        store_type=File(
            storage_dir="./conversation_history"
        )
    )

    # Resume the previous conversation
    history.use_thread(thread_id)
    print(f"Resumed conversation thread: {thread_id}")

    # Show previous messages
    print("\n--- Previous Messages ---")
    messages = history.get_messages()
    for msg in messages:
        print(f"[{msg.role}] {msg.content[:80]}...")

    # Create agent and continue the conversation using HistoryConfig
    math_agent = create_agent(
        name="MathTeacher",
        description="A helpful math teacher",
        persona="You are a patient and encouraging math teacher who helps students understand concepts.",
        model=groq("llama-3.3-70b-versatile"),
        tools=[],
        history=HistoryConfig(
            auto_manage_context=False,
            store=history
        )
    )

    print("\n--- Continuing Conversation ---")
    response = math_agent.run("Now what is 100 - 47?")
    print(f"Agent: {response}")


def example_3_history_with_tools():
    """Example 3: History with tool-using agents."""
    print("\n" + "="*60)
    print("Example 3: History with Tool-Using Agents")
    print("="*60)

    # Create history
    history = create_history(store_type=File(storage_dir="./conversation_history"))
    thread_id = history.create_thread(metadata={"topic": "calculations"})

    # Create a simple calculator tool
    def calculate(expression: str) -> float:
        """Safely evaluate a mathematical expression."""
        try:
            # Remove spaces and evaluate
            result = eval(expression, {"__builtins__": {}}, {})
            return result
        except Exception as e:
            return f"Error: {str(e)}"

    calc_tool = create_tool(
        name="calculator",
        description="Evaluate mathematical expressions",
        input_parameters={"expression": str},
        call_function=calculate
    )

    # Create agent with tools and history using HistoryConfig
    calc_agent = create_agent(
        name="Calculator",
        description="An agent that can perform calculations",
        persona="You are a helpful calculator assistant. Use the calculator tool to compute results.",
        model=groq("llama-3.3-70b-versatile"),
        tools=[calc_tool],
        history=HistoryConfig(
            auto_manage_context=False,
            store=history
        )
    )

    # Run some calculations
    print("\n--- First Calculation ---")
    response1 = calc_agent.run("What is 123 * 456?")
    print(f"Agent: {response1}")

    print("\n--- Second Calculation ---")
    response2 = calc_agent.run("And what about that result divided by 7?")
    print(f"Agent: {response2}")

    # View history including tool calls
    print("\n--- History with Tool Calls ---")
    messages = history.get_messages()
    for msg in messages:
        if msg.role == "tool":
            print(f"[TOOL] {msg.tool_call['name']}: {msg.tool_call['output']}")
        else:
            content = str(msg.content)[:80]
            print(f"[{msg.role.upper()}] {content}...")


def example_4_history_with_pool():
    """Example 4: History with multi-agent pools."""
    print("\n" + "="*60)
    print("Example 4: History with Multi-Agent Pools")
    print("="*60)

    # Create history
    history = create_history(store_type=File(storage_dir="./conversation_history"))
    thread_id = history.create_thread(metadata={"type": "multi_agent"})

    # Create multiple agents
    researcher = create_agent(
        name="Researcher",
        description="Research and gather information",
        persona="You are a thorough researcher who finds relevant information.",
        model=groq("llama-3.3-70b-versatile"),
        tools=[]
    )

    writer = create_agent(
        name="Writer",
        description="Write clear summaries",
        persona="You are a skilled writer who creates clear, concise summaries.",
        model=groq("llama-3.3-70b-versatile"),
        tools=[]
    )

    # Create pool with history using HistoryConfig
    pool = create_pool(
        agents=[researcher, writer],
        default_model=groq("llama-3.3-70b-versatile"),
        max_iter=3,
        history=HistoryConfig(
            auto_manage_context=True,  # Enable auto-management for pool
            max_context_messages=15,
            strategy="smart",
            store=history
        )
    )

    print("\n--- Running Multi-Agent Pool ---")
    result = pool.run("Research the benefits of exercise and write a brief summary.")
    print(f"\nFinal Result: {result}")

    # View all messages including user input
    print("\n--- Conversation Messages ---")
    all_messages = history.get_messages()
    for msg in all_messages:
        if msg.role == "user":
            print(f"[USER] {str(msg.content)[:100]}...")
        else:
            print(f"[{msg.agent.upper()}] {str(msg.content)[:100]}...")

    # Show agent-specific participation
    print("\n--- Agent Participation (assistants only) ---")
    messages = history.get_messages(role="assistant")
    for msg in messages:
        print(f"  - {msg.agent}: {len(str(msg.content))} characters")


def example_5_query_history():
    """Example 5: Query and inspect conversation history."""
    print("\n" + "="*60)
    print("Example 5: Query and Inspect History")
    print("="*60)

    history = create_history(store_type=File(storage_dir="./conversation_history"))

    # List all conversation threads
    print("\n--- All Conversation Threads ---")
    threads = history.list_threads()
    print(f"Found {len(threads)} conversation threads:")
    for thread_id in threads[:5]:  # Show first 5
        thread = history.get_thread(thread_id)
        if thread:
            print(f"  - {thread_id}")
            print(f"    Created: {thread.created_at}")
            print(f"    Messages: {len(thread.messages)}")
            print(f"    Metadata: {thread.metadata}")

    # Inspect a specific thread
    if threads:
        print(f"\n--- Detailed View of Thread: {threads[0]} ---")
        history.use_thread(threads[0])
        thread = history.get_thread(threads[0])

        # Get messages by role
        user_messages = thread.get_messages(role="user")
        assistant_messages = thread.get_messages(role="assistant")

        print(f"User messages: {len(user_messages)}")
        print(f"Assistant messages: {len(assistant_messages)}")

        # Show conversation flow
        print("\n--- Conversation Flow ---")
        for msg in thread.messages:
            timestamp = msg.timestamp.strftime("%H:%M:%S")
            content = str(msg.content)[:60]
            agent_info = f" ({msg.agent})" if msg.agent else ""
            print(f"[{timestamp}] {msg.role.upper()}{agent_info}: {content}...")


def main():
    """Run all examples."""
    print("Persistent Conversation History Examples")
    print("=" * 60)

    # Example 1: Create a conversation with history
    thread_id = example_1_basic_history()

    # Example 2: Resume the conversation
    # example_2_resume_conversation(thread_id)

    # Example 3: History with tools
    # example_3_history_with_tools()

    # Example 4: History with pools
    # example_4_history_with_pool()

    # Example 5: Query history
    # example_5_query_history()

    print("\n" + "="*60)
    print("All examples completed!")
    print(f"History files saved in: ./conversation_history")
    print("="*60)


if __name__ == "__main__":
    main()
