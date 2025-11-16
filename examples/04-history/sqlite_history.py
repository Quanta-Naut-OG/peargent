"""
Example demonstrating SQLite history storage in peargent.

This example shows how to:
1. Set up SQLite history storage
2. Use SQLite with agents for persistent conversations
3. Resume conversations across program restarts
4. Query and manage conversations in SQLite

Prerequisites:
- SQLAlchemy installed: pip install sqlalchemy

Note: SQLite is perfect for:
- Desktop applications
- Mobile apps
- Single-user systems
- Local development
- No server setup required!
"""

from peargent import create_agent, create_history, Sqlite, HistoryConfig
from peargent.models import groq


def example_1_basic_sqlite():
    """Example 1: Basic SQLite history setup."""
    print("\n" + "="*60)
    print("Example 1: Basic SQLite History")
    print("="*60)

    # Create history with SQLite backend
    # Creates a local database file - no server needed!
    history = create_history(
        store_type=Sqlite(
            database_path="./chat_app.db"  # Local file
        )
    )

    # Create a new conversation thread
    thread_id = history.create_thread(metadata={
        "user": "alice",
        "app_version": "1.0.0"
    })
    print(f"✅ Created thread: {thread_id}")

    # Create agent with SQLite history using HistoryConfig DSL
    agent = create_agent(
        name="Assistant",
        description="A helpful assistant",
        persona="You are a friendly AI assistant.",
        model=groq("llama-3.3-70b-versatile"),
        tools=[],
        history=HistoryConfig(
            auto_manage_context=True,
            max_context_messages=15,
            strategy="smart",
            summarize_model=groq("llama-3.1-8b-instant"),
            store=Sqlite(database_path="./chat_app.db")
        )
    )

    # Have a conversation
    print("\n--- Conversation ---")
    response1 = agent.run("Hello! My name is Alice.")
    print(f"Agent: {response1[:150]}...")

    response2 = agent.run("What's my name?")
    print(f"Agent: {response2[:150]}...")

    # Check message count
    message_count = history.get_message_count()
    print(f"\n✅ Total messages in SQLite: {message_count}")
    print(f"✅ Thread ID (save this for later): {thread_id}")
    print(f"✅ Database file: ./chat_app.db")

    return thread_id


def example_2_resume_conversation(thread_id: str):
    """Example 2: Resume conversation from SQLite."""
    print("\n" + "="*60)
    print("Example 2: Resume Conversation from SQLite")
    print("="*60)

    # Create NEW history instance (simulating program restart)
    # Same database file = persistent data!
    history = create_history(
        store_type=Sqlite(
            database_path="./chat_app.db"
        )
    )

    # Resume previous thread
    history.use_thread(thread_id)
    print(f"✅ Resumed thread: {thread_id}")

    # Show previous messages
    print("\n--- Previous Messages ---")
    messages = history.get_messages()
    for msg in messages[:4]:  # Show first 4
        role = msg.role.upper()
        content = str(msg.content)[:80]
        print(f"[{role}] {content}...")

    # Create agent and continue using HistoryConfig
    agent = create_agent(
        name="Assistant",
        description="A helpful assistant",
        persona="You are a friendly AI assistant.",
        model=groq("llama-3.3-70b-versatile"),
        tools=[],
        history=HistoryConfig(
            auto_manage_context=True,
            max_context_messages=15,
            strategy="smart",
            store=history  # Use existing history instance
        )
    )

    print("\n--- Continuing Conversation ---")
    response = agent.run("Do you remember my name?")
    print(f"Agent: {response[:150]}...")


def example_3_multiple_threads():
    """Example 3: Manage multiple conversation threads."""
    print("\n" + "="*60)
    print("Example 3: Multiple Threads in SQLite")
    print("="*60)

    history = create_history(store_type=Sqlite(database_path="./chat_app.db"))

    # Create multiple threads
    print("\n--- Creating threads for different topics ---")
    thread1 = history.create_thread(metadata={"topic": "math"})
    thread2 = history.create_thread(metadata={"topic": "science"})
    thread3 = history.create_thread(metadata={"topic": "history"})

    print(f"Created 3 threads: {thread1[:20]}..., {thread2[:20]}..., {thread3[:20]}...")

    # List all threads
    all_threads = history.list_threads()
    print(f"\n✅ Total threads in database: {len(all_threads)}")


def example_4_context_management():
    """Example 4: Context management with SQLite."""
    print("\n" + "="*60)
    print("Example 4: Context Management with SQLite")
    print("="*60)

    history = create_history(store_type=Sqlite(database_path="./chat_app.db"))
    thread_id = history.create_thread()

    # Add many messages
    print("\n--- Adding 25 messages ---")
    for i in range(12):
        history.add_user_message(f"Question {i+1}")
        history.add_assistant_message(f"Answer {i+1}")

    print(f"Messages before trimming: {history.get_message_count()}")

    # Trim to keep only last 15 messages
    removed = history.trim_messages(strategy="last", count=15)
    print(f"Removed {removed} messages")
    print(f"Messages after trimming: {history.get_message_count()}")
    print("✅ Changes saved to SQLite database")


def example_5_auto_context_management():
    """Example 5: Automatic context management."""
    print("\n" + "="*60)
    print("Example 5: Auto Context Management")
    print("="*60)

    history = create_history(store_type=Sqlite(database_path="./chat_app.db"))
    thread_id = history.create_thread(metadata={"auto_managed": True})

    # Create agent with auto context management using HistoryConfig DSL
    agent = create_agent(
        name="Assistant",
        description="A helpful assistant",
        persona="You are a helpful AI assistant.",
        model=groq("llama-3.3-70b-versatile"),
        history=HistoryConfig(
            auto_manage_context=True,     # Auto-manage!
            max_context_messages=10,      # Keep under 10
            strategy="smart",
            summarize_model=groq("llama-3.1-8b-instant"),
            store=history
        )
    )

    print("\n--- Having a long conversation ---")
    for i in range(7):
        response = agent.run(f"Tell me fact #{i+1} about Python programming")
        print(f"Turn {i+1}: {response[:80]}...")

    print(f"\n✅ Final message count: {history.get_message_count()}")
    print("(Context automatically managed)")


def main():
    """Run all examples."""
    print("=" * 60)
    print("SQLite History Storage Examples")
    print("=" * 60)
    print("\n💡 SQLite = Local database, no server needed!")

    # Example 1: Create and use SQLite history
    thread_id = example_1_basic_sqlite()

    # Example 2: Resume conversation
    example_2_resume_conversation(thread_id)

    # Example 3: Multiple threads
    example_3_multiple_threads()

    # Example 4: Context management
    example_4_context_management()

    # Example 5: Auto context management
    example_5_auto_context_management()

    print("\n" + "="*60)
    print("Examples completed!")
    print("="*60)
    print("\n💡 Tips:")
    print("1. Database file persists across program restarts")
    print("2. Perfect for desktop and mobile applications")
    print("3. Much faster than JSON files")
    print("4. SQL injection protected via SQLAlchemy")
    print("5. No server setup or configuration needed")
    print(f"\n📁 Database location: ./chat_app.db")


if __name__ == "__main__":
    main()
