"""
Example demonstrating PostgreSQL history storage in peargent.

This example shows how to:
1. Set up PostgreSQL history storage
2. Use PostgreSQL with agents for persistent conversations
3. Resume conversations across program restarts
4. Query and manage conversations in PostgreSQL
5. Use context management with PostgreSQL

Prerequisites:
- PostgreSQL server running
- SQLAlchemy installed: pip install sqlalchemy
- Database created (or use default 'postgres' database)

Note: Uses SQLAlchemy for secure, parameterized queries (SQL injection protection)
"""

import os
from peargent import create_agent, create_history, create_tool, Postgresql, Redis, HistoryConfig
from peargent.models import groq

# PostgreSQL connection configuration
# Uses connection string from .env file for security
CONNECTION_STRING = os.getenv("DATABASE_URL")

if not CONNECTION_STRING:
    print("❌ ERROR: DATABASE_URL not found in environment variables!")
    print("Please add your PostgreSQL connection string to .env file:")
    print('DATABASE_URL="postgresql://user:password@host:port/database"')
    exit(1)


def example_1_basic_postgresql():
    """Example 1: Basic PostgreSQL history setup."""
    print("\n" + "="*60)
    print("Example 1: Basic PostgreSQL History")
    print("="*60)

    try:
        # Create history with PostgreSQL backend
        history = create_history(
            store_type=Postgresql(
                connection_string=CONNECTION_STRING,
                table_prefix="peargent"  # Optional: customize table names
            )
        )

        # Create a new conversation thread
        thread_id = history.create_thread(metadata={
            "user_id": "user_123",
            "session": "demo_session",
            "app": "example_app"
        })
        print(f"✅ Created thread: {thread_id}")

        # Create agent with PostgreSQL history using HistoryConfig DSL
        agent = create_agent(
            name="Assistant",
            description="A helpful assistant",
            persona="You are a friendly AI assistant.",
            model=groq("llama-3.3-70b-versatile"),
            tools=[],
            history=HistoryConfig(
                auto_manage_context=True,
                max_context_messages=20,
                strategy="smart",
                summarize_model=groq("llama-3.1-8b-instant"),  # Use faster model for summaries
                store=history
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
        print(f"\n✅ Total messages stored in PostgreSQL: {message_count}")
        print(f"✅ Thread ID (save this for later): {thread_id}")

        return thread_id

    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure:")
        print("1. PostgreSQL server is running")
        print("2. Database exists or you have permission to create tables")
        print("3. SQLAlchemy is installed: pip install sqlalchemy")
        print("4. Connection credentials are correct")
        return None


def example_2_resume_conversation(thread_id: str):
    """Example 2: Resume conversation from PostgreSQL."""
    print("\n" + "="*60)
    print("Example 2: Resume Conversation from PostgreSQL")
    print("="*60)

    if not thread_id:
        print("⚠️ Skipping - no thread_id from example 1")
        return

    # Create NEW history instance (simulating program restart)
    history = create_history(
        store_type=Postgresql(
            connection_string=CONNECTION_STRING
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
            max_context_messages=20,
            strategy="smart",
            store=history  # Use existing history instance
        )
    )

    print("\n--- Continuing Conversation ---")
    response = agent.run("Do you remember my name?")
    print(f"Agent: {response[:150]}...")


def example_3_query_conversations():
    """Example 3: Query and manage conversations."""
    print("\n" + "="*60)
    print("Example 3: Query Conversations")
    print("="*60)

    history = create_history(
        store_type=Postgresql(
            connection_string=CONNECTION_STRING
        )
    )

    # List all threads
    print("\n--- All Conversation Threads ---")
    threads = history.list_threads()
    print(f"Found {len(threads)} conversation(s)")

    for i, thread_id in enumerate(threads[:5], 1):  # Show first 5
        thread = history.get_thread(thread_id)
        if thread:
            print(f"\n{i}. Thread: {thread_id[:20]}...")
            print(f"   Created: {thread.created_at}")
            print(f"   Updated: {thread.updated_at}")
            print(f"   Messages: {len(thread.messages)}")
            print(f"   Metadata: {thread.metadata}")

    # Query specific thread details
    if threads:
        print(f"\n--- Detailed View: {threads[0][:20]}... ---")
        history.use_thread(threads[0])
        thread = history.get_thread()

        user_msgs = thread.get_messages(role="user")
        assistant_msgs = thread.get_messages(role="assistant")

        print(f"User messages: {len(user_msgs)}")
        print(f"Assistant messages: {len(assistant_msgs)}")


def example_4_context_management_postgres():
    """Example 4: Context management with PostgreSQL."""
    print("\n" + "="*60)
    print("Example 4: Context Management with PostgreSQL")
    print("="*60)

    history = create_history(
        store_type=Postgresql(
            connection_string=CONNECTION_STRING
        )
    )

    # Create a new thread for this example
    thread_id = history.create_thread(metadata={"example": "context_mgmt"})
    print(f"Created thread: {thread_id}")

    # Add many messages
    print("\n--- Adding 30 messages ---")
    for i in range(15):
        history.add_user_message(f"User message {i}")
        history.add_assistant_message(f"Assistant response {i}")

    print(f"Messages before management: {history.get_message_count()}")

    # Apply smart context management
    model = groq("llama-3.3-70b-versatile")
    print("\n--- Applying smart context management ---")
    
    # Let's use a more aggressive trim strategy to see the effect
    print("Using 'trim_last' strategy to keep only 10 messages...")
    history.manage_context_window(model, max_messages=10, strategy="trim_last")

    print(f"Messages after management: {history.get_message_count()}")
    
    # Show the remaining messages
    remaining_messages = history.get_messages()
    print(f"Remaining messages:")
    for i, msg in enumerate(remaining_messages[-5:], 1):  # Show last 5
        print(f"  {i}. [{msg.role}] {str(msg.content)[:50]}...")
    
    print("✅ Context managed and saved to PostgreSQL")


def example_5_multi_user_threads():
    """Example 5: Managing multiple user conversations."""
    print("\n" + "="*60)
    print("Example 5: Multi-User Conversations")
    print("="*60)

    history = create_history(
        store_type=Postgresql(
            connection_string=CONNECTION_STRING
        )
    )

    # Create threads for different users
    print("\n--- Creating user sessions ---")
    users = ["alice", "bob", "charlie"]
    user_threads = {}

    for user in users:
        thread_id = history.create_thread(metadata={
            "user_id": user,
            "app": "chat_app",
            "created_by": "example_5"
        })
        user_threads[user] = thread_id
        print(f"✅ Created thread for {user}: {thread_id[:20]}...")

    # Have separate conversations using HistoryConfig
    agent = create_agent(
        name="Assistant",
        description="A helpful assistant",
        persona="You are a helpful AI assistant.",
        model=groq("llama-3.3-70b-versatile"),
        tools=[],
        history=HistoryConfig(
            auto_manage_context=True,
            max_context_messages=15,
            strategy="trim_last",  # Use trim for multi-user scenarios
            store=history
        )
    )

    print("\n--- Separate Conversations ---")
    for user, thread_id in user_threads.items():
        history.use_thread(thread_id)
        response = agent.run(f"Hello! I'm {user}.")
        print(f"{user}: {response[:80]}...")

    print(f"\n✅ All conversations stored separately in PostgreSQL")


def example_6_cleanup():
    """Example 6: Cleanup old conversations."""
    print("\n" + "="*60)
    print("Example 6: Cleanup Operations")
    print("="*60)

    history = create_history(
        store_type=Postgresql(
            connection_string=CONNECTION_STRING
        )
    )

    threads = history.list_threads()
    print(f"Total threads before cleanup: {len(threads)}")

    # Find threads from example 5
    print("\n--- Finding threads to delete ---")
    to_delete = []
    for thread_id in threads:
        thread = history.get_thread(thread_id)
        if thread and thread.metadata.get("created_by") == "example_5":
            to_delete.append(thread_id)

    print(f"Found {len(to_delete)} threads to delete")

    # Delete them
    for thread_id in to_delete:
        deleted = history.delete_thread(thread_id)
        if deleted:
            print(f"✅ Deleted: {thread_id[:20]}...")

    print(f"\nTotal threads after cleanup: {len(history.list_threads())}")


def main():
    """Run all examples."""
    print("=" * 60)
    print("PostgreSQL History Storage Examples")
    print("=" * 60)
    print(f"\nConnection: {CONNECTION_STRING}")
    print("\n⚠️  Make sure PostgreSQL is running and credentials are correct!")

    # Example 1: Basic setup
    thread_id = example_1_basic_postgresql()

    if thread_id:
        # Example 2: Resume conversation
        # example_2_resume_conversation(thread_id)

        # Example 3: Query conversations
        # example_3_query_conversations()

        # Example 4: Context management
        example_4_context_management_postgres()

        # Example 5: Multi-user threads
        # example_5_multi_user_threads()

        # Example 6: Cleanup
        # example_6_cleanup()

    print("\n" + "="*60)
    print("Examples completed!")
    print("="*60)
    print("\n💡 Tips:")
    print("1. Thread IDs persist across program restarts")
    print("2. Use metadata to organize conversations")
    print("3. PostgreSQL provides ACID guarantees for your data")
    print("4. Perfect for production applications with multiple users")


if __name__ == "__main__":
    main()
