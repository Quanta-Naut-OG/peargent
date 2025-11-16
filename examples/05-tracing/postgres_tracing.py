"""
PostgreSQL Database Tracing Example

This example demonstrates how to use PostgreSQL for production-grade tracing storage.
PostgreSQL is recommended for production environments with high traffic and advanced querying needs.

Prerequisites:
  - PostgreSQL server running
  - psycopg2-binary package installed: pip install psycopg2-binary
  - Database created: CREATE DATABASE traces_db;

Connection string format:
  postgresql://username:password@host:port/database
  Example: postgresql://postgres:password@localhost:5432/traces_db
"""

import os

from peargent import create_agent, create_tool
from peargent.models import groq
from peargent.telemetry import enable_tracing


def data_analyzer_func(operation: str, data: str) -> str:
    """Mock data analysis tool."""
    operations = {
        "count_words": lambda d: f"Word count: {len(d.split())}",
        "count_chars": lambda d: f"Character count: {len(d)}",
        "uppercase": lambda d: d.upper(),
        "lowercase": lambda d: d.lower()
    }
    return operations.get(operation, lambda d: "Unknown operation")(data)


data_analyzer = create_tool(
    name="data_analyzer",
    description="Analyzes and processes text data",
    input_parameters={"operation": str, "data": str},
    call_function=data_analyzer_func,
)


def main():
    # PostgreSQL connection string
    # In production, use environment variables for credentials
    connection_string = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:password@localhost:5432/traces_db"
    )

    print("PostgreSQL Database Tracing Example")
    print("=" * 50)

    try:
        # Enable PostgreSQL-based tracing
        tracer = enable_tracing(
            store_type="postgres",
            connection_string=connection_string,
            enabled=True,
            auto_save=True
        )

        print(f"Connected to PostgreSQL database")
        print(f"Tracer enabled: {tracer.enabled}")
        print()

        # Create an agent with tracing
        agent = create_agent(
            name="data_analysis_agent",
            description="A data analysis assistant",
            persona="You are a data analysis assistant. Use the data_analyzer tool to process data.",
            model=groq("llama-3.3-70b-versatile"),
            tools=[data_analyzer],
            tracing=True
        )

        # Run some tasks
        print("Running data analysis tasks...")
        result1 = agent.run("Count the words in this text: 'Hello world from PostgreSQL tracing'")
        print(f"Result 1: {result1}")
        print()

        result2 = agent.run("Convert to uppercase: 'postgresql is awesome'")
        print(f"Result 2: {result2}")
        print()

        # Query traces using PostgreSQL-specific features
        traces = tracer.store.list_traces()
        print(f"\nTotal traces in PostgreSQL: {len(traces)}")

        for i, trace in enumerate(traces, 1):
            print(f"\nTrace {i}:")
            print(f"  UUID: {trace['id']}")
            print(f"  Agent: {trace['agent_name']}")
            print(f"  Duration: {trace.get('duration_ms', 0)}ms")
            print(f"  Total Tokens: {trace.get('total_tokens', 0)}")
            print(f"  Total Cost: ${trace.get('total_cost', 0):.6f}")
            print(f"  Timestamp: {trace.get('start_time')}")

        print("\n" + "=" * 50)
        print("PostgreSQL tracing advantages:")
        print("  - Native UUID support")
        print("  - Advanced querying capabilities")
        print("  - Better concurrency handling")
        print("  - Full ACID compliance")
        print("  - Suitable for high-traffic production environments")
        print("\nQuery traces directly:")
        print(f"  psql -d traces_db -c \"SELECT agent_name, COUNT(*) FROM traces GROUP BY agent_name;\"")

    except ImportError as e:
        print("Error: psycopg2-binary not installed")
        print("Install it with: pip install psycopg2-binary")
        print(f"\nDetails: {e}")
    except Exception as e:
        print(f"Error connecting to PostgreSQL: {e}")
        print("\nMake sure:")
        print("  1. PostgreSQL server is running")
        print("  2. Database exists")
        print("  3. Connection string is correct")
        print(f"  4. Current connection string: {connection_string}")


if __name__ == "__main__":
    main()
