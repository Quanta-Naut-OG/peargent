"""
SQLite Database Tracing Example

This example demonstrates how to use SQLite for persistent tracing storage.
SQLite is perfect for development and smaller projects as it requires no server setup.
"""

from peargent import create_agent, create_tool
from peargent.telemetry import enable_tracing
from peargent.models import groq


def calculator_func(operation: str, a: float, b: float) -> float:
    """Simple calculator tool."""
    operations = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y if y != 0 else float('inf')
    }
    return operations.get(operation, lambda x, y: 0)(a, b)


calculator = create_tool(
    name="calculator",
    description="Performs basic arithmetic operations",
    input_parameters={"operation": str, "a": float, "b": float},
    call_function=calculator_func,
)


def main():
    # Enable SQLite-based tracing
    # This will create a traces.db file in the current directory
    tracer = enable_tracing(
        store_type="sqlite",
        connection_string="sqlite:///./traces.db",
        enabled=True
    )

    print("SQLite Database Tracing Example")
    print("=" * 50)
    print(f"Database: ./traces.db")
    print(f"Tracer enabled: {tracer.enabled}")
    print()

    # Create an agent with tracing
    agent = create_agent(
        name="calculator_agent",
        description="A helpful calculator agent",
        persona="You are a helpful calculator. Use the calculator tool to perform operations.",
        model=groq("llama-3.3-70b-versatile"),
        tools=[calculator],
        tracing=True
    )

    # Run some tasks that will be traced
    print("Running calculations...")
    result1 = agent.run("What is 15 + 27?")
    print(f"Result 1: {result1}")
    print()

    result2 = agent.run("What is 100 divided by 4?")
    print(f"Result 2: {result2}")
    print()

    # Get trace information
    traces = tracer.store.list_traces()
    print(f"\nTotal traces recorded: {len(traces)}")

    # Display trace details
    for i, trace in enumerate(traces, 1):
        print(f"\nTrace {i}:")
        print(f"  Agent: {trace['agent_name']}")
        print(f"  Duration: {trace.get('duration_ms', 0)}ms")
        print(f"  Total Tokens: {trace.get('total_tokens', 0)}")
        print(f"  Total Cost: ${trace.get('total_cost', 0):.6f}")
        print(f"  Status: {'Error' if trace.get('error') else 'Success'}")

        # Get spans for this trace
        query = f"SELECT * FROM spans WHERE trace_id = ?"
        cursor = tracer.store.connection.cursor()
        cursor.execute(query, (trace['id'],))
        spans = [dict(row) for row in cursor.fetchall()]

        print(f"  Spans: {len(spans)}")
        for span in spans:
            print(f"    - {span['span_type']}: {span['name']} ({span.get('duration_ms', 0)}ms)")

    print("\n" + "=" * 50)
    print("Traces have been persisted to SQLite database!")
    print("You can query the database directly using sqlite3 or any SQLite client.")
    print("\nExample query:")
    print("  sqlite3 traces.db \"SELECT agent_name, COUNT(*) FROM traces GROUP BY agent_name;\"")


if __name__ == "__main__":
    main()
