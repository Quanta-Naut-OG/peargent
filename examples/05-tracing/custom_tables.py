"""
Custom Table Names Example

This example demonstrates how to use custom table names for traces and spans.
This is useful when you want to store traces from different applications
in the same database, or when you need to follow specific naming conventions.
"""

from peargent import create_agent, create_tool
from peargent.telemetry import enable_tracing
from peargent.models import groq


def get_weather_func(city: str) -> str:
    """Mock weather tool."""
    weather_data = {
        "New York": "Sunny, 72°F",
        "London": "Cloudy, 15°C",
        "Tokyo": "Rainy, 20°C",
        "Sydney": "Clear, 25°C"
    }
    return weather_data.get(city, "Weather data not available")


get_weather = create_tool(
    name="get_weather",
    description="Gets weather information for a city",
    input_parameters={"city": str},
    call_function=get_weather_func,
)

def main():
    # Enable SQLite-based tracing with custom table names
    # This creates tables named 'my_app_traces' and 'my_app_spans'
    tracer = enable_tracing(
        store_type="sqlite",
        connection_string="sqlite:///./custom_traces.db",
        traces_table="my_app_traces",
        spans_table="my_app_spans",
        enabled=True
    )

    print("Custom Table Names Example")
    print("=" * 50)
    print(f"Database: ./custom_traces.db")
    print(f"Traces table: my_app_traces")
    print(f"Spans table: my_app_spans")
    print()

    # Create an agent with tracing
    agent = create_agent(
        name="weather_agent",
        description="A helpful weather assistant",
        persona="You are a helpful weather assistant. Use the get_weather tool to fetch weather information.",
        model=groq("llama-3.3-70b-versatile"),
        tools=[get_weather],
        tracing=True
    )

    # Run some tasks
    print("Running weather queries...")
    result1 = agent.run("What's the weather like in Tokyo?")
    print(f"Result: {result1}")
    print()

    result2 = agent.run("Tell me about the weather in Sydney.")
    print(f"Result: {result2}")
    print()

    # Query traces from custom table
    query = f"SELECT * FROM {tracer.store.traces_table} ORDER BY created_at DESC"
    cursor = tracer.store.connection.cursor()
    cursor.execute(query)
    traces = [dict(row) for row in cursor.fetchall()]

    print(f"Traces in '{tracer.store.traces_table}' table: {len(traces)}")

    for i, trace in enumerate(traces, 1):
        print(f"\nTrace {i}:")
        print(f"  ID: {trace['id']}")
        print(f"  Agent: {trace['agent_name']}")
        print(f"  Duration: {trace.get('duration_ms', 0)}ms")

        # Query spans from custom table
        span_query = f"SELECT * FROM {tracer.store.spans_table} WHERE trace_id = ?"
        cursor.execute(span_query, (trace['id'],))
        spans = [dict(row) for row in cursor.fetchall()]

        print(f"  Spans in '{tracer.store.spans_table}' table: {len(spans)}")

    print("\n" + "=" * 50)
    print("Custom table names allow you to:")
    print("  - Organize traces from multiple applications")
    print("  - Follow database naming conventions")
    print("  - Avoid table name conflicts")
    print("  - Implement multi-tenancy patterns")


if __name__ == "__main__":
    main()
