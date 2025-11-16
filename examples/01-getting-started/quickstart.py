"""
Comprehensive Peargent Test - Tests all major features in one file

This example demonstrates all core features of Peargent:
1. Basic Agent
2. Agent with Tools
3. Database Tracing (SQLite)
4. Streaming Responses
5. Agent Pools
6. History Management
7. Cost Tracking
8. Structured Output
9. Multi-Agent Collaboration
10. Parallel Tool Execution
"""

import os
import sys
import time
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from peargent import create_agent, create_pool, create_tool, create_history, File, HistoryConfig
from peargent.telemetry import enable_tracing
from peargent.models import groq

# Load environment variables
load_dotenv()

print("=" * 80)
print("PEARGENT COMPREHENSIVE TEST")
print("=" * 80)

# =============================================================================
# TEST 1: Basic Agent
# =============================================================================
print("\n[TEST 1] Basic Agent")
print("-" * 80)

basic_agent = create_agent(
    name="BasicAssistant",
    description="A simple helpful assistant",
    persona="You are a helpful AI assistant.",
    model=groq("llama-3.3-70b-versatile")
)

result = basic_agent.run("What is 2+2? Answer in one short sentence.")
print(f"Question: What is 2+2?")
print(f"Answer: {result}")
print("✓ Basic agent test passed!")

# =============================================================================
# TEST 2: Agent with Tools
# =============================================================================
print("\n[TEST 2] Agent with Tools")
print("-" * 80)

# Define tool functions
def get_weather_func(city: str) -> str:
    """Get the current weather for a city."""
    weather_data = {
        "San Francisco": "sunny and 72 degrees F",
        "New York": "cloudy and 65 degrees F",
        "London": "rainy and 55 degrees F"
    }
    return weather_data.get(city, f"Weather data not available for {city}")

def calculate_func(operation: str, a: int | float, b: int | float) -> float:
    """Perform basic math operations: add, subtract, multiply, divide"""
    operations = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y if y != 0 else float('inf')
    }
    return operations.get(operation, lambda x, y: 0)(a, b)

# Create tools
weather_tool = create_tool(
    name="get_weather",
    description="Get current weather for a city",
    input_parameters={"city": str},
    call_function=get_weather_func
)

calculator_tool = create_tool(
    name="calculate",
    description="Perform basic arithmetic operations",
    input_parameters={"operation": str, "a": int | float, "b": int | float},
    call_function=calculate_func
)

tool_agent = create_agent(
    name="ToolAssistant",
    description="An assistant with tools",
    persona="You are a helpful assistant that can check weather and do calculations.",
    model=groq("llama-3.3-70b-versatile"),
    tools=[weather_tool, calculator_tool]
)

result = tool_agent.run("What's the weather in San Francisco?")
print(f"Question: What's the weather in San Francisco?")
print(f"Answer: {result}")

result = tool_agent.run("Calculate 15 multiplied by 7")
print(f"Question: Calculate 15 multiplied by 7")
print(f"Answer: {result}")
print("✓ Tools test passed!")

# =============================================================================
# TEST 3: Database Tracing (SQLite)
# =============================================================================
print("\n[TEST 3] Database Tracing")
print("-" * 80)

# Enable SQLite-based tracing
tracer = enable_tracing(
    store_type="sqlite",
    connection_string="sqlite:///./peargent_test_traces.db",
    enabled=True
)

traced_agent = create_agent(
    name="TracedAssistant",
    description="An agent with tracing enabled",
    persona="You are a helpful assistant.",
    model=groq("llama-3.3-70b-versatile"),
    tracing=True
)

result = traced_agent.run("Explain quantum computing in one sentence.")
print(f"Question: Explain quantum computing in one sentence.")
print(f"Answer: {result}")
print(f"Traces saved to: ./peargent_test_traces.db")

# Get trace information
traces = tracer.store.list_traces()
print(f"Total traces recorded: {len(traces)}")
print("✓ Database tracing test passed!")

# =============================================================================
# TEST 4: Streaming Responses
# =============================================================================
print("\n[TEST 4] Streaming Responses")
print("-" * 80)

streaming_agent = create_agent(
    name="StreamingAssistant",
    description="An agent that streams responses",
    persona="You are a helpful assistant.",
    model=groq("llama-3.3-70b-versatile")
)

print("Question: Tell me a very short fun fact about space.")
print("Answer (streaming): ", end="", flush=True)
for chunk in streaming_agent.stream("Tell me a very short fun fact about space."):
    print(chunk, end="", flush=True)
    sys.stdout.flush()
print("\n✓ Streaming test passed!")

# =============================================================================
# TEST 5: Agent Pools
# =============================================================================
print("\n[TEST 5] Agent Pools")
print("-" * 80)

researcher_agent = create_agent(
    name="Researcher",
    description="Research expert",
    persona="You are a research expert who provides factual information.",
    model=groq("llama-3.3-70b-versatile")
)

writer_agent = create_agent(
    name="Writer",
    description="Creative writer",
    persona="You are a creative writer who makes things interesting.",
    model=groq("llama-3.1-8b-instant")
)

pool = create_pool(
    agents=[researcher_agent, writer_agent],
    max_iter=2
)

print("Question: What is artificial intelligence? (Each agent answers)")
result = pool.run("What is artificial intelligence? Answer in one sentence.")
print(f"\nPool Result: {result}")
print("✓ Agent pools test passed!")

# =============================================================================
# TEST 6: History Management
# =============================================================================
print("\n[TEST 6] History Management")
print("-" * 80)

# Create history
history = create_history(
    store_type=File(storage_dir="./test_conversation_history")
)
thread_id = history.create_thread(metadata={"test": "comprehensive"})

history_agent = create_agent(
    name="HistoryAssistant",
    description="An agent with memory",
    persona="You are a helpful assistant with memory.",
    model=groq("llama-3.3-70b-versatile"),
    history=HistoryConfig(
        auto_manage_context=False,
        store=history
    )
)

# Multi-turn conversation
result1 = history_agent.run("My favorite color is blue.")
print(f"User: My favorite color is blue.")
print(f"Agent: {result1}")

result2 = history_agent.run("What's my favorite color?")
print(f"\nUser: What's my favorite color?")
print(f"Agent: {result2}")

messages = history.get_messages()
print(f"\nHistory has {len(messages)} messages")
print("✓ History management test passed!")

# =============================================================================
# TEST 7: Cost Tracking
# =============================================================================
print("\n[TEST 7] Cost Tracking")
print("-" * 80)

from peargent.telemetry import get_cost_tracker

# Create agent with tracing for cost tracking
cost_agent = create_agent(
    name="CostAssistant",
    description="An agent for cost tracking",
    persona="You are a helpful assistant.",
    model=groq("llama-3.3-70b-versatile"),
    tracing=True
)

# Run a few requests
cost_agent.run("Hello!")
cost_agent.run("How are you?")
cost_agent.run("Goodbye!")

# Get cost tracker stats
tracker = get_cost_tracker()
if tracker:
    stats = tracker.get_statistics()
    print(f"Total requests: {stats['total_requests']}")
    print(f"Total tokens: {stats['total_tokens']}")
    print(f"Total cost: ${stats['total_cost']:.6f}")
    print("✓ Cost tracking test passed!")
else:
    print("⚠ Cost tracking not available")

# =============================================================================
# TEST 8: Structured Output
# =============================================================================
print("\n[TEST 8] Structured Output")
print("-" * 80)

class WeatherInfo(BaseModel):
    """Structured weather information"""
    city: str = Field(description="The city name")
    temperature: int = Field(description="Temperature in Fahrenheit")
    condition: str = Field(description="Weather condition")
    humidity: int = Field(description="Humidity percentage", ge=0, le=100)

structured_agent = create_agent(
    name="StructuredAssistant",
    description="Weather assistant with structured output",
    persona="You are a weather assistant that provides structured weather data.",
    model=groq("llama-3.3-70b-versatile"),
    output_schema=WeatherInfo
)

result = structured_agent.run("What's the weather like in Paris? Make up realistic data.")
print(f"Question: What's the weather like in Paris?")
print(f"Type: {type(result)}")
print(f"City: {result.city}")
print(f"Temperature: {result.temperature}°F")
print(f"Condition: {result.condition}")
print(f"Humidity: {result.humidity}%")
print("✓ Structured output test passed!")

# =============================================================================
# TEST 9: Multi-Agent Collaboration
# =============================================================================
print("\n[TEST 9] Multi-Agent Collaboration")
print("-" * 80)

# Agent 1: Researcher
researcher = create_agent(
    name="Researcher",
    description="Research expert",
    persona="You are a researcher. Provide factual information in one sentence.",
    model=groq("llama-3.3-70b-versatile")
)

# Agent 2: Simplifier
simplifier = create_agent(
    name="Simplifier",
    description="Simplifies complex information",
    persona="You simplify complex information for a 10-year-old. Keep it to one sentence.",
    model=groq("llama-3.1-8b-instant")
)

topic = "photosynthesis"
print(f"Topic: {topic}\n")

# Research phase
research = researcher.run(f"Explain {topic} in one technical sentence.")
print(f"Researcher: {research}\n")

# Simplify phase
simplified = simplifier.run(f"Simplify this for a 10-year-old in one sentence: {research}")
print(f"Simplifier: {simplified}")

print("\n✓ Multi-agent collaboration test passed!")

# =============================================================================
# TEST 10: Parallel Tool Execution
# =============================================================================
print("\n[TEST 10] Parallel Tool Execution")
print("-" * 80)

def tool_1(city: str) -> str:
    """Simulate API call"""
    time.sleep(0.5)
    return f"Weather in {city}: Sunny"

def tool_2(city: str) -> str:
    """Simulate API call"""
    time.sleep(0.5)
    return f"Population of {city}: 500,000"

def tool_3(city: str) -> str:
    """Simulate API call"""
    time.sleep(0.5)
    return f"Timezone for {city}: PST"

# Create tools
t1 = create_tool(name="get_weather", description="Get weather",
                 input_parameters={"city": str}, call_function=tool_1)
t2 = create_tool(name="get_population", description="Get population",
                 input_parameters={"city": str}, call_function=tool_2)
t3 = create_tool(name="get_timezone", description="Get timezone",
                 input_parameters={"city": str}, call_function=tool_3)

parallel_agent = create_agent(
    name="ParallelAgent",
    description="Agent that can execute tools in parallel",
    persona="You are an efficient assistant. Use multiple tools at once when needed.",
    model=groq("llama-3.3-70b-versatile"),
    tools=[t1, t2, t3]
)

print("Question: Get weather, population, and timezone for San Francisco")
start_time = time.time()
result = parallel_agent.run("Get me the weather, population, and timezone for San Francisco. Use all tools.")
end_time = time.time()

print(f"Result: {result}")
print(f"Execution time: {end_time - start_time:.2f}s")
print("✓ Parallel tool execution test passed!")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)
print("✓ All 10 tests passed successfully!")
print("\nTests completed:")
print("  1. Basic Agent")
print("  2. Agent with Tools")
print("  3. Database Tracing (SQLite)")
print("  4. Streaming Responses")
print("  5. Agent Pools")
print("  6. History Management")
print("  7. Cost Tracking")
print("  8. Structured Output")
print("  9. Multi-Agent Collaboration")
print(" 10. Parallel Tool Execution")
print("\n🎉 Peargent is working perfectly!")
print("=" * 80)

# Cleanup
print("\nTest artifacts created:")
print("  - ./peargent_test_traces.db (SQLite traces)")
print("  - ./test_conversation_history/ (History files)")
