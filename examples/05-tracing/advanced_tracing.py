"""
Improved Tracing DSL Example

This example demonstrates the new, simplified tracing API.
"""

from peargent import create_agent
from peargent.telemetry import enable_tracing
from peargent.models import groq

# ============================================================================
# OPTION 1: Simplest approach - one line setup with in-memory storage
# ============================================================================

# Enable tracing and get the tracer in one call
tracer = enable_tracing()

# Create agent with tracing enabled
agent = create_agent(
    name="MathAgent",
    description="A helpful math assistant",
    persona="You are a helpful math tutor.",
    model=groq("llama-3.3-70b-versatile"),
    tracing=True
)

# Run agent - traces are automatically captured
print("Running agent with query 1...")
result1 = agent.run("What is 2+2?")
print(f"Result 1: {result1}\n")

print("Running agent with query 2...")
result2 = agent.run("What is 5*6?")
print(f"Result 2: {result2}\n")

# Access traces directly from tracer - no need to manage store separately!
print("\n" + "="*80)
print("LISTING ALL TRACES")
print("="*80 + "\n")

traces = tracer.list_traces()
print(f"Found {len(traces)} traces\n")

# Print traces with nice formatting
tracer.print_traces(limit=5, format="terminal")

# ============================================================================
# OPTION 2: File-based storage (also simple!)
# ============================================================================

print("\n" + "="*80)
print("FILE-BASED STORAGE EXAMPLE")
print("="*80 + "\n")

# Enable tracing with file storage
file_tracer = enable_tracing(
    store_type="file",
    storage_dir="./my_traces"
)

# Create a different agent
agent2 = create_agent(
    name="HistoryAgent",
    description="An agent that knows history",
    persona="You are a history expert.",
    model=groq("llama-3.3-70b-versatile"),
    tracing=True
)

print("Running history agent...")
result3 = agent2.run("When was the Declaration of Independence signed?")
print(f"Result: {result3}\n")

# List traces from file storage
file_traces = file_tracer.list_traces()
print(f"\nFound {len(file_traces)} traces in file storage")

# ============================================================================
# OPTION 3: Advanced usage with configure_tracing (for custom stores)
# ============================================================================

from peargent.telemetry import configure_tracing, FileTracingStore

# Create a custom store with specific settings
custom_store = FileTracingStore(storage_dir="./custom_traces")

# Configure tracing with the custom store
custom_tracer = configure_tracing(
    store=custom_store,
    enabled=True,
    auto_save=True
)

# Now use it the same way
agent3 = create_agent(
    name="CustomAgent",
    description="Agent with custom tracing",
    persona="You are helpful.",
    model=groq("llama-3.3-70b-versatile"),
    tracing=True
)

print("\nRunning custom agent...")
result4 = agent3.run("Hello!")
print(f"Result: {result4}\n")

# Access store if needed (though usually not necessary)
store = custom_tracer.get_store()
print(f"Store type: {type(store).__name__}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("""
The new API is much cleaner:

OLD WAY (verbose):
    store = InMemoryTracingStore()
    configure_tracing(enabled=True, store=store)
    agent = create_agent(..., tracing=True)
    tracer = get_tracer()
    traces = store.list_traces()  # Need to remember store!

NEW WAY (simple):
    tracer = enable_tracing()
    agent = create_agent(..., tracing=True)
    traces = tracer.list_traces()  # Everything from tracer!
    tracer.print_traces()  # Nice formatting built-in!
""")
