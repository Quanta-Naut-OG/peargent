"""
Advanced Streaming with stream_observe()

Demonstrates rich streaming updates with metadata like tokens, cost, and timing.
"""

import sys
from peargent import create_agent
from peargent.telemetry import enable_tracing
from peargent.models import groq

# Enable tracing
tracer = enable_tracing()

# Create agent
agent = create_agent(
    name="ObservableAgent",
    description="An agent with observable execution",
    persona="You are a helpful assistant.",
    model=groq("llama-3.3-70b-versatile"),
    tracing=True
)

print("="*80)
print("STREAM_OBSERVE() - RICH STREAMING WITH METADATA")
print("="*80 + "\n")

# Example 1: Basic observation with status updates
print("Example 1: Simple observation with status updates\n")

for update in agent.stream_observe("What is artificial intelligence in 2 sentences?"):
    if update.is_agent_start:
        print(f"[START] Agent '{update.agent}' starting...")
        print(f"        Input: {update.metadata.get('input', 'N/A')[:50]}...\n")
        print("        Response: ", end="", flush=True)

    elif update.is_token:
        print(update.content, end="", flush=True)
        sys.stdout.flush()

    elif update.is_agent_end:
        print(f"\n\n[DONE] Agent '{update.agent}' finished!")
        print(f"       Duration: {update.duration:.3f}s")
        print(f"       Tokens: {update.tokens}")
        print(f"       Cost: ${update.cost:.6f}")

print("\n" + "-"*80 + "\n")

# Example 2: Real-time cost tracking
print("Example 2: Real-time cost tracking\n")

total_cost = 0.0
total_tokens = 0

for update in agent.stream_observe("Explain quantum computing in 3 short paragraphs."):
    if update.is_agent_start:
        print(f"[START] Starting: {update.agent}")
        print("\n", end="")

    elif update.is_token:
        print(update.content, end="", flush=True)

    elif update.is_agent_end:
        total_cost += update.cost or 0.0
        total_tokens += update.tokens or 0

        print(f"\n\n[DONE] Complete!")
        print(f"       This query: {update.tokens} tokens, ${update.cost:.6f}")
        print(f"       Running total: {total_tokens} tokens, ${total_cost:.6f}")

print("\n" + "-"*80 + "\n")

# Example 3: Progress bar style
print("Example 3: Custom progress display\n")

query = "List 5 programming languages and their use cases"
print(f"Query: {query}\n")

response_text = ""
for update in agent.stream_observe(query):
    if update.is_agent_start:
        print("#", end="", flush=True)  # Progress indicator

    elif update.is_token:
        response_text += update.content
        # Print a progress indicator every 10 characters
        if len(response_text) % 10 == 0:
            print("#", end="", flush=True)

    elif update.is_agent_end:
        print(" DONE!\n")
        print(f"Response:\n{response_text}\n")
        print(f"Stats: {update.tokens} tokens in {update.duration:.2f}s")
        print(f"Cost: ${update.cost:.6f}")

print("\n" + "="*80)

# Final summary
tracer.print_summary()

print("\n" + "="*80)
print("STREAM_OBSERVE() vs STREAM() COMPARISON")
print("="*80)
print("""
STREAM() - Simple text chunks:
    for chunk in agent.stream("query"):
        print(chunk, end="")

    + Simple API
    + Just text chunks
    - No metadata
    - No timing/cost info

STREAM_OBSERVE() - Rich updates with metadata:
    for update in agent.stream_observe("query"):
        if update.is_agent_start:
            print(f"Starting {update.agent}...")
        elif update.is_token:
            print(update.content, end="")
        elif update.is_agent_end:
            print(f"Done! {update.tokens} tokens, ${update.cost:.6f}")

    + Rich metadata (tokens, cost, duration)
    + Event-based (start, token, end)
    + Perfect for dashboards/UIs
    + Real-time cost tracking
    + Works with tracing

Use stream() for simple chatbots.
Use stream_observe() for production apps with monitoring.
""")
