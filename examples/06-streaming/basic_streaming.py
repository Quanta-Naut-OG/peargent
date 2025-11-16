"""
Basic Streaming Example

Demonstrates simple text streaming from agents.
"""

import sys
from peargent import create_agent
from peargent.telemetry import enable_tracing
from peargent.models import groq

# Enable tracing
tracer = enable_tracing()

# Create agent
agent = create_agent(
    name="StreamingAgent",
    description="An agent that streams responses",
    persona="You are a helpful assistant. Be concise but informative.",
    model=groq("llama-3.3-70b-versatile"),
    tracing=True
)

print("="*80)
print("BASIC STREAMING DEMO")
print("="*80 + "\n")

# Example 1: Simple streaming
print("Example 1: Simple question")
print("Q: What is Python?")
print("A: ", end="", flush=True)

for chunk in agent.stream("What is Python in 2 sentences?"):
    print(chunk, end="", flush=True)
    sys.stdout.flush()

print("\n\n" + "-"*80 + "\n")

# Example 2: Longer response
print("Example 2: Longer response")
print("Q: Explain machine learning")
print("A: ", end="", flush=True)

for chunk in agent.stream("Explain machine learning in simple terms. Use 2 short paragraphs."):
    print(chunk, end="", flush=True)
    sys.stdout.flush()

print("\n\n" + "-"*80 + "\n")

# Example 3: Multiple queries
queries = [
    "What is 5+5?",
    "What is the capital of France?",
    "What color is the sky?"
]

print("Example 3: Multiple quick queries\n")
for i, query in enumerate(queries, 1):
    print(f"{i}. Q: {query}")
    print(f"   A: ", end="", flush=True)

    for chunk in agent.stream(query):
        print(chunk, end="", flush=True)
        sys.stdout.flush()

    print()

print("\n" + "="*80)

# Check traces
traces = tracer.list_traces()
print(f"\nTraces captured: {len(traces)}")
tracer.print_summary()

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("""
stream() - Simple text streaming

Usage:
    for chunk in agent.stream("question"):
        print(chunk, end="", flush=True)

Benefits:
- Immediate feedback to users
- Lower perceived latency
- Better UX for long responses
- Works automatically with tracing

Perfect for:
- Chatbots and conversational AI
- Real-time assistants
- Interactive applications
- Any user-facing application
""")
