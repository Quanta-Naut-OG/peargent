"""
Pool Streaming with stream_observe()

Demonstrates observing multi-agent workflows with rich updates.
"""

import sys
from peargent import create_agent, create_pool, enable_tracing
from peargent.models import groq
from peargent.core.streaming import UpdateType

# Enable tracing
tracer = enable_tracing()

# Create specialized agents
researcher = create_agent(
    name="Researcher",
    description="Research specialist",
    persona="You are a research specialist. Provide factual, concise information.",
    model=groq("llama-3.3-70b-versatile"),
    tracing=True
)

writer = create_agent(
    name="Writer",
    description="Content writer",
    persona="You are a professional writer. Make content clear and engaging.",
    model=groq("llama-3.3-70b-versatile"),
    tracing=True
)

reviewer = create_agent(
    name="Reviewer",
    description="Content reviewer",
    persona="You are a reviewer. Provide final polish and make improvements brief. You add lot of emojis",
    model=groq("llama-3.3-70b-versatile"),
    tracing=True
)

# Create pool
def sequential_router(state, call_count, last_result):
    from peargent.core.router import RouterResult
    agents = ["Researcher", "Writer", "Reviewer"]
    if call_count < len(agents):
        return RouterResult(agents[call_count])
    return RouterResult(None)

pool = create_pool(
    agents=[researcher, writer, reviewer],
    router=sequential_router,
    max_iter=3,
    tracing=True
)

print("="*80)
print("POOL STREAMING WITH STREAM_OBSERVE()")
print("="*80 + "\n")

print("Query: Explain what blockchain is\n")
print("="*80 + "\n")

# Track pool metrics
pool_tokens = 0
pool_cost = 0.0
current_agent = None
for update in pool.stream("Explain what blockchain is in simple terms"):
    # if update.type == UpdateType.POOL_START:
    #     print("🎬 Pool execution starting...")
    #     print(f"   Max iterations: {update.metadata.get('max_iter')}\n")

    # elif update.is_agent_start:
    #     current_agent = update.agent
    #     print(f"\n{'='*80}")
    #     print(f"🔵 [{update.agent}] Starting...")
    #     print(f"{'='*80}")
    #     print("\n", end="")

    # elif update.is_token:
    print(update, end="", flush=True)
    sys.stdout.flush()

    # elif update.is_agent_end and update.agent == "Researcher":
    #     pool_tokens += update.tokens or 0
    #     pool_cost += update.cost or 0.0

    #     print(f"\n\n✅ [{update.agent}] Complete")
    #     print(f"   Duration: {update.duration:.3f}s")
    #     print(f"   Tokens: {update.tokens}")
    #     print(f"   Cost: ${update.cost:.6f}")

    # elif update.type == UpdateType.POOL_END:
    #     print(f"\n{'='*80}")
    #     print("🎉 Pool execution complete!")
    #     print(f"{'='*80}")
    #     print(f"\n   Total Duration: {update.metadata.get('duration', 0):.3f}s")
    #     print(f"   Total Agents: {update.metadata.get('agents_executed', 0)}")
    #     print(f"   Total Tokens: {update.metadata.get('total_tokens', 0)}")
    #     print(f"   Total Cost: ${update.metadata.get('total_cost', 0.0):.6f}")

print("\n" + "="*80 + "\n")

# Verify with tracer
print("Verification from tracer:")
tracer.print_summary()

print("\n" + "="*80)
print("POOL STREAM_OBSERVE() - KEY FEATURES")
print("="*80)
print("""
1. FULL VISIBILITY
   - See every agent's execution
   - Real-time progress updates
   - Complete transparency

2. PER-AGENT METRICS
   - Individual tokens, cost, duration
   - Track which agents are expensive
   - Optimize specific agents

3. POOL-LEVEL METRICS
   - Total cost across all agents
   - Total tokens used
   - End-to-end duration
   - Number of agents executed

4. EVENT TYPES
   - POOL_START: Pool begins
   - AGENT_START: Agent begins
   - TOKEN: Text chunk from agent
   - AGENT_END: Agent finishes (with metadata)
   - POOL_END: Pool completes (with totals)

5. PERFECT FOR
   - Production monitoring
   - Cost optimization
   - Debugging workflows
   - Building dashboards
   - User progress indicators

SIMPLE EXAMPLE:
    for update in pool.stream_observe("query"):
        if update.is_agent_start:
            show_loading(update.agent)
        elif update.is_token:
            append_text(update.content)
        elif update.is_agent_end:
            log_metrics(update.tokens, update.cost)
        elif update.type == UpdateType.POOL_END:
            show_complete(update.metadata)
""")
