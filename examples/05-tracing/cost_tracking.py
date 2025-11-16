"""
Cost Tracking Test

This tests that costs are being calculated correctly for different models.
"""

from peargent import create_agent
from peargent.telemetry import enable_tracing
from peargent.models import groq

# Enable tracing
tracer = enable_tracing()

# Create agent with Groq model
agent = create_agent(
    name="CostTestAgent",
    description="Agent for testing cost tracking",
    persona="You are helpful. Give very short answers.",
    model=groq("llama-3.3-70b-versatile"),
    tracing=True
)

print("="*80)
print("COST TRACKING TEST")
print("="*80 + "\n")

# Run the agent with a simple query
result = agent.run("What is 2+2? Answer in one sentence.")
print(f"Result: {result}\n")

# Get the trace
traces = tracer.list_traces()
if traces:
    trace = traces[0]

    print("Trace Details:")
    print(f"  Model: {trace.spans[0].model if trace.spans else 'N/A'}")
    print(f"  Prompt Tokens: {trace.spans[0].token_prompt if trace.spans else 'N/A'}")
    print(f"  Completion Tokens: {trace.spans[0].token_completion if trace.spans else 'N/A'}")
    print(f"  Total Tokens: {trace.total_tokens}")
    print(f"  Span Cost: ${trace.spans[0].cost:.6f}" if trace.spans and trace.spans[0].cost else "  Span Cost: $0.000000")
    print(f"  Total Cost: ${trace.total_cost:.6f}")
    print()

    # Show pricing calculation
    if trace.spans and trace.spans[0].token_prompt and trace.spans[0].token_completion:
        prompt_tokens = trace.spans[0].token_prompt
        completion_tokens = trace.spans[0].token_completion

        print("Cost Calculation Breakdown:")
        print(f"  llama-3.3-70b pricing:")
        print(f"    Prompt: $0.59 per million tokens")
        print(f"    Completion: $0.79 per million tokens")
        print()
        print(f"  This query:")
        print(f"    Prompt tokens: {prompt_tokens}")
        print(f"    Completion tokens: {completion_tokens}")
        print(f"    Prompt cost: ({prompt_tokens} / 1,000,000) * $0.59 = ${(prompt_tokens / 1_000_000) * 0.59:.6f}")
        print(f"    Completion cost: ({completion_tokens} / 1,000,000) * $0.79 = ${(completion_tokens / 1_000_000) * 0.79:.6f}")
        print(f"    Total cost: ${trace.total_cost:.6f}")

# Print summary
print("\n" + "="*80)
tracer.print_summary()

print("\n" + "="*80)
print("EXPECTED BEHAVIOR")
print("="*80)
print("""
For Groq llama-3.3-70b-versatile:
- Model name should be detected as "llama-3.3-70b-versatile"
- Pricing should match "llama-3.3-70b" ($0.59 prompt, $0.79 completion per million tokens)
- Cost should be > $0.000000 for any real query

If cost is still $0.000000:
- Check that model.model_name attribute is being read correctly
- Check that pricing table includes the model
- Check cost calculation logic
""")
