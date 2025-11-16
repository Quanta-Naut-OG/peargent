# Peargent Examples

This directory contains examples demonstrating all features of Peargent. Examples are organized by topic for easy navigation.

## 📚 Quick Links

- [Getting Started](#getting-started) - Basic agent creation and usage
- [Tools](#tools) - Creating and using tools with agents
- [Agent Pools](#agent-pools) - Multi-agent orchestration
- [History](#history) - Conversation memory and persistence
- [Tracing](#tracing) - Observability and monitoring
- [Streaming](#streaming) - Real-time response streaming
- [Structured Output](#structured-output) - Type-safe responses with Pydantic
- [Advanced](#advanced) - Complex patterns and use cases

---

## Getting Started

**Location:** `01-getting-started/`

| File | Description | Key Features |
|------|-------------|--------------|
| `quickstart.py` | Complete overview of all Peargent features | All-in-one demo with 10 tests |
| `basic_agent.py` | Your first agent - simple Q&A | Agent creation, basic run() |

**Start here if you're new to Peargent!**

---

## Tools

**Location:** `02-tools/`

Tools allow agents to interact with external systems and perform actions.

| File | Description | Key Features |
|------|-------------|--------------|
| `basic_tools.py` | Create and use simple tools | create_tool(), weather & calculator |
| `parallel_tools.py` | Execute multiple tools simultaneously | Parallel execution, performance gains |
| `tool_timeout.py` | Handle slow tools with timeouts | Timeout configuration, fallbacks |
| `tool_retry.py` | Automatic retry on tool failures | Retry logic, exponential backoff |
| `tool_error_handling.py` | Graceful error handling | Try/catch, error messages |
| `tool_validation.py` | Validate tool inputs and outputs | Pydantic validation, type safety |

---

## Agent Pools

**Location:** `03-agent-pools/`

Run multiple agents together for complex tasks.

| File | Description | Key Features |
|------|-------------|--------------|
| `basic_pool.py` | Multiple agents working together | create_pool(), multi-agent workflow |

---

## History

**Location:** `04-history/`

Give your agents memory across conversations.

| File | Description | Key Features |
|------|-------------|--------------|
| `basic_history.py` | Conversation memory basics | create_history(), threads, messages |
| `sqlite_history.py` | Persist history to SQLite | File-based persistence, recovery |
| `postgresql_history.py` | Production history with PostgreSQL | Scalable storage, queries |
| `redis_history.py` | Fast history with Redis | In-memory cache, performance |

---

## Tracing

**Location:** `05-tracing/`

Monitor and debug your agents with comprehensive tracing.

| File | Description | Key Features |
|------|-------------|--------------|
| `sqlite_tracing.py` | Trace to SQLite database | enable_tracing(), local storage |
| `postgres_tracing.py` | Trace to PostgreSQL | Production tracing, scalability |
| `custom_tables.py` | Custom database schema | Table customization, queries |
| `advanced_tracing.py` | Advanced tracing patterns | Spans, metrics, filtering |
| `cost_tracking.py` | Track LLM costs and tokens | Cost analysis, budgets |
| `metrics.py` | Aggregate metrics and analytics | Statistics, reporting |

---

## Streaming

**Location:** `06-streaming/`

Stream responses in real-time for better UX.

| File | Description | Key Features |
|------|-------------|--------------|
| `basic_streaming.py` | Stream text responses | agent.stream(), real-time output |
| `async_streaming.py` | Async streaming with asyncio | Async/await, concurrent streams |
| `streaming_with_tracing.py` | Combine streaming + tracing | Observability for streams |

---

## Structured Output

**Location:** `07-structured-output/`

Get type-safe, validated responses using Pydantic models.

| File | Description | Key Features |
|------|-------------|--------------|
| `basic_structured.py` | Simple Pydantic schemas | output_schema, validation |
| `nested_structured.py` | Complex nested models | Lists, nested objects |
| `advanced_structured.py` | Advanced patterns | Unions, optionals, custom validators |
| `comparison.py` | Structured vs unstructured | Performance, reliability |

---

## Advanced

**Location:** `08-advanced/`

Complex patterns and real-world use cases.

| File | Description | Key Features |
|------|-------------|--------------|
| `routing_agent.py` | Route requests to specialized agents | Dynamic routing, delegation |
| `research_expert.py` | Build a research assistant | Multi-tool, complex workflows |
| `story_generator.py` | Creative writing agent | Long-form generation, creativity |

---

## Running Examples

All examples are standalone and can be run directly:

```bash
# Install Peargent first
pip install peargent

# Set up your API keys in .env
GROQ_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here  # optional
ANTHROPIC_API_KEY=your_key_here  # optional

# Run any example
python examples/01-getting-started/quickstart.py
python examples/02-tools/basic_tools.py
python examples/04-history/sqlite_history.py
```

---

## Example Template

Want to create your own example? Use this template:

```python
"""
Brief Description

This example demonstrates [feature]. It shows how to [key capability].
"""

from peargent import create_agent
from peargent.models import groq

# Your example code here
agent = create_agent(
    name="ExampleAgent",
    persona="Your persona here",
    model=groq("llama-3.3-70b-versatile")
)

result = agent.run("Your question here")
print(result)
```

---

## Need Help?

- **Documentation:** [Link to docs]
- **GitHub Issues:** https://github.com/Quanta-Naut/peargent/issues
- **Discussions:** https://github.com/Quanta-Naut/peargent/discussions

---

## Contributing Examples

Have a great example to share? We'd love to include it!

1. Create your example following the template
2. Add clear comments explaining what it does
3. Test it works with the latest Peargent version
4. Submit a PR with description of what it demonstrates

Check out [CONTRIBUTING.md](../CONTRIBUTING.md) for more details.
