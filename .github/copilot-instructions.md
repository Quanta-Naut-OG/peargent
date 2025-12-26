# Peargent Python Agent Framework - Code Review Guidelines

## Purpose & Scope

Guidelines for reviewing Peargent Python framework PRs. PyPI package (MIT), Python 3.9-3.12, ~41 source files for AI agents with multi-LLM support.

---

## Review Comment Format

**Use collapsible sections for long review comments to keep PRs readable:**

```markdown
<details>
<summary>⚠️ Issue: Missing type hints</summary>

The function `calculate_result` needs type hints for parameters and return value.

**Suggested fix:**
\`\`\`python
def calculate_result(x: int, y: int) -> int:
    return x + y
\`\`\`
</details>
```

---

## Critical Build Requirements

### Environment Setup - ALWAYS REQUIRED FIRST

```bash
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/macOS
pip install -e .
```

### Validation Commands

```bash
# Verify imports
python -c "from peargent import create_agent, create_tool, create_pool; print('OK')"

# Run tests - MUST use python -m pytest (not just pytest)
python -m pytest tests/

# Format & lint
black peargent/
flake8 peargent/

# Build
python -m build
twine check dist/*
```

**Known Issues:**
- Must use `python -m pytest`, not `pytest` (import errors otherwise)
- Tests requiring API keys will skip/fail without `.env` config (expected)
- License deprecation warnings in build are non-blocking (safe to ignore)

---

## CI Pipeline (PR Requirements)

`.github/workflows/pr-test.yml` runs on every PR:
- Python 3.9, 3.10, 3.11, 3.12 matrix test
- Import validation, package build, distribution check
- **All versions must pass for merge**

---

## Code Style Rules

- **PEP 8 compliance** (Black + Flake8 required)
- **Type hints** on all function signatures
- **Docstrings** for public APIs
- **Conventional commits**: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`

### Import Pattern (CRITICAL)

```python
# ✅ CORRECT - Use factory functions
from peargent import create_agent, create_tool, create_pool
from peargent.models import groq, anthropic, openai

# ❌ WRONG - Never import internal classes
from peargent._core.agent import Agent  # Internal class
from peargent._core.tool import Tool    # Internal class
```

### Example Code Patterns

**Tool definition:**
```python
@create_tool(description="Calculate expression")
def calculator(expression: str) -> str:
    return str(eval(expression))
```

**Agent creation:**
```python
agent = create_agent(
    name="assistant",
    persona="You are helpful.",
    model=groq("llama-3.3-70b-versatile"),
    tools=[calculator]
)
```

---

## Project Structure

**Key files:**
- `peargent/__init__.py` (577 lines) - Public API exports
- `peargent/_core/agent.py` (1160 lines) - Core agent logic
- `peargent/_core/tool.py` - Tool system & validation
- `peargent/models/*.py` - LLM provider adapters (OpenAI, Anthropic, Groq, Gemini, Azure)
- `pyproject.toml` - Build config, dependencies, version 0.1.4

**Tests:** `tests/test_tool.py` (8 tests), `test_smoke.py`, `test_persona.py`, `test_anthropic.py`

---

## Common Review Scenarios

### Adding New Tool
1. Create in `peargent/tools/`
2. Export from `peargent/tools/__init__.py`
3. Register in `get_tool_by_name()` if built-in
4. Add tests to `tests/test_tool.py`
5. Add example to `examples/02-tools/`

### Adding New Model Provider
1. Create `peargent/models/newprovider.py` following `base.py` interface
2. Export from `peargent/models/__init__.py`
3. Add example to `examples/01-getting-started/`
4. Update README.md

---

## Review Checklist

Before approving:
- [ ] Venv activated & `pip install -e .` run
- [ ] `black peargent/` formatted
- [ ] `flake8 peargent/` passes
- [ ] `python -m pytest tests/` passes
- [ ] Imports work: `python -c "from peargent import create_agent, create_tool, create_pool"`
- [ ] `python -m build && twine check dist/*` succeeds
- [ ] Conventional commit message
- [ ] Type hints added
- [ ] Docstrings for public APIs
