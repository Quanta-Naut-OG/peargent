"""
Tests for the Agent class and create_agent functionality.
Tests basic agent creation, configuration, and tool integration.
"""

from typing import Any, Dict

import pytest

from peargent import create_agent, create_tool


class MockModel:
    """Mock LLM model for testing without API keys."""

    def __init__(self, model_name: str = "mock-model", **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs

    def generate(self, prompt: str, **kwargs) -> str:
        return "Mock response"


class TestAgentCreation:
    """Test agent creation and initialization."""

    def test_create_basic_agent(self) -> None:
        """Test creating a basic agent with minimal parameters."""
        model = MockModel()
        agent = create_agent(
            name="test-agent",
            description="A test agent",
            persona="You are a helpful assistant",
            model=model,
        )

        assert agent.name == "test-agent"
        assert agent.description == "A test agent"
        assert agent.model is not None

    def test_create_agent_without_tools(self) -> None:
        """Test creating an agent without any tools."""
        model = MockModel()
        agent = create_agent(
            name="simple-agent",
            description="A simple agent without tools",
            persona="You are a simple assistant",
            model=model,
        )

        assert agent.name == "simple-agent"
        assert len(agent.tools) == 0

    def test_agent_has_required_attributes(self) -> None:
        """Test that agent has all required attributes."""
        model = MockModel()
        agent = create_agent(
            name="test-agent",
            description="Test agent",
            persona="You are helpful",
            model=model,
        )

        assert hasattr(agent, "name")
        assert hasattr(agent, "description")
        assert hasattr(agent, "model")
        assert hasattr(agent, "tools")


class TestAgentWithTools:
    """Test agent creation with tools."""

    def test_create_agent_with_single_tool(self) -> None:
        """Test creating an agent with a single tool."""

        def add(a: int, b: int) -> int:
            return a + b

        tool = create_tool(
            name="add",
            description="Add two numbers",
            input_parameters={"a": int, "b": int},
            call_function=add,
        )

        model = MockModel()
        agent = create_agent(
            name="calculator-agent",
            description="An agent with calculator tool",
            persona="You are a calculator assistant",
            model=model,
            tools=[tool],
        )

        assert agent.name == "calculator-agent"
        assert len(agent.tools) == 1
        assert "add" in agent.tools
        assert agent.tools["add"].name == "add"

    def test_create_agent_with_multiple_tools(self) -> None:
        """Test creating an agent with multiple tools."""

        def add(a: int, b: int) -> int:
            return a + b

        def subtract(a: int, b: int) -> int:
            return a - b

        def multiply(a: int, b: int) -> int:
            return a * b

        tools = [
            create_tool(
                name="add",
                description="Add numbers",
                input_parameters={"a": int, "b": int},
                call_function=add,
            ),
            create_tool(
                name="subtract",
                description="Subtract numbers",
                input_parameters={"a": int, "b": int},
                call_function=subtract,
            ),
            create_tool(
                name="multiply",
                description="Multiply numbers",
                input_parameters={"a": int, "b": int},
                call_function=multiply,
            ),
        ]

        model = MockModel()
        agent = create_agent(
            name="math-agent",
            description="A mathematical agent",
            persona="You are a math assistant",
            model=model,
            tools=tools,
        )

        assert len(agent.tools) == 3
        assert "add" in agent.tools
        assert "subtract" in agent.tools
        assert "multiply" in agent.tools

    def test_agent_tool_parameter_types(self) -> None:
        """Test that agent correctly stores tool parameter types."""

        def sample_func(x: int, y: str) -> str:
            return f"{y}: {x}"

        tool = create_tool(
            name="sample",
            description="A sample tool",
            input_parameters={"x": int, "y": str},
            call_function=sample_func,
        )

        model = MockModel()
        agent = create_agent(
            name="schema-agent",
            description="Agent for testing tool parameters",
            persona="You are a test assistant",
            model=model,
            tools=[tool],
        )

        assert len(agent.tools) == 1
        assert "sample" in agent.tools
        assert agent.tools["sample"].input_parameters["x"] is int
        assert agent.tools["sample"].input_parameters["y"] is str


class TestAgentConfiguration:
    """Test agent configuration options."""

    def test_agent_with_max_retries(self) -> None:
        """Test agent with custom max_retries."""
        model = MockModel()
        agent = create_agent(
            name="retry-agent",
            description="Agent with retries",
            persona="You are helpful",
            model=model,
            max_retries=5,
        )

        assert agent.max_retries == 5

    def test_agent_with_tracing_enabled(self) -> None:
        """Test agent with tracing enabled."""
        model = MockModel()
        agent = create_agent(
            name="traced-agent",
            description="Agent with tracing",
            persona="You are helpful",
            model=model,
            tracing=True,
        )

        assert agent.tracing is True

    def test_agent_with_tracing_disabled(self) -> None:
        """Test agent with tracing disabled."""
        model = MockModel()
        agent = create_agent(
            name="untraced-agent",
            description="Agent without tracing",
            persona="You are helpful",
            model=model,
            tracing=False,
        )

        assert agent.tracing is False


class TestAgentModel:
    """Test agent model configuration."""

    def test_agent_with_different_models(self) -> None:
        """Test creating agents with different mock models."""
        model1 = MockModel("gpt-4")
        model2 = MockModel("claude-3")

        agent1 = create_agent(
            name="gpt-agent",
            description="Agent with GPT",
            persona="You are helpful",
            model=model1,
        )

        agent2 = create_agent(
            name="claude-agent",
            description="Agent with Claude",
            persona="You are helpful",
            model=model2,
        )

        assert agent1.model.model_name == "gpt-4"
        assert agent2.model.model_name == "claude-3"

    def test_create_multiple_independent_agents(self) -> None:
        """Test creating multiple independent agents."""
        model = MockModel()

        agent1 = create_agent(
            name="agent1",
            description="First agent",
            persona="You are helpful",
            model=model,
        )

        agent2 = create_agent(
            name="agent2",
            description="Second agent",
            persona="You are helpful",
            model=model,
        )

        assert agent1.name == "agent1"
        assert agent2.name == "agent2"
        assert agent1.name != agent2.name
