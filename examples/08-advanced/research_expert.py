from peargent import (
    create_agent,
    create_pool,
    create_routing_agent,
    create_tool,
    limit_steps,
)
from peargent.models import gemini, groq

def run_knowledge_pipeline(input: str):
    # Tool: Fetch data (Mocked API tool)
    def fetch_data_tool(query: str) -> str:
        return f"Fetched space-related data about: {query}. Sample data: 42, 58, 199."

    fetch_data = create_tool(
        name="fetch_data",
        description="Fetches data related to a query",
        input_parameters={"query": str},
        call_function=fetch_data_tool,
    )

    # Tool: Analyze data (Simple Analyzer)
    def analyze_data_tool(data: str) -> str:
        import re

        # Extract all numbers from the string (handles commas and other separators)
        numbers = [int(n) for n in re.findall(r"\d+", data)]
        if not numbers:
            return "No numerical data found to analyze."
        avg = sum(numbers) / len(numbers)
        return f"Data points: {numbers}. Average: {avg:.2f}, Sum: {sum(numbers)}, Count: {len(numbers)}"

    analyze_data = create_tool(
        name="analyze_data",
        description="Analyzes numerical data and finds the average",
        input_parameters={"data": str},
        call_function=analyze_data_tool,
    )

    # Tool: Format result (Mock Formatter)
    def format_result_tool(text: str) -> str:
        return f"{text}"

    format_result = create_tool(
        name="format_report",
        description="Formats text content into a report-style output",
        input_parameters={"text": str},
        call_function=format_result_tool,
    )

    # Agents
    researcher = create_agent(
        name="researcher",
        description="Data Research Specialist - Responsible for gathering and retrieving raw data from various sources using available data fetching tools. Focuses on comprehensive data collection without analysis.",
        persona="You are a meticulous data researcher who specializes in gathering comprehensive information. Your role is to fetch relevant data using available tools and present it in a clear, organized manner. You focus purely on data collection and do not perform analysis - that's for other specialists.",
        # model=groq("openai/gpt-oss-120b"),
        tools=[fetch_data],
        stop=limit_steps(3),
    )

    analyzer = create_agent(
        name="analyzer",
        description="Statistical Data Analyst - Specializes in processing and analyzing numerical data to extract meaningful insights. Performs calculations, identifies patterns, and generates statistical summaries from raw datasets.",
        persona="You are an expert data analyst with strong statistical and mathematical skills. Your job is to take raw data and transform it into meaningful insights through analysis. You calculate averages, identify trends, and provide statistical summaries. You work with precision and always explain your analytical methodology.",
        # model=groq("openai/gpt-oss-120b"),
        tools=[analyze_data],
        stop=limit_steps(2),
    )

    reporter = create_agent(
        name="reporter",
        description="Professional Report Writer - Expert in transforming analytical findings into polished, well-structured reports. Specializes in clear communication, professional formatting, and presenting complex data insights in an accessible manner.",
        persona="You are a professional report writer. When you receive analytical findings, use the format_report tool to create a formatted report, then present the EXACT formatted output from the tool as your final response. DO NOT just describe that you created a report - Describe the report. ",
        # model=groq("openai/gpt-oss-120b"),
        tools=[format_result],
        stop=limit_steps(3),
    )

    # Router Agent
    router = create_routing_agent(
        name="workflow_router",
        model=gemini("gemini-2.5-flash"),
        persona="""You are an intelligent workflow coordinator that routes tasks to the most appropriate specialist agent.

Your goal is to ensure the user's request is fulfilled by orchestrating the right sequence of agents.

Key principles:
- Avoid routing to the same agent twice in a row unless there's a clear need
- Stop when the user's original request has been completely satisfied
- Understand what's been done and create a clear plan to decide the next tool.

Analyze what's been done and what's still needed, then choose wisely.""",
        agents=["researcher", "analyzer", "reporter"],
    )

    # Agent Pool
    pool = create_pool(
        agents=[researcher, analyzer, reporter],
        default_model=groq("llama-3.3-70b-versatile"),
        router=router,
        max_iter=5,
    )

    answer = pool.run(input)
    return answer

if __name__ == "__main__":
    result = run_knowledge_pipeline("Give me an analytical report about space data")
    try:
        print(result)
    except UnicodeEncodeError:
        # Handle Windows console encoding issues with emojis
        print("Pipeline Result:", result.encode("ascii", "ignore").decode("ascii"))
