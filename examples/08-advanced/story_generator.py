from peargent import (
    create_agent,
    create_pool,
    create_routing_agent,
    create_tool,
    limit_steps,
)
from peargent.models import gemini, groq

def run_story_pipeline(input: str):
    # Tool: Format final output
    def format_output_tool(story: str, prompt: str) -> str:
        return (
            f"📝 **Story:**\n{story}\n\n"
            f"🎨 **Image Prompt:**\n{prompt}"
        )

    format_output = create_tool(
        name="format_output",
        description="Formats the story and illustration prompt into a combined response",
        input_parameters={"story": str, "prompt": str},
        call_function=format_output_tool,
    )

    # Agents
    story_writer = create_agent(
        name="story_writer",
        description="Creative storyteller who crafts unique short stories based on prompts.",
        persona="You are a creative writer who weaves compelling and imaginative stories. Respond directly with your story - do not use any tools or JSON formatting.",
        tools=[],
        stop=limit_steps(2),
    )

    prompt_designer = create_agent(
        name="prompt_designer",
        description="Expert at crafting text prompts for image generation models based on textual input.",
        persona="You are a visual prompt designer who translates stories into detailed visual descriptions for image generation. Respond directly with your image prompt - do not use any tools or JSON formatting.",
        tools=[],
        stop=limit_steps(2),
    )

    synthesizer = create_agent(
        name="synthesizer",
        description="Responsible for combining the story and image prompt into a structured output.",
        persona="You specialize in formatting outputs. Use the format_output tool to deliver the final result.",
        tools=[format_output],
        stop=limit_steps(1),
    )

    # Router Agent
    router = create_routing_agent(
        name="creative_router",
        model=gemini("gemini-2.5-flash"),
        persona="""You are a smart routing coordinator for a creative pipeline.
        You first instruct the story_writer to create a story.
        Then instruct the prompt_designer to generate an illustration prompt based on the story.
        Finally, have the synthesizer package everything using the format_output tool.""",
        agents=["story_writer", "prompt_designer", "synthesizer"],
    )

    # Agent Pool
    pool = create_pool(
        agents=[story_writer, prompt_designer, synthesizer],
        default_model=groq("llama-3.1-8b-instant"),
        router=router,
        max_iter=5,
    )

    answer = pool.run(input)
    return answer

if __name__ == "__main__":
    result = run_story_pipeline("Write a whimsical story about a lonely robot on Mars learning to make friends.")
    try:
        print(result)
    except UnicodeEncodeError:
        # Handle Windows console encoding issues with emojis
        print(result.encode('ascii', 'ignore').decode('ascii'))
