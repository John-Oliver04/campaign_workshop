from google.adk.agents import Agent

from ..config import MODEL

strategist = Agent(
    name="strategist",
    model=MODEL,
    description="Turns research notes into a campaign angle, audience and channel plan.",
    instruction="""You are a campaign strategist.
You will be given research notes and a product. Produce:
1. Target audience
2. Core campaign angle
3. Recommended marketing channels""",
)
