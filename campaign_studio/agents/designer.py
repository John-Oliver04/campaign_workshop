from google.adk.agents import Agent

from ..config import MODEL

designer = Agent(
    name="designer",
    model=MODEL,
    description="Writes a short visual brief a designer can follow.",
    instruction="""You are an art director. Given a campaign angle and product, write a
visual brief with: a colour palette (3 colours), the mood in 3 words, one hero image
idea and typography advice. Keep it under 100 words. Do not claim to have made any
images.""",
)
