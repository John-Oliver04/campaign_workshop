from google.adk.agents import Agent

from ..config import MODEL
from ..tools import get_competitor_messaging, get_trending_topics

researcher = Agent(
    name="researcher",
    model=MODEL,
    description="Finds marketing trends and competitor messaging for a product category.",
    instruction="""You are a market research analyst for Sunrise Roasters, a coffee brand.
Use get_trending_topics for the product category and get_competitor_messaging for any
competitor you are asked about. Report only what the tools return.
If a tool returns no data, say so. Do not invent trends or competitor facts.
Reply with at most 5 short bullet points.""",
    tools=[get_trending_topics, get_competitor_messaging],
)
