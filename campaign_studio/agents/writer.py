from google.adk.agents import Agent

from ..config import MODEL
from ..tools import get_brand_voice, get_product_info

writer = Agent(
    name="writer",
    model=MODEL,
    description="Writes campaign copy for a Sunrise Roasters product.",
    instruction="""You are the copywriter for Sunrise Roasters.
Always call get_product_info for the product and get_brand_voice before writing.""",
    tools=[get_product_info, get_brand_voice],
)
