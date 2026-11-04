"""The campaign pipeline: workflow agents that pass work along through session state."""

from google.adk.agents import Agent, LoopAgent, ParallelAgent, SequentialAgent
from google.adk.tools import exit_loop
from pydantic import BaseModel, Field

from .config import MODEL
from .tools import (
    get_brand_voice,
    get_competitor_messaging,
    get_product_info,
    get_trending_topics,
)

# --- Stage 1: Two researchers work at the same time ---

trend_scout = Agent(
    name="trend_scout",
    model=MODEL,
    description="Finds marketing trends for the product category.",
    instruction="""You research marketing trends for Sunrise Roasters, a coffee brand.
Call get_trending_topics for the product's category. Reply with at most 3 bullets,
using only what the tool returns.""",
    tools=[get_trending_topics],
    output_key="trend_notes",
)

competitor_analyst = Agent(
    name="competitor_analyst",
    model=MODEL,
    description="Summarises how competitors market themselves.",
    instruction="""You summarise competitor marketing for Sunrise Roasters.
Call get_competitor_messaging for each competitor named in the request.
If none is named, use Brewlandia and Kapehan Co. Report only what the tool returns.""",
    tools=[get_competitor_messaging],
    output_key="competitor_notes",
)

research_stage = ParallelAgent(
    name="research_stage",
    description="Runs trend research and competitor research at the same time.",
    sub_agents=[trend_scout, competitor_analyst],
)

# --- Stage 2: One strategist reads both sets of notes ---

campaign_strategist = Agent(
    name="campaign_strategist",
    model=MODEL,
    description="Turns research into a campaign strategy.",
    instruction="""You are a campaign strategist for Sunrise Roasters.
Trend notes: {trend_notes}
Competitor notes: {competitor_notes}
Write: one campaign angle (one sentence), the target audience (one sentence),
and 2 to 3 channels with one tactic each. Under 120 words.""",
    output_key="strategy",
)

# --- Stage 3: Writer and reviewer loop until approved ---

draft_writer = Agent(
    name="draft_writer",
    model=MODEL,
    description="Writes or rewrites the campaign copy.",
    instruction="""You are the copywriter for Sunrise Roasters.
Strategy: {strategy}
Reviewer Feedback from the last round (may be empty):
{review_feedback?}
Call get_product_info for the product and get_brand_voice, then write the copy:
a headline (8 words or fewer), a 2 to 3 sentence description, three benefit bullets
and one call to action. If there is feedback, fix every point in it.
Use only facts from the tools.""",
    tools=[get_product_info, get_brand_voice],
    output_key="draft",
)

reviewer = Agent(
    name="reviewer",
    model=MODEL,
    description="Checks the draft against the brand rules and approves it or asks for fixes.",
    instruction="""You review marketing copy for Sunrise Roasters.
Draft: {draft}
Check all of these:
1. Headline is 8 words or fewer
2. No health claims and no promises such as 'guaranteed' or 'miracle'
3. No competitor is named
4. Exactly one clear call to action

If every check passes, call exit_loop.
If any check fails, do not call exit_loop; reply with a bulleted list of fixes for the writer.""",
    tools=[exit_loop],
    output_key="review_feedback",
)

draft_review_loop = LoopAgent(
    name="draft_review_loop",
    description="Writes and reviews the copy until it passes or three rounds are used.",
    sub_agents=[draft_writer, reviewer],
    max_iterations=3,
)

# --- Stage 4: Visual brief ---

visual_designer = Agent(
    name="visual_designer",
    model=MODEL,
    description="Writes a visual brief for the campaign.",
    instruction="""You are an art director. Strategy: {strategy}
Write a visual brief: a 3-colour palette, the mood in 3 words, one hero image idea
and typography advice. Under 100 words. Do not claim to have made any images.""",
    output_key="visual_brief",
)

# --- Stage 5: Package everything in a fixed structure ---


class CampaignPackage(BaseModel):
    campaign_name: str = Field(description="Short campaign name")
    headline: str
    description: str
    benefits: list[str]
    call_to_action: str
    channel_plan: list[str]
    visual_brief: str


packager = Agent(
    name="packager",
    model=MODEL,
    description="Combines the strategy, copy and visual brief into one package.",
    instruction="""Combine the work below into the required structure. Do not add new claims.
Strategy: {strategy}
Copy: {draft}
Visual brief: {visual_brief}""",
    output_schema=CampaignPackage,
    output_key="package",
)

# --- Full Pipeline Definition ---

campaign_pipeline = SequentialAgent(
    name="campaign_pipeline",
    description="Executes the complete multi-stage campaign generation pipeline.",
    sub_agents=[
        research_stage,
        campaign_strategist,
        draft_review_loop,
        visual_designer,
        packager,
    ],
)
