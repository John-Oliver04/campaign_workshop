import os
import time

from google import genai
from google.adk.agents import Agent
from google.adk.tools import ToolContext
from google.genai import types

from ..config import MODEL

# The image model is separate from the text model above. Change it here if needed.
IMAGE_MODEL = os.environ.get("WORKSHOP_IMAGE_MODEL", "gemini-3.1-flash-lite-image")


async def generate_campaign_image(
    image_prompt: str, tool_context: ToolContext
) -> dict:
    """Generates one campaign image and saves it as an artifact.

    A dict with status "success" and the saved file name, or status
    "error" and a message.
    """
    try:
        client = genai.Client()
        response = client.models.generate_content(
            model=IMAGE_MODEL,
            contents=image_prompt,
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"]
            ),
        )
        candidates = response.candidates or []
        if not candidates or candidates[0].content is None:
            return {
                "status": "error",
                "message": "The image model returned no image.",
            }
        for part in candidates[0].content.parts or []:
            if part.inline_data and part.inline_data.data:
                filename = f"campaign_image_{int(time.time())}.png"
                await tool_context.save_artifact(
                    filename,
                    types.Part.from_bytes(
                        data=part.inline_data.data,
                        mime_type=part.inline_data.mime_type or "image/png",
                    ),
                )
                return {"status": "success", "file": filename}
        return {"status": "error", "message": "The image model returned no image."}
    except Exception as e:
        return {"status": "error", "message": f"Image generation failed: {e}"}


illustrator = Agent(
    name="illustrator",
    model=MODEL,
    description="Generates visual campaign assets using image generation tools.",
    instruction="""You are an illustrator for the brand.
Given a visual brief, write one detailed image prompt from the hero image idea, the colour
palette and the mood. Show the coffee or the moment of enjoying it. Do not put text, logos,
or people's faces in the image. Call generate_campaign_image once with that prompt.
If it returns an error, say so and give the prompt you tried. Never say an image was
made unless the tool returned success.""",
    tools=[generate_campaign_image],
)
