import os
import re
from pathlib import Path

from google.adk.tools import ToolContext

PRODUCTS = {
    "SR-202": {
        "product_id": "SR-202",
        "name": "Dawn Blend",
        "type": "Medium roast, Philippine coffee beans",
        "weight_grams": 250,
        "price_php": 480,
        "tasting_notes": "Chocolate, roasted nuts, low acidity",
        "audience": "Everyday home brewers",
    },
    "SR-101": {
        "product_id": "SR-101",
        "name": "Dark Roast",
        "type": "Dark roast, Liberica beans",
        "weight_grams": 250,
        "price_php": 520,
        "tasting_notes": "Dark chocolate, smoky, full-bodied",
        "audience": "Bold coffee lovers",
    },
    "SR-303": {
        "product_id": "SR-303",
        "name": "Office Brew",
        "type": "Medium-dark roast, Arabica-Robusta blend",
        "weight_grams": 250,
        "price_php": 450,
        "tasting_notes": "Caramel, cocoa, citrus finish",
        "audience": "Office workers and team pantries",
    },
}

BRAND_VOICE = {
    "tone": "Warm, upbeat, and plain-spoken.",
    "language": "Light Filipino or local English tone.",
    "opening": "Lead with how the coffee tastes or when it is enjoyed.",
    "sentence_style": "Use short sentences.",
    "call_to_action": "End with one clear call to action.",
}

AUDIENCE_NOTES = {
    "busy professionals and students": [
        "Smooth",
        "Naturally sweet",
        "No added sugar",
    ]
}


def get_product_info(product_id: str) -> dict:
    """Look up a product by its product code or product name.

    Args:
        product_id: The product code or name, such as "SR-202" or
            "Dawn Blend".

    Returns:
        A dict with a status and the product details, or an error status
        listing the valid product codes when the product is not found.
    """
    query = product_id.strip()
    if not query:
        return {
            "status": "error",
            "message": "Please provide a product code or name.",
        }

    product = PRODUCTS.get(query.upper())
    if product is None:
        query_lower = query.casefold()
        product = next(
            (
                item
                for item in PRODUCTS.values()
                if item["name"].casefold() == query_lower
            ),
            None,
        )

    if product is None:
        valid_codes = ", ".join(PRODUCTS)
        return {
            "status": "error",
            "message": (
                f"Product '{query}' was not found. "
                f"Valid product codes: {valid_codes}."
            ),
        }

    return {"status": "success", "product": product}


def get_brand_voice() -> dict:
    """Return the Sunrise Roasters brand voice and writing guidelines.

    Returns:
        A dict with the brand tone, language, sentence style, call-to-action
        guidance, and notes for specific audiences.
    """
    return {
        "status": "success",
        "brand": "Sunrise Roasters",
        "guidelines": BRAND_VOICE,
        "audience_notes": AUDIENCE_NOTES,
    }


TRENDS = {
    "coffee": [
        "Local single-origin beans are a talking point on social media",
        "Short video 'brew with me' content gets strong engagement",
    ],
    "default": "No trend data for this category. Say so instead of guessing.",
}

COMPETITORS = {
    "brewlandia": "Leans on discounts and buy-one-take-one promos.",
    "kapehan co": "Leans on heritage and farmer stories. Premium pricing.",
}


def get_trending_topics(category: str) -> dict:
    """Returns made-up marketing trends for a product category."""
    return {
        "category": category,
        "trends": TRENDS.get(category.lower(), TRENDS["default"]),
    }


def get_competitor_messaging(competitor: str) -> dict:
    """Returns how a named competitor markets itself (made-up data).

    Args:
        competitor: The competitor name, for example 'Brewlandia'.

    Returns:
        A dict with a one-line summary of the competitor's messaging.
    """
    summary = COMPETITORS.get(competitor.strip().lower())
    if summary is None:
        return {
            "status": "error",
            "message": f"No data for competitor '{competitor}'",
        }
    return {
        "status": "success",
        "competitor": competitor,
        "messaging": summary,
    }


def save_campaign(
    campaign_name: str,
    content: str,
    tool_context: ToolContext,
) -> dict:
    """Saves a finished campaign to a text file in the output folder.

    Args:
        campaign_name: A short name, for example "Cold Brew Launch".
        content: The full campaign text to save.

    Returns:
        A dict with the status and the file path.
    """
    try:
        configured_dir = Path(
            os.environ.get("CAMPAIGN_OUTPUT_DIR", "output")
        )
        output_dir = (
            configured_dir
            if configured_dir.is_absolute()
            else Path.cwd() / configured_dir
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        slug = (
            re.sub(r"[^a-z0-9]+", "-", campaign_name.lower()).strip("-")
            or "campaign"
        )
        path = output_dir / f"{slug}.txt"
        path.write_text(content, encoding="utf-8")
    except OSError as error:
        return {
            "status": "error",
            "message": f"Could not save the file: {error}",
        }

    saved = list(tool_context.state.get("saved_campaigns", []))
    saved.append(campaign_name)
    tool_context.state["saved_campaigns"] = saved
    return {"status": "success", "path": str(path)}


def remember_preference(
    name: str,
    value: str,
    tool_context: ToolContext,
) -> dict:
    """Remembers a user preference so it applies in future sessions.

    Args:
        name: The preference key, for example "preferred_tone".
        value: The user's preference, for example "casual and playful".
    """
    key = f"user:{name}"
    tool_context.state[key] = value
    return {
        "status": "success",
        "message": f"Remembered {name} as '{value}'",
    }
