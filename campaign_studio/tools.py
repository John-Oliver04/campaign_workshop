import os
from datetime import datetime
from pathlib import Path

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


def save_campaign(campaign: str, filename: str = "") -> dict:
    """Save generated campaign copy to a text file in the output folder.

    Args:
        campaign: The complete campaign copy to save.
        filename: Optional file name, such as "dawn-blend.txt". Defaults to a
            timestamped file name.

    Returns:
        A dict with a success status and the saved file path, or an error
        status with a message.
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
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if filename.strip():
            name = Path(filename.strip()).name
            name = "".join(
                char if char.isalnum() or char in "-_." else "_"
                for char in name
            )
            if not name.lower().endswith(".txt"):
                name += ".txt"
        else:
            name = f"campaign_{stamp}.txt"

        if name in {"", ".txt"}:
            name = f"campaign_{stamp}.txt"

        path = output_dir / name
        path.write_text(campaign, encoding="utf-8")
        return {"status": "success", "path": str(path)}
    except OSError as exc:
        return {
            "status": "error",
            "message": f"Could not save campaign: {exc}",
        }
