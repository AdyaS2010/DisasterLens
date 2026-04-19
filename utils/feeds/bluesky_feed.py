# ──────────────────────────────────────────────────────────────────────────────
# DisasterLens AI
# Developer: Adya S.
# Email:     adyasastry@gmail.com
# ──────────────────────────────────────────────────────────────────────────────
# utils/feeds/bluesky_feed.py — Bluesky Live Feed Connector
#
# Fetches disaster-related posts from Bluesky via the AT Protocol (atproto).
# Uses a two-signal relevance filter (disaster type + impact/action keywords)
# and an exclusion list to ensure only actionable disaster content is returned.
# ──────────────────────────────────────────────────────────────────────────────

"""
DisasterLens AI — Bluesky Live Feed Connector
Fetches disaster-related posts from Bluesky via the AT Protocol.
"""

import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Focused queries — only actionable disaster situations
DISASTER_SEARCH_QUERIES = [
    "trapped flood rescue needed",
    "hurricane damage evacuation order",
    "earthquake collapsed trapped people",
    "wildfire evacuation mandatory homes",
    "tornado destroyed houses damage",
    "flash flood warning emergency",
    "storm surge flooding rescue",
    "earthquake aftershock damage report",
]

# ── Two-signal relevance system ──
# A post is relevant ONLY if it contains at least one DISASTER TYPE
# AND at least one IMPACT/ACTION term. This prevents matching poetry,
# TV shows, metaphors, and casual mentions.

DISASTER_TYPE_KEYWORDS = {
    "hurricane", "earthquake", "flood", "flooding", "wildfire",
    "tornado", "tsunami", "typhoon", "cyclone", "landslide",
    "mudslide", "avalanche", "volcanic", "eruption", "blizzard",
    "flash flood", "storm surge", "tropical storm", "seismic",
}

IMPACT_ACTION_KEYWORDS = {
    "trapped", "rescue", "evacuate", "evacuation", "evacuated",
    "damage", "damaged", "destroyed", "collapsed", "stranded",
    "missing", "casualties", "injured", "killed", "deaths",
    "shelter", "displaced", "debris", "rubble", "emergency",
    "warning", "mandatory", "first responders", "search and rescue",
    "power outage", "levee", "breach", "surge", "floodwater",
    "aftershock", "magnitude", "category", "winds mph",
    "making landfall", "struck", "hit by", "swept away",
    "under water", "submerged", "on fire", "burning",
    "aid", "fema", "red cross", "national guard",
}

# Reject posts containing any of these — strong signals of irrelevance
EXCLUSION_KEYWORDS = {
    "netflix", "hulu", "disney+", "movie", "film", "trailer",
    "album", "song", "lyrics", "playlist", "spotify",
    "poem", "poetry", "novel", "book review", "fiction",
    "game", "gaming", "xbox", "playstation", "nintendo",
    "crypto", "bitcoin", "nft", "stock market",
    "horoscope", "zodiac", "astrology",
    "recipe", "cooking", "restaurant",
    "metaphor", "figuratively", "feels like a",
    "tv show", "series premiere", "season finale",
    "box office", "streaming", "binge",
}


def _is_relevant(text: str) -> bool:
    """
    Check if a post is about a REAL, actionable disaster.

    Requires BOTH:
      1. At least one disaster-type keyword (hurricane, earthquake, etc.)
      2. At least one impact/action keyword (damage, rescue, trapped, etc.)

    Also rejects posts with entertainment/irrelevant keywords.
    """
    text_lower = text.lower()

    # Reject if it contains exclusion keywords
    if any(ex in text_lower for ex in EXCLUSION_KEYWORDS):
        return False

    # Must contain a disaster type
    has_disaster = any(d in text_lower for d in DISASTER_TYPE_KEYWORDS)
    if not has_disaster:
        return False

    # Must contain an impact/action term
    has_impact = any(a in text_lower for a in IMPACT_ACTION_KEYWORDS)
    return has_impact


def _get_bluesky_client():
    """
    Create an authenticated Bluesky client using Streamlit secrets.

    Requires these keys in .streamlit/secrets.toml:
        [bluesky]
        handle = "yourhandle.bsky.social"
        app_password = "xxxx-xxxx-xxxx-xxxx"

    Returns:
        atproto.Client instance or None
    """
    try:
        from atproto import Client
        import streamlit as st

        secrets = st.secrets.get("bluesky", {})
        handle = secrets.get("handle", "")
        app_password = secrets.get("app_password", "")

        if not handle or not app_password:
            logger.info("Bluesky credentials not configured in secrets")
            return None

        client = Client()
        client.login(handle, app_password)
        return client

    except ImportError:
        logger.info("atproto not installed — run: pip install atproto")
        return None
    except Exception as e:
        logger.warning("Bluesky client init failed: %s", e)
        return None


def fetch_bluesky_posts(
    queries: list = None,
    limit_per_query: int = 25,
) -> list:
    """
    Fetch disaster-related posts from Bluesky.

    Args:
        queries: Search queries (defaults to DISASTER_SEARCH_QUERIES)
        limit_per_query: Max posts per query

    Returns:
        list of dicts: [{text, source, author, timestamp, url}, ...]
    """
    client = _get_bluesky_client()
    if client is None:
        return []

    queries = queries or DISASTER_SEARCH_QUERIES
    posts = []
    seen_uris = set()

    for query in queries:
        try:
            results = client.app.bsky.feed.search_posts(
                params={
                    "q": query,
                    "limit": min(limit_per_query, 100),
                    "sort": "latest",
                    "lang": "en",
                }
            )

            for post in results.posts:
                uri = post.uri
                if uri in seen_uris:
                    continue
                seen_uris.add(uri)

                text = post.record.text if post.record else ""
                if not text or len(text.strip()) < 30:
                    continue

                # Skip posts that don't contain actual disaster keywords
                if not _is_relevant(text):
                    continue

                author_handle = post.author.handle if post.author else "unknown"
                author_display = (
                    post.author.display_name
                    if post.author and post.author.display_name
                    else author_handle
                )

                # Parse timestamp
                try:
                    created = post.record.created_at
                    if isinstance(created, str):
                        # Handle ISO format
                        ts = created[:19]  # Trim timezone
                        timestamp = datetime.fromisoformat(ts).strftime(
                            "%Y-%m-%d %H:%M UTC"
                        )
                    else:
                        timestamp = str(created)
                except Exception:
                    timestamp = "Unknown"

                # Build Bluesky post URL
                # Format: https://bsky.app/profile/{handle}/post/{rkey}
                rkey = uri.split("/")[-1] if "/" in uri else ""
                post_url = f"https://bsky.app/profile/{author_handle}/post/{rkey}"

                posts.append({
                    "text": text[:1000],
                    "source": "Bluesky",
                    "author": f"@{author_handle}",
                    "author_display": author_display,
                    "timestamp": timestamp,
                    "url": post_url,
                    "likes": post.like_count if hasattr(post, "like_count") else 0,
                    "reposts": post.repost_count if hasattr(post, "repost_count") else 0,
                })

        except Exception as e:
            logger.warning("Error searching Bluesky for '%s': %s", query, e)
            continue

    # Sort by newest first
    posts.sort(key=lambda p: p["timestamp"], reverse=True)

    logger.info("Fetched %d Bluesky posts", len(posts))
    return posts


def is_configured() -> bool:
    """Check if Bluesky credentials are available."""
    try:
        import streamlit as st
        secrets = st.secrets.get("bluesky", {})
        return bool(secrets.get("handle")) and bool(secrets.get("app_password"))
    except Exception:
        return False
