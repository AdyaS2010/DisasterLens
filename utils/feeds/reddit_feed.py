# ──────────────────────────────────────────────────────────────────────────────
# DisasterLens AI
# Developer: Adya S.
# Email:     adyasastry@gmail.com
# ──────────────────────────────────────────────────────────────────────────────
# utils/feeds/reddit_feed.py — Reddit Live Feed Connector
#
# Fetches disaster-related posts from Reddit using PRAW (Python Reddit API
# Wrapper). Monitors curated disaster subreddits, searches by keyword, and
# returns structured post data for urgency classification.
# ──────────────────────────────────────────────────────────────────────────────

"""
DisasterLens AI — Reddit Live Feed Connector
Fetches disaster-related posts from Reddit using PRAW.
"""

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Subreddits monitored for disaster content
DISASTER_SUBREDDITS = [
    "TropicalWeather",
    "weather",
    "Earthquakes",
    "wildfireCA",
    "NaturalDisasters",
    "preppers",
    "hurricane",
    "tornado",
    "flooding",
]

# Search keywords
DISASTER_KEYWORDS = [
    "hurricane", "earthquake", "flood", "wildfire", "tornado",
    "tsunami", "storm", "evacuation", "emergency", "disaster",
    "rescue", "shelter", "trapped", "damage", "destroyed",
]


def _get_reddit_client():
    """
    Create an authenticated Reddit client using Streamlit secrets.

    Requires these keys in .streamlit/secrets.toml:
        [reddit]
        client_id = "..."
        client_secret = "..."
        user_agent = "DisasterLens/1.0"

    Returns:
        praw.Reddit instance or None
    """
    try:
        import praw
        import streamlit as st

        secrets = st.secrets.get("reddit", {})
        client_id = secrets.get("client_id", "")
        client_secret = secrets.get("client_secret", "")
        user_agent = secrets.get("user_agent", "DisasterLens/1.0")

        if not client_id or not client_secret:
            logger.info("Reddit API credentials not configured in secrets")
            return None

        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
        )
        # Verify connection (read-only is fine)
        reddit.read_only = True
        return reddit

    except ImportError:
        logger.info("PRAW not installed — run: pip install praw")
        return None
    except Exception as e:
        logger.warning("Reddit client init failed: %s", e)
        return None


def fetch_reddit_posts(
    keywords: list = None,
    subreddits: list = None,
    limit: int = 50,
    time_filter: str = "week",
) -> list:
    """
    Fetch disaster-related posts from Reddit.

    Args:
        keywords: Search terms (defaults to DISASTER_KEYWORDS)
        subreddits: Subreddits to search (defaults to DISASTER_SUBREDDITS)
        limit: Max posts per subreddit search
        time_filter: 'hour', 'day', 'week', 'month', 'year', 'all'

    Returns:
        list of dicts: [{text, source, author, timestamp, url, subreddit}, ...]
    """
    reddit = _get_reddit_client()
    if reddit is None:
        return []

    keywords = keywords or DISASTER_KEYWORDS
    subreddits = subreddits or DISASTER_SUBREDDITS
    query = " OR ".join(keywords[:10])  # Reddit has query length limits

    posts = []
    seen_ids = set()

    for sub_name in subreddits:
        try:
            subreddit = reddit.subreddit(sub_name)
            results = subreddit.search(
                query,
                sort="new",
                time_filter=time_filter,
                limit=limit,
            )

            for post in results:
                if post.id in seen_ids:
                    continue
                seen_ids.add(post.id)

                # Combine title + selftext for richer analysis
                text = post.title
                if post.selftext and len(post.selftext.strip()) > 0:
                    text = f"{post.title}. {post.selftext[:500]}"

                created = datetime.fromtimestamp(
                    post.created_utc, tz=timezone.utc
                )

                posts.append({
                    "text": text[:1000],  # Reasonable limit
                    "source": "Reddit",
                    "author": f"u/{post.author.name}" if post.author else "u/[deleted]",
                    "timestamp": created.strftime("%Y-%m-%d %H:%M UTC"),
                    "url": f"https://reddit.com{post.permalink}",
                    "subreddit": f"r/{sub_name}",
                    "score": post.score,
                    "num_comments": post.num_comments,
                })

        except Exception as e:
            logger.warning("Error fetching from r/%s: %s", sub_name, e)
            continue

    # Sort by newest first
    posts.sort(key=lambda p: p["timestamp"], reverse=True)

    logger.info("Fetched %d Reddit posts", len(posts))
    return posts


def is_configured() -> bool:
    """Check if Reddit API credentials are available."""
    try:
        import streamlit as st
        secrets = st.secrets.get("reddit", {})
        return bool(secrets.get("client_id")) and bool(secrets.get("client_secret"))
    except Exception:
        return False
