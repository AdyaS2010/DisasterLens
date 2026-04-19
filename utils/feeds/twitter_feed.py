# ──────────────────────────────────────────────────────────────────────────────
# DisasterLens AI
# Developer: Adya S.
# Email:     adyasastry@gmail.com
# ──────────────────────────────────────────────────────────────────────────────
# utils/feeds/twitter_feed.py — Twitter/X API Connector (Dormant)
#
# Provides hooks for Twitter/X API v2 via Tweepy. Intentionally dormant by
# default — activates only when a valid bearer token is configured and the
# enabled flag is set to true. Designed as a premium feature for enterprise
# disaster management clients willing to pay for API access.
# ──────────────────────────────────────────────────────────────────────────────

"""
DisasterLens AI — Twitter/X Live Feed Connector (DORMANT)

This module provides hooks for the Twitter/X API v2 via Tweepy.
It is INTENTIONALLY DORMANT and will only activate when:
    1. A valid TWITTER_BEARER_TOKEN is set in Streamlit secrets
    2. The [twitter] enabled flag is set to true

This is designed as a premium feature for enterprise clients
(e.g., local disaster management agencies) who are willing to
pay for Twitter/X API access.

ACTIVATION INSTRUCTIONS:
    1. Create a Twitter/X Developer account at developer.twitter.com
    2. Create a Project + App, generate a Bearer Token
    3. Add to .streamlit/secrets.toml:
        [twitter]
        bearer_token = "your-bearer-token-here"
        enabled = true
    4. The Live Feed page will automatically detect and enable Twitter

PRICING NOTE (as of 2025):
    - Free tier: 1 app, 1,500 tweets/month read
    - Basic tier ($100/mo): 10,000 tweets/month read
    - Pro tier ($5,000/mo): 1M tweets/month read
"""

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Disaster-related search queries for Twitter
TWITTER_QUERIES = [
    "(hurricane OR earthquake OR flood OR wildfire OR tornado) (help OR emergency OR rescue OR damage OR trapped) -is:retweet lang:en",
    "(disaster OR evacuation OR storm surge OR tsunami) (urgent OR critical OR stranded) -is:retweet lang:en",
]


def _get_twitter_client():
    """
    Create an authenticated Twitter/X API v2 client using Streamlit secrets.

    Requires these keys in .streamlit/secrets.toml:
        [twitter]
        bearer_token = "..."
        enabled = true

    Returns:
        tweepy.Client instance or None
    """
    try:
        import tweepy
        import streamlit as st

        secrets = st.secrets.get("twitter", {})
        bearer_token = secrets.get("bearer_token", "")
        enabled = secrets.get("enabled", False)

        if not enabled:
            logger.info("Twitter API is disabled in secrets (enabled=false)")
            return None

        if not bearer_token:
            logger.info("Twitter bearer_token not configured in secrets")
            return None

        client = tweepy.Client(bearer_token=bearer_token)
        return client

    except ImportError:
        logger.info("tweepy not installed — run: pip install tweepy")
        return None
    except Exception as e:
        logger.warning("Twitter client init failed: %s", e)
        return None


def fetch_twitter_posts(
    queries: list = None,
    max_results: int = 50,
) -> list:
    """
    Fetch disaster-related tweets from X/Twitter API v2.

    This function only works when Twitter API is ENABLED in secrets.

    Args:
        queries: Twitter search queries (defaults to TWITTER_QUERIES)
        max_results: Max tweets per query (10-100)

    Returns:
        list of dicts: [{text, source, author, timestamp, url}, ...]
    """
    client = _get_twitter_client()
    if client is None:
        return []

    queries = queries or TWITTER_QUERIES
    posts = []
    seen_ids = set()

    for query in queries:
        try:
            tweets = client.search_recent_tweets(
                query=query,
                max_results=min(max_results, 100),
                tweet_fields=["created_at", "author_id", "lang", "public_metrics"],
                user_fields=["username", "name"],
                expansions=["author_id"],
            )

            if not tweets.data:
                continue

            # Build user lookup from includes
            users = {}
            if tweets.includes and "users" in tweets.includes:
                for user in tweets.includes["users"]:
                    users[user.id] = user

            for tweet in tweets.data:
                if tweet.id in seen_ids:
                    continue
                seen_ids.add(tweet.id)

                author = users.get(tweet.author_id)
                username = f"@{author.username}" if author else f"@user_{tweet.author_id}"

                created = tweet.created_at
                if isinstance(created, datetime):
                    timestamp = created.strftime("%Y-%m-%d %H:%M UTC")
                else:
                    timestamp = str(created) if created else "Unknown"

                metrics = tweet.public_metrics or {}

                posts.append({
                    "text": tweet.text[:1000],
                    "source": "Twitter/X",
                    "author": username,
                    "timestamp": timestamp,
                    "url": f"https://x.com/i/status/{tweet.id}",
                    "likes": metrics.get("like_count", 0),
                    "retweets": metrics.get("retweet_count", 0),
                    "replies": metrics.get("reply_count", 0),
                })

        except Exception as e:
            logger.warning("Error searching Twitter for query: %s", e)
            continue

    posts.sort(key=lambda p: p["timestamp"], reverse=True)

    logger.info("Fetched %d Twitter posts", len(posts))
    return posts


def is_configured() -> bool:
    """Check if Twitter API is enabled and credentials are available."""
    try:
        import streamlit as st
        secrets = st.secrets.get("twitter", {})
        return (
            bool(secrets.get("enabled", False))
            and bool(secrets.get("bearer_token"))
        )
    except Exception:
        return False


def get_status() -> str:
    """
    Get human-readable status of Twitter API connectivity.

    Returns:
        str: Status message for display in the UI
    """
    try:
        import streamlit as st
        secrets = st.secrets.get("twitter", {})

        if not secrets.get("bearer_token"):
            return "🔒 Not configured — requires API key purchase"
        if not secrets.get("enabled", False):
            return "⏸️  Configured but disabled — set enabled=true to activate"
        return "✅ Active"
    except Exception:
        return "🔒 Not configured"
