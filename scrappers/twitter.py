import tweepy
import json
import logging
from datetime import datetime, timedelta
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Twitter API Credentials
API_KEY = "mQYrnS1PGBOyfamvNUV6izKSo"
API_SECRET = "WCsAc2SAC2Ys9Zs8eRRF6auZhnC9Dhp1qv565uN9GiwmwcYOEu"
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAADWxzwEAAAAAInzQDHWEEuudiFzbDfUubDpLK94%3DmISrdL4mevkHgUTA1KJ0lIs9oOBm08AzLYm3NVhtvbLSZXbjAr"
ACCESS_TOKEN = "1900346814643228672-9qtDx1dEpiYEAaIRK3xxeg43V1k4N4"
ACCESS_TOKEN_SECRET = "oDOPHUKIINB49ZlPXnfaz2X3txu5UoX6hc6rmVGnuQOsA"

# Configuration
QUERY = "Space Science"
MAX_RESULTS = 5  # Number of tweets to fetch
MAX_REPLIES = 5   # Number of replies to fetch per tweet
OUTPUT_FILE = "twitter_results123.json"

def get_tweet_replies(client, tweet_id, author_username, max_replies=5):
    """Fetch replies for a specific tweet"""
    replies = []
    try:
        # Search for replies using conversation_id
        query = f"conversation_id:{tweet_id} to:{author_username}"
        response = client.search_recent_tweets(
            query=query,
            max_results=max_replies,
            tweet_fields=['created_at', 'public_metrics', 'author_id', 'in_reply_to_user_id'],
            user_fields=['username'],
            expansions=['author_id']
        )
        
        if not response.data:
            return replies

        # Create users lookup dictionary
        users = {user.id: user for user in response.includes['users']} if 'users' in response.includes else {}

        # Process replies
        for reply in response.data:
            if reply.in_reply_to_user_id:  # Ensure it's a reply
                user = users.get(reply.author_id)
                if user:
                    reply_data = {
                        "user": user.username,
                        "date": reply.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                        "content": reply.text,
                        "likes": reply.public_metrics['like_count'],
                        "retweets": reply.public_metrics['retweet_count'],
                        "replies": reply.public_metrics['reply_count'],
                        "url": f"https://twitter.com/{user.username}/status/{reply.id}"
                    }
                    replies.append(reply_data)

    except Exception as e:
        logger.error(f"Error fetching replies for tweet {tweet_id}: {str(e)}")

    return replies

def scrape_tweets():
    output = []
    try:
        # Initialize Twitter API v2 client
        client = tweepy.Client(
            bearer_token=BEARER_TOKEN,
            consumer_key=API_KEY,
            consumer_secret=API_SECRET,
            access_token=ACCESS_TOKEN,
            access_token_secret=ACCESS_TOKEN_SECRET,
            wait_on_rate_limit=True
        )

        # Search tweets using Twitter API v2
        tweets = client.search_recent_tweets(
            query=QUERY,
            max_results=MAX_RESULTS,
            tweet_fields=['created_at', 'public_metrics', 'author_id', 'conversation_id'],
            user_fields=['username'],
            expansions=['author_id']
        )

        # Create a user dictionary to look up user information
        users = {user.id: user for user in tweets.includes['users']}

        for tweet in tweets.data:
            user = users[tweet.author_id]
            metrics = tweet.public_metrics
            
            # Get replies for this tweet
            replies = get_tweet_replies(client, tweet.id, user.username, MAX_REPLIES)
            
            tweet_data = {
                "user": user.username,
                "date": tweet.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                "content": tweet.text,
                "likes": metrics['like_count'],
                "retweets": metrics['retweet_count'],
                "replies_count": metrics['reply_count'],
                "url": f"https://twitter.com/{user.username}/status/{tweet.id}",
                "replies": replies  # Add replies to the tweet data
            }
            output.append(tweet_data)
            logger.info(f"Processed tweet and {len(replies)} replies from @{user.username}")
            
            # Add a small delay between processing tweets to avoid rate limits
            time.sleep(1)

        # Save results to JSON file
        if output:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            print(f"✅ Saved {len(output)} tweets with replies to {OUTPUT_FILE}")
        else:
            print("❌ No tweets were found")

    except Exception as e:
        logger.error(f"Error during scraping: {str(e)}")
        raise

if __name__ == "__main__":
    scrape_tweets()
