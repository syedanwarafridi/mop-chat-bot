import tweepy
import os
import openpyxl
from dotenv import load_dotenv
import os
import pandas as pd
from urllib.parse import urlparse, unquote
from datetime import datetime, timedelta, timezone
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

consumer_key = os.getenv("CONSUMER_API_KEY")
consumer_secret = os.getenv("CONSUMER_API_SECRET")
access_token = os.getenv("ACCESS_TOKEN")
access_token_secret = os.getenv("ACCESS_TOKEN_SECRET")
bearer_token = os.getenv("BEARER_TOKEN")

logger.info(f"Credentials: consumer_key={consumer_key[:4]}..., access_token={access_token[:4]}..., bearer_token={bearer_token[:4]}...")

client = tweepy.Client(
    bearer_token=bearer_token,
    consumer_key=consumer_key,
    consumer_secret=consumer_secret,
    access_token=access_token,
    access_token_secret=access_token_secret,
    wait_on_rate_limit=True
)
# ----------------> Get Me <--------------------
def get_my_user_id():
    try:
        user = client.get_me()
        logger.info(f"Authenticated user: {user.data.username}")
        return user.data.id if user.data else None
    except tweepy.TweepyException as e:
        logger.error(f"Error retrieving user ID: {e}")
        return None

user_name = get_my_user_id()
# ----------------> Post Tweets <----------------
def post_tweets(text, media_paths=None):
    try:
        if not text or not isinstance(text, str):
            return {"error": "Tweet text must be a non-empty string"}

        media_ids = []
        if media_paths:
            auth = tweepy.OAuth1UserHandler(consumer_key, consumer_secret, access_token, access_token_secret)
            api = tweepy.API(auth)
            for media_path in media_paths:
                if os.path.isfile(media_path):
                    media = api.media_upload(media_path)
                    media_ids.append(media.media_id)
                else:
                    return {"error": f"File not found: {media_path}"}

        if media_ids:
            response = client.create_tweet(text=text, media_ids=media_ids)
        else:
            response = client.create_tweet(text=text)
        
        print(f"Tweet posted successfully: https://twitter.com/user/status/{response.data['id']}")
        return {"success": True, "tweet_id": response.data["id"]}
    except tweepy.TweepyException as e:
        print(f"An error occurred while posting the tweet: {e}")
        return {"error": str(e)}
    except Exception as e:
        print(f"Unexpected error in post_tweets: {e}")
        return {"error": str(e)}
    

# ----------------> Lateste 3 Posts <----------------
def get_latest_top3_posts():
    try:
        user = client.get_me()
        user_id = user.data.id if user.data else None
        if not user_id:
            logger.error("Failed to get user ID")
            return {"error": "Failed to get user ID"}

        response = client.get_users_tweets(
            id=user_id,
            max_results=5,
            tweet_fields=['created_at', 'public_metrics'],
            exclude=['retweets', 'replies']
        )

        tweets = response.data if response.data else []
        top3 = sorted(tweets, key=lambda x: x.created_at, reverse=True)[:1]

        result = []
        for tweet in top3:
            result.append({
                'tweet_id': tweet.id,
                'created_at': tweet.created_at,
                'text': tweet.text,
                'like_count': tweet.public_metrics['like_count'],
                'retweet_count': tweet.public_metrics['retweet_count']
            })

        logger.info(f"Returning {len(result)} recent posts: {result}")
        return result

    except tweepy.TweepyException as e:
        logger.error(f"Failed to get tweets: {e}")
        return {"error": f"Failed to get tweets: {str(e)}"}
    except Exception as e:
        logger.error(f"Unexpected error in get_latest_top3_posts: {e}")
        return {"error": f"Unexpected error: {str(e)}"}
    
# ----------------> Extracting Tweet-Replies <----------------
def get_replies_to_tweets(tweets_info):
    all_replies = []

    for tweet in tweets_info:
        tweet_id = tweet['tweet_id']
        post_text = tweet['text']
        query = f'conversation_id:{tweet_id} -is:retweet'

        try:
            for response in tweepy.Paginator(
                client.search_recent_tweets,
                query=query,
                tweet_fields=['author_id', 'created_at', 'in_reply_to_user_id'],
                expansions='author_id',
                user_fields=['username'],
                max_results=100
            ):
                if response.data:
                    users = {u['id']: u for u in response.includes['users']}
                    for reply in response.data:
                        author = users.get(reply.author_id)
                        if author:
                            all_replies.append({
                                'conversation_id': tweet_id,
                                'parent_post_text': post_text,
                                'tweet_id': reply.id,
                                'username': author.username,
                                'created_at': reply.created_at,
                                'text': reply.text
                            })
        except Exception as e:
            print(f"An error occurred while fetching replies for tweet {tweet_id}: {e}")

    return all_replies


# ----------------> Extracting Usernames from Excel <----------------
def extract_usernames_from_excel():
    file_path = 'Notebooks/data/MIND.xlsx'
    df = pd.read_excel(file_path)
    
    if 'Profile URL' not in df.columns:
        raise ValueError("The Excel file must contain a 'Profile URL' column.")
    
    usernames = []
    for url in df['Profile URL']:
        if isinstance(url, str):
            parsed_url = urlparse(url)
            path = parsed_url.path
            
            if path.startswith('/search') or path == '/':
                continue
            
            segments = path.strip('/').split('/')
            if segments:
                username = unquote(segments[0])
                usernames.append(username)
    
    return usernames

# ----------------> Filter based on Excel Sheet <----------------
def filter_replies_by_usernames(replies, target_usernames):
    filtered_replies = []

    for reply in replies:
        if reply['username'] in target_usernames:
            filtered_replies.append({
                'tweet_id': reply['tweet_id'],
                'username': reply['username'],
                'created_at': reply['created_at'],
                'text': reply['text'],
                'parent_post_text': reply['parent_post_text']
            })

    return filtered_replies

# -------------> Filtered Replies based on time <------------- #
def filter_recent_replies(replies, hours=15):
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=hours)

    recent_replies = sorted(
        [reply for reply in replies if reply['created_at'] >= cutoff],
        key=lambda r: r['created_at']
    )

    return recent_replies

# ----------------> Filter unreplied tweets  <----------------
def filter_unreplied_tweets(tweets, my_username=user_name):
    client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)
    unreplied = []

    for tweet in tweets:
        tweet_id = tweet['tweet_id']
        query = f'conversation_id:{tweet_id} -is:retweet'

        try:
            found_reply = False
            for response in tweepy.Paginator(
                client.search_recent_tweets,
                query=query,
                tweet_fields=['author_id', 'created_at'],
                expansions='author_id',
                user_fields=['username'],
                max_results=100
            ):
                if response.data:
                    users = {u['id']: u for u in response.includes['users']}
                    for reply in response.data:
                        author = users.get(reply.author_id)
                        if author and author.username.lower() == my_username.lower():
                            found_reply = True
                            break
                if found_reply:
                    break

            if not found_reply:
                unreplied.append(tweet)

        except Exception as e:
            return f"Error checking tweet {tweet_id}: {e}"

    return unreplied

# ----------------> Reply to tweets <---------------
def reply_to_tweet(tweet_id, reply_text):
    try:
        if not reply_text or not isinstance(reply_text, str):
            logger.error("Invalid reply text: must be a non-empty string")
            return {"error": "Reply text must be a non-empty string"}
        if len(reply_text) > 280:
            logger.error("Reply text exceeds 280 characters")
            return {"error": "Reply text exceeds 280 characters"}

        logger.info(f"Posting reply to tweet {tweet_id}: {reply_text[:50]}...")
        response = client.create_tweet(
            text=reply_text,
            in_reply_to_tweet_id=tweet_id
        )

        logger.info(f"Reply posted: https://twitter.com/user/status/{response.data['id']}")
        return {
            "success": True,
            "tweet_id": response.data["id"]
        }
    except tweepy.TweepyException as e:
        logger.error(f"Failed to post reply to tweet {tweet_id}: {e}")
        return {"error": f"Failed to post reply: {str(e)}"}
    except Exception as e:
        logger.error(f"Unexpected error in reply_to_tweet: {e}")
        return {"error": f"Unexpected error: {str(e)}"}

# ----------------> Extract mentions  <----------------
def extract_mentions():
    try:
        username = "Shift1646020"

        # Get user ID
        user = client.get_user(username=username)
        if not user.data:
            return []
        user_id = user.data.id

        # Fetch mentions
        mentions = client.get_users_mentions(
            id=user_id,
            max_results=100,
            expansions=['author_id', 'referenced_tweets.id.author_id'],
            tweet_fields=['created_at', 'referenced_tweets'],
            user_fields=['username']
        )

        mention_details = []
        if not mentions.data:
            return mention_details

        # Map author IDs to usernames
        author_ids = {user.id: user.username for user in mentions.includes.get('users', [])}

        # Process each mention
        for tweet in mentions.data:

            author_username = author_ids.get(tweet.author_id, 'Unknown')
            parent_author_id = None
            parent_post_text = None

            if tweet.referenced_tweets:
                for ref_tweet in tweet.referenced_tweets:
                    if ref_tweet.type in ['replied_to', 'quoted']:
                        ref_tweet_id = ref_tweet.id
                        ref_tweet_data = client.get_tweet(
                            id=ref_tweet_id,
                            tweet_fields=['author_id', 'text']
                        )
                        if ref_tweet_data.data:
                            parent_author_id = ref_tweet_data.data.author_id
                            parent_post_text = ref_tweet_data.data.text
                        else:
                            continue
            else:
                parent_author_id = tweet.author_id
                parent_post_text = tweet.text

            if parent_author_id and parent_author_id != user_id:
                mention_details.append({
                    'username': author_username,
                    'tweet_id': tweet.id,
                    'text': tweet.text,
                    'created_at': tweet.created_at,
                    'parent_post_text': parent_post_text
                })
            else:
                print(f"Skipping mention: tweet_id={tweet.id}, parent_author_id={parent_author_id} is user or invalid")

        return mention_details

    except tweepy.TweepyException as e:
        return []
    except Exception as e:
        return []

#  -----------------> STATS <---------------- #
def get_my_tweets_and_replies():
    user_response = client.get_me()
    if user_response.data is None:
        print("Authenticated user not found.")
        return None
    user_id = user_response.data.id

    tweets = []
    replies = []

    for response in tweepy.Paginator(
        client.get_users_tweets,
        id=user_id,
        tweet_fields=['created_at', 'in_reply_to_user_id'],
        max_results=100,
        exclude=['retweets']
    ):
        if response.data is None:
            continue
        for tweet in response.data:
            if tweet.in_reply_to_user_id is None:
                tweets.append(tweet)
            else:
                replies.append(tweet)

    last_tweet_timestamp = max((tweet.created_at for tweet in tweets), default=None)
    last_reply_timestamp = max((reply.created_at for reply in replies), default=None)

    result = {
        'total_tweets': len(tweets),
        'total_replies': len(replies),
        'last_tweet_timestamp': last_tweet_timestamp,
        'last_reply_timestamp': last_reply_timestamp,
        'tweets': [tweet.text for tweet in tweets],
        'replies': [reply.text for reply in replies]
    }

    return result