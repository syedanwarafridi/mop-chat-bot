import tweepy
import os
import openpyxl
from dotenv import load_dotenv
import os
import pandas as pd
from urllib.parse import urlparse, unquote
from datetime import datetime, timedelta, timezone

load_dotenv()

consumer_key = os.getenv("CONSUMER_API_KEY")
consumer_secret = os.getenv("CONSUMER_API_SECRET")
access_token = os.getenv("ACCESS_TOKEN")
access_token_secret = os.getenv("ACCESS_TOKEN_SECRET")
bearer_token = os.getenv("BEARER_TOKEN")

client = tweepy.Client(
    bearer_token=bearer_token,
    consumer_key=consumer_key,
    consumer_secret=consumer_secret,
    access_token=access_token,
    access_token_secret=access_token_secret,
    wait_on_rate_limit=True
)

# ----------------> Post Tweets <----------------
def post_tweets(client, text, media_paths=None):
    auth = tweepy.OAuth1UserHandler(
        consumer_key,
        consumer_secret,
        access_token,
        access_token_secret
    )
    api = tweepy.API(auth)

    media_ids = []
    if media_paths:
        for media_path in media_paths:
            if os.path.isfile(media_path):
                try:
                    media = api.media_upload(media_path)
                    media_ids.append(media.media_id)
                except Exception as e:
                    print(f"Error uploading {media_path}: {e}")
            else:
                print(f"File not found: {media_path}")

    try:
        if media_ids:
            response = client.create_tweet(
                text=text,
                media_ids=media_ids,
                user_auth=True
            )
        else:
            response = client.create_tweet(
                text=text,
                user_auth=True
            )
        print(f"Tweet posted successfully: https://twitter.com/user/status/{response.data['id']}")
        return response
    except Exception as e:
        return f"An error occurred while posting the tweet: {e}"
    

# ----------------> Lateste 3 Posts <----------------
def get_latest_top3_posts():
    # client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)

    try:
        userr = client.get_me()
        username = userr.data.name if user.data else "MIND_agent"
        user = client.get_user(username=username)
        user_id = user.data.id
    except Exception as e:
        return f"Failed to get user ID: {e}"

    try:
        response = client.get_users_tweets(
            id=user_id,
            max_results=10,
            tweet_fields=['created_at', 'public_metrics'],
            exclude=['retweets', 'replies']
        )

        tweets = response.data if response.data else []
        top3 = sorted(tweets, key=lambda x: x.created_at, reverse=True)[:3]

        result = []
        for tweet in top3:
            result.append({
                'tweet_id': tweet.id,
                'created_at': tweet.created_at,
                'text': tweet.text,
                'like_count': tweet.public_metrics['like_count'],
                'retweet_count': tweet.public_metrics['retweet_count']
            })

        return result

    except Exception as e:
        return f"Failed to get tweets: {e}"
    
# ----------------> Extracting Tweet-Replies <----------------
def get_replies_to_tweets(tweet_ids):
    # client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)
    all_replies = []

    for tweet_id in tweet_ids:
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
                    for tweet in response.data:
                        author = users.get(tweet.author_id)
                        if author:
                            all_replies.append({
                                'conversation_id': tweet_id,
                                'tweet_id': tweet.id,
                                'username': author.username,
                                'created_at': tweet.created_at,
                                'text': tweet.text
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
                'text': reply['text']
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