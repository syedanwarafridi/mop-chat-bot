from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv
from inference import load_fine_tuned_model, inference
from classifier import classifier_model, twitter_post_writer
from twitter_apis import post_tweets, get_latest_top3_posts, get_replies_to_tweets, extract_usernames_from_excel, filter_replies_by_usernames, filter_recent_replies, filter_unreplied_tweets, reply_to_tweet, extract_mentions
from fastapi.responses import JSONResponse
import traceback

load_dotenv()
import tweepy


# -----> Fastapi Setup <----- #
print("Loading FastAPI...")
@asynccontextmanager
async def lifespan(app: FastAPI):
    model_id = os.getenv("MODEL_ID")
    model, tokenizer = load_fine_tuned_model(model_id)
    app.state.model = model
    app.state.tokenizer = tokenizer
    yield
    del app.state.model
    del app.state.tokenizer

app = FastAPI(lifespan=lifespan)


# -----> MOP-Bot Response Generation API <----- #
@app.get("/bot-response", summary="Generate Bot Response", response_description="The generated response from the model.")
async def get_bot_response(request: Request, query: str):
    try:
        if not query or not query.strip():
            return {
                "success": False,
                "error": {
                    "message": "Query cannot be empty."
                }
            }

        model = request.app.state.model
        tokenizer = request.app.state.tokenizer

        response = inference(model, tokenizer, query)

        return {
            "success": True,
            "response": {
                "message": response
            }
        }

    except Exception as e:
        return {
            "success": False,
            "error": {
                "message": str(e)
            }
        }


# -----> Query Classification Response Generation API <----- #
@app.get("/classifier-response", summary="Classifier Response", response_description="The generated response from the classifier.")
async def get_classifier_response(query: str):
    try:
        classification = classifier_model(query)
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "response": {
                    "message": classification
                }
            }
        )
    except Exception as e:
        error_message = str(e)
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": {
                    "message": error_message
                }
            }
        )

# -------> Twitter Post API <----- #
@app.post("/post-tweet", summary="Post a Tweet", response_description="The tweet content and status.")
async def post_tweet(request: Request):
    try:
        tweet_content = twitter_post_writer()
        print("Tweet Content: ", tweet_content)

        response = post_tweets(tweet_content)
        if response.get("error"):
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": {
                        "message": response["error"]
                    }
                }
            )
        
        return {
            "success": True,
            "response": {
                "message": "Tweet posted successfully.",
                "tweet": tweet_content
            }
        }

    except Exception as e:
        error_message = str(e)
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": {
                    "message": error_message
                }
            }
        )
    
#----------> My Posts Tweet-Replies API <-------- #
@app.post("/reply-to-recent", summary="Reply to Recent Tweets", response_description="Replies posted successfully.")
async def reply_to_recent_tweets(request: Request):
    try:
        model = request.app.state.model
        tokenizer = request.app.state.tokenizer

        list_of_posts = get_latest_top3_posts()
        posts = [post['tweet_id'] for post in list_of_posts] 

        list_of_replies = get_replies_to_tweets(posts)
        usernames = extract_usernames_from_excel()
        usernames_filtered_replies = filter_replies_by_usernames(list_of_replies, usernames)

        time_filtered_replies = filter_recent_replies(usernames_filtered_replies)
        unreplied_tweets = filter_unreplied_tweets(time_filtered_replies)

        for tweet in unreplied_tweets:
            query = tweet['text']
            tweet_id = tweet['tweet_id']
            response = inference(model, tokenizer, query)
            reply_to_tweet(tweet_id, response)

        return {
            "success": True,
            "response": {
                "message": "Replies posted successfully.",
                "replied_tweets": [tweet['tweet_id'] for tweet in unreplied_tweets]
            }
        }

    except Exception as e:
        error_message = str(e)
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": {
                    "message": error_message
                }
            }
        )

# -------------> Reply to (who mention me) API  <------------- #
@app.post("/reply-to-mention", summary="Reply to mention Tweets", response_description="Replies posted to mention successfully.")
async def reply_to_mention_tweets(request: Request):
    try:
        model = request.app.state.model
        tokenizer = request.app.state.tokenizer

        list_of_replies = extract_mentions()
        usernames = extract_usernames_from_excel()
        usernames_filtered_replies = filter_replies_by_usernames(list_of_replies, usernames)

        time_filtered_replies = filter_recent_replies(usernames_filtered_replies)
        unreplied_tweets = filter_unreplied_tweets(time_filtered_replies)

        for tweet in unreplied_tweets:
            query = tweet['text']
            tweet_id = tweet['tweet_id']
            response = inference(model, tokenizer, query)
            reply_to_tweet(tweet_id, response)

        return {
            "success": True,
            "response": {
                "message": "Replies posted to mention successfully.",
                "replied_tweets": [tweet['tweet_id'] for tweet in unreplied_tweets]
            }
        }

    except Exception as e:
        error_message = str(e)
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": {
                    "message": error_message
                }
            }
        )

