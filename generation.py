from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv
from inference import load_fine_tuned_model, inference, twitter_post_writer
from classifier import classifier_model
from twitter_apis import post_tweets, get_latest_top3_posts, get_replies_to_tweets, extract_usernames_from_excel, filter_replies_by_usernames, filter_recent_replies
from fastapi.responses import JSONResponse
import traceback

load_dotenv()

# -----> Fastapi Setup <----- #
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
        model = request.app.state.model
        tokenizer = request.app.state.tokenizer

        # Generate tweet content
        tweet_content = twitter_post_writer(model, tokenizer)

        # Post the tweet
        post_tweets(tweet_content)

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
list_of_posts = get_latest_top3_posts() # return my latest top 3 posts
posts = [post['tweet_id'] for post in list_of_posts] # extract ids
list_of_replies = get_replies_to_tweets(posts) # return replies to my latest top 3 posts
usernames = extract_usernames_from_excel()
usernames_filtered_replies = filter_replies_by_usernames(list_of_replies, usernames) # filter replies by usernames from excel file
time_filtered_replies = filter_recent_replies(usernames_filtered_replies)