from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv
from inference import load_fine_tuned_model, inference
from classifier import classifier_model
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


# -----> query Classification Response Generation API <----- #
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

