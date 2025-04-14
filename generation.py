from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv
from inference import load_fine_tuned_model, inference
from classifier import classifier_model

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
    model = request.app.state.model
    tokenizer = request.app.state.tokenizer
    response = inference(model, tokenizer, query)
    return {"response": response}

# -----> query Classification Response API <----- #
@app.get("/classifier-response", summary="Classifier Response", response_description="The generated response from the classifier.")
async def get_classifier_response(query: str):
    classification = classifier_model(query)
    return {"classification": classification}
