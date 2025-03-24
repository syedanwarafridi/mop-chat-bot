from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
import os
import json
load_dotenv()

classifier_model = os.getenv('CLASSIFIER_MODEL')

model = AutoModelForCausalLM.from_pretrained(
    classifier_model,
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(classifier_model)

def classifier_model(user_input):
    messages = [
        {"role": "system", "content": 
        """
                You are an AI assistant tasked with classifying user queries into one of the following categories: 'general' or 'token'. 
                A 'general' query is a question about services, while a 'token' query relates to information about tokens, cryptocurrencies, or contract addresses.

                If a token name, ticker symbol (e.g., "$BTC"), or contract address (CA) is mentioned, extract them and return them in the response.

                **Important:** 
                - Token names may appear in different formats such as "$TOKEN", "TOKEN_NAME", or "TOKEN TICKER". Extract all mentioned tokens.
                - Contract addresses (CA) can have different formats (e.g., Ethereum-style `0x...` or Solana-style alphanumeric). Extract and return them if mentioned.

                Please provide your response in **strict JSON format**:
                {{
                    "category": "<Category of the query>",
                    "token_names": ["List of token names if mentioned, otherwise null"],
                    "token_address": "<Token address from the query if mentioned, otherwise empty string>"
                }}
                """},
        {"role": "user", "content": user_input}
        ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(**model_inputs, max_length=1024,)

    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    classification = json.loads(response)
    return classification