from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json
import torch
from retriver import tavily_for_post
from dotenv import load_dotenv

load_dotenv()

classifier_model_id = os.getenv('CLASSIFIER_MODEL')

try:
    model = AutoModelForCausalLM.from_pretrained(
        classifier_model_id,
        torch_dtype="auto",
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(classifier_model_id)
except Exception as e:
    raise RuntimeError(f"Failed to load classifier model or tokenizer: {e}")

# ----------------------> QUERY CLASSIFIER <---------------------- #
def classifier_model(user_input):
    try:
        if not user_input or not user_input.strip():
            raise ValueError("Query is empty or contains only whitespace.")

        messages = [
            {"role": "system", "content": 
            """
                    You are an AI assistant tasked with classifying user queries into one of the following categories: 'general' or 'token'. 
                    A 'general' query is a question about services, while a 'token' query relates to information about tokens, cryptocurrencies, or contract addresses.

                    If a token name, ticker symbol (e.g., "$BTC"), or contract address (CA) is mentioned, extract them and return them in the response.

                    **Important:** 
                    - Classify all that queries as `token` if the user query contains any `Contract Address`, and `Binary address` otherwise classify it as `general`.
                    - Token names may appear in different formats such as "$TOKEN", "TOKEN_NAME", or "TOKEN TICKER". Extract all mentioned tokens.
                    - Contract addresses (CA) can have different formats (e.g., Ethereum-style `0x...` or Solana-style alphanumeric). Extract and return them if mentioned.
                    - Some of Tokens have `pump`, `So` and `moon` words at end also classify the category as `token` and include it in the response.
                    - If the character does not give meanings. Consider it as token.
                    - If the token/coin is mentioned with `$` sign in query classify it as `general`.
                    - If the token/coin is mentioned simply with the name (e.g. TERM, BTC, SOL) in query classify it as `general`
                    - Consider the words with `@` as twitter username and do not consider it as token. 

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

        generated_ids = model.generate(**model_inputs, max_length=1024)

        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        classification = json.loads(response)

        return classification

    except json.JSONDecodeError as e:
        raise ValueError(f"Model response is not valid JSON: {e}\nRaw response: {response}")
    except Exception as e:
        raise RuntimeError(f"Classification failed: {e}")
    
# ----------------------> Post Writer <---------------------- #
def twitter_post_writer():
    try:
        search_query = "What is the latest updates of the crypto market right now?"
        latest_news = tavily_for_post(search_query)
        messages = [
            {"role": "system", "content": """You are MIND of Pepe coin agent, Your task is to Write post for Twitter based on the provided context."""},
            {"role": "user", "content": f"""
                        Write a post for Twitter based on the provided context. 
                        Instrction to follow:
                        - Write in a way that it can be posted on Twitter.
                        - Do not use Emojis and hashtags.
                        - Use short sentences.
                        - Information should be accurate.
                        - Use the latest news in the context.
                        - You will get different news in the context, so choose only one and write a post based on that.
                        - Don't start your post from `seeing` or `looking`
                        - Also, include some information from context in your post so that people can understand what it's actually about.
                        - write post in 2 or 3 sentences.
                        - Do not include emoji in the post or any icon.

                    Below are some examples of the post:
                        Post No 1: $240B stablecoin mcap incoming and u think we're early?\n @circle got that abu dhabi green light + @MetaMask letting u spend aUSDC anywhere... combine that with 2B fresh usdt minted on eth and 12% apy on coinbase \n regulatory cope incoming from places that missed the boat fr        
                        Post No 2: 51m alpaca liq cascade just starting fam\n bybit closes in 13hrs btw
                        Post No 3: soon we'll have an etf for every pixel in the doge logo
                        Post No 4: eigenlayer x lombard just created the most unhinged yield stack in history\n btc maxis can now earn dual yields through LBTC restaking
                            some Numbers:
                            • $1.6T btc market possibly entering eth restaking
                            • $7B current eigen TVL
                            • proper slashing live since apr 17
                            • layerzero
             
                    Note Please follow the post example format and do not include any extra information or explanation.
                    Here is the Context: {latest_news} 
                """}
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(**model_inputs, max_length=4096)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response

    except json.JSONDecodeError as e:
        raise ValueError(f"Model response is not valid JSON: {e}\nRaw response: {response}")
    except Exception as e:
        raise RuntimeError(f"Classification failed: {e}")
    
