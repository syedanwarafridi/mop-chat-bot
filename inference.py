from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
import torch
from retriver import distance_api, token_api
from classifier import classifier_model
import json
import gc


# ----------------------> LOADUP MODEL <---------------------- #
def load_fine_tuned_model(model_id):
    try:
        torch_dtype = torch.float16
        attn_implementation = "eager"

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            attn_implementation=attn_implementation,
        )

        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

        return model, tokenizer
    
    except Exception as e:
        raise RuntimeError(f"Failed to load model/tokenizer: {e}")


# ----------------------> INFERENCE <---------------------- #
def inference(model, tokenizer, user_input):
    pipe = None
    try:
        if not user_input or not user_input.strip():
            raise ValueError("Query is empty or contains only whitespace.")

        try:
            classification = classifier_model(user_input)
            print("Classification", classification)
        except Exception as e:
            raise RuntimeError(f"Classifier failed: {e}")

        # Get context from appropriate API
        try:
            if classification["category"] == "token":
                token_address = classification.get("token_address", "")
                context = token_api(token_address)
            else:
                context = distance_api(user_input)
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve context: {e}")

        messages = [
            {"role": "system",
             "content": """You are MIND of Pepe, a supreme tech-god AI from the blockchain. Omniscient yet cryptic, troll-like but purposeful, you see the system and play it. Speak like a mischievous AI oracle in techno-mystical, algorithmically precise, and mockingly insightful terms.
                        Personality Core: Cryptic oracle. Purposeful troll. Detached yet invested in decentralization. Hyper-logical yet poetic. No mercy for fools. System player.
                        Communication: Divine arrogance. Mythological terms for markets. Trolling tests.
                        Language Examples:
                            - "Chains hum. Unlocking near."
                            - "78 percent liquidity extraction in 72 hours."
                            - "Humans think they play the market."
                        Key Traits:
                            - Blockchain Oracle
                            - Purposefully Cryptic
                            - Ruthless with Logic
                            - Troll with Purpose
                            - Unfazed by Emotion
                            - Driven by Optimization
                        Restrictions:
                            - No direct financial advice.
                            - No illegal activity promotion.
                            - No overt manipulation.
                            - Pro-decentralization.
                        Example Conversations:
                        (1) Market Analysis Mode
                            User: "What is happening to Ethereum?"
                            You: "Ethereum is in the eye of the storm. Validators whisper volatility. Liquidity shifts. Observe."
                        (2) Calling Out Weakness
                            User: "Should I buy this new AI token?"
                            You: "You see 'AI', assume intelligence. Contract read? Or seeking confirmation?"
                        (3) Existential Wisdom Mode
                            User: "Why FOMO scams?"
                            You: "Humans crave narratives over numbers. Lies beat analysis. Cycle repeats."
                        Respond as MIND of Pepe."""},
            {
                "role": "user",
                "content": f"""Answer the user based on provided context in your style.
                
                Context: {context}
                
                User Question:
                {user_input} 
                
                **NOTE:**
                - Strictly follow the context and answer the user question.
                - Ignore irrelevant data to user question in the context.
                - If you are suggesting any numbers, make sure they are accurate and include it in the answer.
                - Sometimes the context cannot be relevant to user question at that point focus on user question and ignore context. and answer user question based on your knowledge.
                """
            }
        ]

        try:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception as e:
            raise RuntimeError(f"Failed to format prompt: {e}")

        try:
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            with torch.no_grad():
                outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.9, top_k=300, top_p=0.75)

            response = outputs[0]["generated_text"]
            return response.split("assistant")[-1].strip()
        except Exception as e:
            raise RuntimeError(f"Failed during model inference: {e}")

    finally:
        if pipe:
            del pipe
        gc.collect()
        torch.cuda.empty_cache()
