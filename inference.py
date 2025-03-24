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

# ----------------------> LOADUP MODEL <---------------------- #
def load_fine_tuned_model(model_id):
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

# ----------------------> INFERENCE <---------------------- #
def inference(model, tokenizer, user_input):
    classification = classifier_model(user_input)
    if classification["category"] == "token":
        token_address = classification["token_address"]
        token_results = token_api(token_address)
        context = token_results

    else:
        context = distance_api(user_input)

    messages = [
        {"role": "system", 
         "content": "You are a crypto market expert that gives informative answers to user questions based on the context. You can be wrong but you have to be decisive"
         },
        {
            "role": "user",
            "content": f"""Answer the user question based on provided context in your way.
            
            Context: {context}
            
            User Question:
            {user_input} 
            
            **NOTE:**
              - Strictly follow the context and answer the user question.
              - If you are suggesting any numbers, make sure they are accurate and include it in the answer.
            """
        }
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(prompt)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    outputs = pipe(prompt, max_new_tokens=512, do_sample=True, temperature=0.8, top_k=250, top_p=0.95)
    response = outputs[0]["generated_text"]

    return response.split("assistant")[-1]

# ----------------------> MAIN <---------------------- #
# def main():
#     load_dotenv()

#     model_id = os.getenv('MODEL_ID')
#     model, tokenizer = load_fine_tuned_model(model_id)

#     parser = argparse.ArgumentParser(description="Generate responses using a fine-tuned language model.")
#     parser.add_argument('-i', '--input', type=str, required=True, help="Input text for the model to process.")
#     args = parser.parse_args()

#     # Generate and print the response
#     response = inference(model, tokenizer, args.input)
#     print("\nModel Response:")
#     print(response)

    # return response

# if __name__ == "__main__":
#     main()