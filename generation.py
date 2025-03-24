from inference import load_fine_tuned_model, inference
import argparse, os

from dotenv import load_dotenv

load_dotenv()

model_id = os.getenv('MODEL_ID')
classifier_model = os.getenv('CLASSIFIER_MODEL')

model = AutoModelForCausalLM.from_pretrained(
    classifier_model,
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(classifier_model)






model, tokenizer = load_fine_tuned_model(model_id)

parser = argparse.ArgumentParser(description="Generate responses using a fine-tuned language model.")
parser.add_argument('-i', '--input', type=str, required=True, help="Input text for the model to process.")
args = parser.parse_args()

    # Generate and print the response
response = inference(model, tokenizer, args.input)
print("\nModel Response:")
print(response)