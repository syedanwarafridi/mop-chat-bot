import gradio as gr
from inference import load_fine_tuned_model, inference
from dotenv import load_dotenv
from retriver import last_update_api, update_api
import os

# Load environment variables
load_dotenv()
model_id = os.getenv('NEW_MODEL_ID')

# Load the fine-tuned model and tokenizer
model, tokenizer = load_fine_tuned_model(model_id)

# Define the inference function
def gradio_inference(user_input):
    response, classification, context = inference(model, tokenizer, user_input)
    return response, classification, context

# Define the function to update the database
def update_database():
    update_api()
    return last_update_api()

# Create the Gradio interface using Blocks
with gr.Blocks(css="""
    .orange-button {
        background-color: #ff7f0e !important;
        color: white !important;
        border: none;
    }
""") as demo:
    gr.Markdown("## Crypto Query Assistant")
    gr.Markdown("Enter your query about cryptocurrencies to get a response, classification, and context.")

    with gr.Row():
        with gr.Column(scale=1):
            user_input = gr.Textbox(lines=2, placeholder="Enter your query here...", label="Your Query")
            submit_button = gr.Button("Submit Query", elem_classes="orange-button")
            
            gr.Markdown("---")
            last_updated = gr.Textbox(value=last_update_api(), label="Last Updated", interactive=False)
            update_button = gr.Button("Update Database", elem_classes="orange-button")

        with gr.Column(scale=2):
            response = gr.Textbox(label="Model Response")
            classification = gr.Textbox(label="Classification")
            context = gr.Textbox(label="Context")

    # Button click events
    submit_button.click(fn=gradio_inference, inputs=user_input, outputs=[response, classification, context])
    user_input.submit(fn=gradio_inference, inputs=user_input, outputs=[response, classification, context])
    update_button.click(fn=update_database, outputs=last_updated)

# Launch the app
demo.launch()
