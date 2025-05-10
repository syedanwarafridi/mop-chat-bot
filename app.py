import gradio as gr
from inference import load_fine_tuned_model, grok_inference
from dotenv import load_dotenv
from retriver import last_update_api, update_api
import os

# Load environment variables
load_dotenv()
model_id = os.getenv('MODEL_ID')
username = os.getenv('GRADIO_USERNAME')
password = os.getenv('GRADIO_PASSWORD')

# Load model and tokenizer
# model, tokenizer = load_fine_tuned_model(model_id)

# Define inference function
def gradio_inference(user_input, tweet):
    response, classification, context = grok_inference(user_input, tweet)
    return str(response), str(classification), str(context)

# Define database update function
def update_database():
    update_api()
    return last_update_api()

# Build Gradio interface
with gr.Blocks(css="""
    .orange-button {
        background-color: #ff7f0e !important;
        color: white !important;
        border: none;
    }
""") as demo:
    gr.Markdown("## 🧠 MIND of Pepe")
    gr.Markdown("Enter your query and a relevant tweet to receive a blockchain oracle response.")

    with gr.Row():
        with gr.Column(scale=1):
            user_input = gr.Textbox(lines=2, placeholder="Enter your query here...", label="Your Query")
            tweet_input = gr.Textbox(lines=2, placeholder="Paste a relevant tweet or message here...", label="Tweet/Post")
            submit_button = gr.Button("Submit Query", elem_classes="orange-button")

            gr.Markdown("---")
            last_updated = gr.Textbox(value=last_update_api(), label="Last Updated", interactive=False)
            update_button = gr.Button("Update Database", elem_classes="orange-button")

        with gr.Column(scale=2):
            response = gr.Textbox(label="Model Response", lines=4)
            classification = gr.Textbox(label="Classification", lines=3)
            context = gr.Textbox(label="Context", lines=6)

    # Set up event handlers
    submit_button.click(fn=gradio_inference, inputs=[user_input, tweet_input], outputs=[response, classification, context])
    user_input.submit(fn=gradio_inference, inputs=[user_input, tweet_input], outputs=[response, classification, context])
    tweet_input.submit(fn=gradio_inference, inputs=[user_input, tweet_input], outputs=[response, classification, context])
    update_button.click(fn=update_database, outputs=last_updated)

# Launch the app with authentication
demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=False,
    auth=(username, password)
)
