import gradio as gr
from inference import load_fine_tuned_model, terminal_inference
from dotenv import load_dotenv
from retriver import last_update_api, update_api
import os

load_dotenv()
model_id = os.getenv('MODEL_ID')

model, tokenizer = load_fine_tuned_model(model_id)

def gradio_inference(user_input):
    response, classification, context = terminal_inference(model, tokenizer, user_input)
    return response, classification, context

def update_database():
    update_api()
    return last_update_api()

with gr.Blocks(css="""
    .orange-button {
        background-color: #ff7f0e !important;
        color: white !important;
        border: none;
    }
""") as demo:
    gr.Markdown("## Mind of Pepe")
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

demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

