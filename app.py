import gradio as gr
from transformers import pipeline

# Load the sentiment-analysis pipeline
classifier = pipeline("sentiment-analysis")

# Define the function that will process input text
def analyze_sentiment(text):
    return classifier(text)

# Set up the Gradio interface with a submit button instead of clear
interface = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(label="Input Text", placeholder="Enter your text here..."),
    outputs="json",
    live=False,  # Set live to False, so the model runs when the "Submit" button is clicked
    title="Sentiment Analysis",  # Optional, to give the interface a title
    description="Enter text to analyze its sentiment. The model will predict whether the sentiment is positive or negative.",
    theme="compact"  # Optional, to make the interface look more compact
)

# Launch the Gradio app
interface.launch()
