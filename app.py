import os
import gradio as gr
import joblib
import json
import re
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

# --- Your Credentials ---
import os # Make sure this is at the top of your file

# --- Secure Credentials ---
# We will get the API key from Hugging Face "Secrets"
IBM_API_KEY = os.environ.get("IBM_API_KEY")

if not IBM_API_KEY:
    print("CRITICAL ERROR: IBM_API_KEY not found. Please set it in Hugging Face Spaces secrets.")
    # We can still let the app load, but the LLM will fail

MY_CREDENTIALS = {
    "url": "https://us-south.ml.cloud.ibm.com",
    "apikey": IBM_API_KEY 
}
PROJECT_ID = "07104504-7691-470a-97af-450927eb3ada"

# --- Your LLM Function (Modified for UI) ---
def get_watsonx_prediction(article_text):
    """
    Runs the LLM and returns BOTH classification and reasoning.
    """
    params = {
        GenParams.DECODING_METHOD: "greedy",
        GenParams.MAX_NEW_TOKENS: 500,
    }
    model = ModelInference(
        model_id="ibm/granite-3-3-8b-instruct",
        params=params,
        credentials=MY_CREDENTIALS,
        project_id=PROJECT_ID
    )

    prompt = f"""
    You are an expert news analyst. Your task is to analyze a news article and determine if it is "Real" or "Fake".
    You must ONLY respond with a valid JSON object. Do not include any text before or after the JSON.

    Here is an example of the desired output format:
    ---
    Article: "A new study shows that chocolate is good for you."
    JSON Output:
    {{
      "reasoning": "The article makes a claim but does not cite the study or provide any verifiable sources. The tone is overly positive and lacks scientific rigor.",
      "classification": "Fake"
    }}
    ---

    Now, analyze the following article and provide the JSON output.

    Article:
    "{article_text}"

    JSON Output:
    """

    try:
        response = model.generate(prompt=prompt)
        generated_text = response['results'][0]['generated_text']

        # FIX 1: Use non-greedy regex
        json_match = re.search(r"\{.*?\}", generated_text, re.DOTALL)
        
        if json_match:
            json_string = json_match.group(0)
            result = json.loads(json_string)
        else:
            raise ValueError("No JSON object found in the model's response.")

        # FIX 2: Return both values for the UI
        classification = result.get('classification', 'N/A')
        reasoning = result.get('reasoning', 'N/A')
        return classification, reasoning

    except Exception as e:
        print(f"Error in Watsonx: {e}")
        # Return error info to the UI
        return "Error", str(e)

# --- Load Baseline Model ---
try:
    baseline_model = joblib.load('baseline_model.pkl')
    print("Baseline model loaded successfully.")
except FileNotFoundError:
    print("ERROR: 'baseline_model.pkl' not found. Please run 'train_baseline.py' first.")
    # Create a dummy function if model is missing, so UI can still load
    baseline_model = lambda x: ["Error: 'baseline_model.pkl' not found."]

# --- Main Function for Gradio ---
def analyze_news(article_text):
    """
    Runs both models and returns all results.
    """
    if not article_text.strip():
        return "No input", "No input", "Please paste an article."
        
    # 1. Run Baseline Model
    baseline_prediction = baseline_model.predict([article_text])[0]
    
    # 2. Run Explainable LLM
    llm_classification, llm_reasoning = get_watsonx_prediction(article_text)
        
    return baseline_prediction, llm_classification, llm_reasoning

# --- Build the Gradio UI ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # Explainable vs. Black Box Fake News Detection
        This demo compares the two models from my research project. 
        Paste a news article to see the "Black Box" classification vs. the "Explainable LLM" classification and its reasoning.
        """
    )
    
    with gr.Row():
        text_input = gr.Textbox(
            lines=15, 
            placeholder="Paste your full news article text here...", 
            label="Input News Article"
        )
        
        with gr.Column():
            gr.Markdown("### Black Box Model (97% Accuracy)")
            baseline_output = gr.Textbox(label="Classification", interactive=False)
            
            gr.Markdown("---") 
            
            gr.Markdown("### Explainable LLM (With Reasoning)")
            llm_class_output = gr.Textbox(label="Classification", interactive=False)
            llm_reason_output = gr.Textbox(label="Reasoning", lines=10, interactive=False)

    analyze_btn = gr.Button("Analyze Article")
    analyze_btn.click(
        fn=analyze_news, 
        inputs=text_input, 
        outputs=[baseline_output, llm_class_output, llm_reason_output],
        api_name="predict"
    )

if __name__ == "__main__":
    print("Starting Gradio UI...")
    demo.launch() # share=True is optional if you need a public link