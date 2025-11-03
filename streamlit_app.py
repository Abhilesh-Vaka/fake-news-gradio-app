import streamlit as st
import joblib
import json
import re
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

# --- Your Credentials ---
# (Note: For a real-world app, use st.secrets)
MY_CREDENTIALS = {
    "url": "https://us-south.ml.cloud.ibm.com",
    "apikey": "clSYhPa9D3XkDZKs8tWV1XOq29vtam9otRI1Kzb3bbbI"
}
PROJECT_ID = "07104504-7691-470a-97af-450927eb3ada"

# --- Model Loading (with Caching) ---
# @st.cache_resource is a special command to load these models only ONCE.
@st.cache_resource
def load_baseline_model():
    """Loads the baseline .pkl model from disk."""
    try:
        model = joblib.load('baseline_model.pkl')
        return model
    except FileNotFoundError:
        st.error("ERROR: 'baseline_model.pkl' not found. Please ensure the file is in the GitHub repository.")
        return None

@st.cache_resource
def load_llm_model_inference():
    """Creates a reusable instance of the LLM inference model."""
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
    return model

# --- Your Watsonx Function (Modified for Streamlit) ---
def get_watsonx_prediction(model, article_text):
    """
    Runs the LLM and returns BOTH classification and reasoning.
    """
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

        # Use non-greedy regex to find the first JSON object
        json_match = re.search(r"\{.*?\}", generated_text, re.DOTALL)
        
        if json_match:
            json_string = json_match.group(0)
            result = json.loads(json_string)
        else:
            raise ValueError("No JSON object found in the model's response.")

        classification = result.get('classification', 'N/A')
        reasoning = result.get('reasoning', 'N/A')
        return classification, reasoning

    except Exception as e:
        st.error(f"Error contacting Watsonx API: {e}")
        return "Error", str(e)

# --- Load the models ---
baseline_model = load_baseline_model()
llm_model = load_llm_model_inference()

# --- Streamlit UI ---
st.title("Explainable vs. Black Box Fake News Detection")
st.markdown("This app demonstrates the two models from my project. Paste a news article to compare the 'Black Box' classification against the 'Explainable LLM's' reasoning.")

# Input Text Area
article_text = st.text_area("Paste your full news article text here:", height=250)

# Analyze Button
if st.button("Analyze Article") and baseline_model is not None:
    if not article_text.strip():
        st.warning("Please paste an article to analyze.")
    else:
        # Run both models and show spinners
        with st.spinner("Analyzing with 'Black Box' Model..."):
            baseline_prediction = baseline_model.predict([article_text])[0]
        
        with st.spinner("Analyzing with 'Explainable LLM'... (This may take a moment)"):
            llm_classification, llm_reasoning = get_watsonx_prediction(llm_model, article_text)
        
        st.success("Analysis Complete!")
        
        # Display results side-by-side
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Black Box Model (97% Accuracy)")
            st.metric("Classification", baseline_prediction)
        
        with col2:
            st.subheader("Explainable LLM (With Reasoning)")
            st.metric("Classification", llm_classification)
            st.text_area("Reasoning:", value=llm_reasoning, height=150, disabled=True)