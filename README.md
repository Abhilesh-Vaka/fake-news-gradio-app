# Fake News Gradio App 🚀

A Gradio web app that compares a traditional baseline model with an explainable IBM watsonx LLM to classify news articles as Real or Fake and provide reasoning.

## Demo
- Live Space: https://huggingface.co/spaces/AbhileshV/fake-news-gradio-app

## How it works
- Baseline: Loads `baseline_model.pkl` (optional). If missing, the UI still loads with a fallback message.
- LLM: Calls IBM watsonx (`ibm/granite-3-3-8b-instruct`) and returns both classification and reasoning as JSON.

## Run locally
```bash
pip install -r requirements.txt
python app.py
```
Then open the URL printed by Gradio.

## Space configuration
- SDK: Gradio
- App file: `app.py`
- Requirements: `requirements.txt`

### Secrets (Variables and secrets)
Set these in your Space Settings → Variables and secrets:
- `WATSONX_URL` = `https://us-south.ml.cloud.ibm.com`
- `WATSONX_APIKEY` = your IBM Cloud API key
- `WATSONX_PROJECT_ID` = your IBM Watsonx Project ID

## Files
- `app.py` — Gradio UI and inference logic
- `requirements.txt` — Python deps (`gradio`, `joblib`, `ibm-watsonx-ai`)
- `baseline_model.pkl` — optional sklearn model for the baseline path
- `README.md` — this file

## Notes
- Keep your Space Private unless you’re comfortable with public usage.
- If the LLM returns a credentials error, verify the secrets above and rebuild the Space.
