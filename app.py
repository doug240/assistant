from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

app = Flask(__name__)

MODEL_NAME = "EleutherAI/gpt-neo-125M"
FINE_TUNED_DIR = "./fine_tuned_model"

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if os.path.exists(FINE_TUNED_DIR):
        print("Loading fine-tuned model...")
        model = AutoModelForCausalLM.from_pretrained(FINE_TUNED_DIR)
        tokenizer = AutoTokenizer.from_pretrained(FINE_TUNED_DIR)
    else:
        print("Loading base model...")
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    return tokenizer, model

tokenizer, model = load_model()
model.eval()

if torch.cuda.is_available():
    model = model.to("cuda")

@app.route("/infer", methods=["POST"])
def infer():
    data = request.get_json()
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.7)

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({"response": generated_text})

if __name__ == "__main__":
    app.run(debug=True)
