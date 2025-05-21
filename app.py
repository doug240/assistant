import os
import torch
from pathlib import Path
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import traceback

app = Flask(__name__)

model_name_or_path = "D:/trained_models/fine_tuned_model"  # Change after fine tuning
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
model.eval()
if torch.cuda.is_available():
    model.to("cuda")

# Simple Memory Codex class
class MemoryCodex:
    def __init__(self, max_memory=5):
        self.max_memory = max_memory
        self.history = []

    def add(self, role, content):
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_memory:
            self.history.pop(0)

    def get_context(self):
        # Combine conversation history into a single string prompt
        context = ""
        for entry in self.history:
            prefix = "User: " if entry["role"] == "user" else "Assistant: "
            context += prefix + entry["content"] + "\n"
        return context

memory = MemoryCodex(max_memory=8)

@app.route("/infer", methods=["POST"])
def infer():
    try:
        data = request.json
        user_prompt = data.get("prompt", "").strip()
        if not user_prompt:
            return jsonify({"error": "Empty prompt"}), 400

        memory.add("user", user_prompt)
        context_prompt = memory.get_context()

        inputs = tokenizer.encode(context_prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")

        outputs = model.generate(
            inputs,
            max_length=inputs.shape[1] + 100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=2,
        )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only assistant reply after last user prompt
        assistant_reply = generated[len(context_prompt):].strip()

        memory.add("assistant", assistant_reply)

        return jsonify({"response": assistant_reply})

    except Exception as e:
        return jsonify({"error": f"Exception during inference: {traceback.format_exc()}"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
