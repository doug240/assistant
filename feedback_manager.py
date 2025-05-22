import json
from pathlib import Path
from datetime import datetime

FEEDBACK_FILE = Path("./logs/feedback.jsonl")

def ask_for_feedback(output):
    print("Output:\n", output)
    rating = input("Rate this output (1-5): ")
    comment = input("Any feedback or corrections?: ")
    record = {
        "output": output,
        "rating": rating,
        "correction": comment,
        "timestamp": datetime.utcnow().isoformat()
    }
    with open(FEEDBACK_FILE, 'a') as f:
        f.write(json.dumps(record) + "\n")
    return record
