import json
from datetime import datetime
from pathlib import Path

LOG_DIR = Path("./logs")
LOG_DIR.mkdir(exist_ok=True)

def log_interaction(user_input, assistant_output, metadata=None):
    log_entry = {
        "id": datetime.utcnow().strftime("%Y%m%d%H%M%S%f"),
        "timestamp": datetime.utcnow().isoformat(),
        "user_input": anonymize(user_input),
        "assistant_output": assistant_output,
        "metadata": metadata or {}
    }
    log_day = datetime.utcnow().strftime("%Y-%m-%d")
    with open(LOG_DIR / f"interactions_{log_day}.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    with open(LOG_DIR / "results.jsonl", "a") as rf:
        rf.write(json.dumps({"id": log_entry["id"], "output": assistant_output}) + "\n")

def anonymize(text):
    return text.replace("@", "[at]").replace(".", "[dot]")
