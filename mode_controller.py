from pathlib import Path
import json
from datetime import datetime

current_mode = "guided"
MODE_FILE = Path("./memory/mode.json")

def set_mode(mode):
    global current_mode
    valid_modes = ["guided", "autonomous", "exploration", "silent"]
    if mode in valid_modes:
        current_mode = mode
        with open(MODE_FILE, 'w') as f:
            json.dump({"mode": mode, "timestamp": datetime.utcnow().isoformat()}, f)

def get_mode():
    global current_mode
    if MODE_FILE.exists():
        with open(MODE_FILE) as f:
            data = json.load(f)
            current_mode = data.get("mode", "guided")
    return current_mode
