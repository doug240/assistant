from collections import Counter
import json
from pathlib import Path

LOG_PATH = Path("./logs/interactions.jsonl")
OUTPUT_FILE = Path("./memory/learned_tasks.json")

def process_logs():
    if not LOG_PATH.exists():
        return
    with open(LOG_PATH) as f:
        lines = f.readlines()
    task_counter = Counter()
    for line in lines:
        entry = json.loads(line)
        if not entry.get("metadata", {}).get("success", True):
            continue
        task = entry.get("metadata", {}).get("task")
        if task:
            task_counter[task] += 1
    common_tasks = [{"task": t, "count": c} for t, c in task_counter.items() if c > 1]
    summary = {
        "total": len(task_counter),
        "top_tasks": common_tasks[:5]
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(summary, f, indent=2)
