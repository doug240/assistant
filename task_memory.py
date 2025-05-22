# task_memory.py
import json
from pathlib import Path
from datetime import datetime
from difflib import SequenceMatcher

MEMORY_FILE = Path("./memory/tasks.json")
ARCHIVE_FILE = Path("./memory/tasks_archive.json")

def load_tasks():
    if MEMORY_FILE.exists():
        with open(MEMORY_FILE, 'r') as f:
            return json.load(f)
    return []

def save_tasks(tasks):
    with open(MEMORY_FILE, 'w') as f:
        json.dump(sorted(tasks, key=lambda x: x.get("timestamp", ""), reverse=True), f, indent=2)

def validate_task(task):
    return isinstance(task, dict) and "description" in task and isinstance(task["description"], str)

def save_task(task):
    if not validate_task(task):
        raise ValueError("Invalid task format")
    task.setdefault("timestamp", datetime.utcnow().isoformat())
    task.setdefault("completed", False)
    tasks = load_tasks()
    tasks.append(task)
    save_tasks(tasks)

def update_task(index, updated_task):
    tasks = load_tasks()
    if 0 <= index < len(tasks):
        tasks[index].update(updated_task)
        save_tasks(tasks)
    else:
        raise IndexError("Invalid task index")

def delete_task(index):
    tasks = load_tasks()
    if 0 <= index < len(tasks):
        tasks.pop(index)
        save_tasks(tasks)
    else:
        raise IndexError("Invalid task index")

def search_by_metadata(key, value):
    return [t for t in load_tasks() if t.get(key) == value]

def archive_completed():
    tasks = load_tasks()
    active, completed = [], []
    for t in tasks:
        if t.get("completed"):
            completed.append(t)
        else:
            active.append(t)

    # Archive completed tasks
    if completed:
        archived = []
        if ARCHIVE_FILE.exists():
            with open(ARCHIVE_FILE, 'r') as f:
                archived = json.load(f)
        archived.extend(completed)
        with open(ARCHIVE_FILE, 'w') as f:
            json.dump(archived, f, indent=2)

    save_tasks(active)

def export_tasks(path):
    tasks = load_tasks()
    with open(path, 'w') as f:
        json.dump(tasks, f, indent=2)

def merge_similar_tasks(threshold=0.8):
    tasks = load_tasks()
    merged = []
    seen = set()

    for i, task in enumerate(tasks):
        if i in seen:
            continue
        for j in range(i + 1, len(tasks)):
            if j in seen:
                continue
            sim = SequenceMatcher(None, task['description'], tasks[j]['description']).ratio()
            if sim >= threshold:
                # Merge metadata if present
                merged_metadata = {**task.get("metadata", {}), **tasks[j].get("metadata", {})}
                task['metadata'] = merged_metadata
                seen.add(j)
        merged.append(task)

    save_tasks(merged)
