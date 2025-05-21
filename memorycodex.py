import os
import json
from datetime import datetime
from typing import List, Dict, Optional
import hashlib
import threading

# === Vector DB Integration ===
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# === Memory Layer Paths ===
MEMORY_ROOT = os.path.expanduser("~/ai_assistant_memory")
EPISODIC_PATH = os.path.join(MEMORY_ROOT, "episodic")
SEMANTIC_PATH = os.path.join(MEMORY_ROOT, "semantic")
PROCEDURAL_PATH = os.path.join(MEMORY_ROOT, "procedural")
CODEX_PATH = os.path.join(MEMORY_ROOT, "codex")
VECTOR_INDEX_PATH = os.path.join(MEMORY_ROOT, "vector_index")

for path in [EPISODIC_PATH, SEMANTIC_PATH, PROCEDURAL_PATH, CODEX_PATH, VECTOR_INDEX_PATH]:
    os.makedirs(path, exist_ok=True)

# === Utility ===
def timestamp():
    return datetime.utcnow().isoformat()

def save_json(data: dict, path: str):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def content_hash(content: str) -> str:
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

# === Metadata Normalization Utility ===
def normalize_metadata(metadata: Dict) -> Dict:
    flat_metadata = {}
    for k, v in metadata.items():
        if isinstance(v, dict):
            flat_metadata[k] = json.dumps(v)
        elif isinstance(v, (list, tuple)):
            flat_metadata[k] = ', '.join(map(str, v))
        else:
            flat_metadata[k] = str(v)
    return flat_metadata

# === Threading Lock for Concurrency Support ===
lock = threading.Lock()

# === Episodic Memory ===
def log_episode(event: str, metadata: Dict, priority: str = "normal"):
    metadata["priority"] = priority
    entry = {
        "timestamp": timestamp(),
        "event": event,
        "metadata": metadata
    }
    filename = f"{timestamp().replace(':', '_')}_{event}.json"
    with lock:
        save_json(entry, os.path.join(EPISODIC_PATH, filename))

# === Semantic Memory ===
def is_summary_new(topic: str, new_summary: str) -> bool:
    path = os.path.join(SEMANTIC_PATH, f"{topic}.json")
    if not os.path.exists(path):
        return True
    with open(path, 'r', encoding='utf-8') as f:
        old = json.load(f)
    return content_hash(old["summary"]) != content_hash(new_summary)

def save_summary(topic: str, summary: str):
    if not is_summary_new(topic, summary):
        return
    entry = {
        "timestamp": timestamp(),
        "topic": topic,
        "summary": summary
    }
    with lock:
        save_json(entry, os.path.join(SEMANTIC_PATH, f"{topic}.json"))
    index_to_vector_store(topic, summary, metadata={"type": "semantic_summary"})

# === Procedural Memory ===
def save_procedure(task_name: str, steps: List[str]):
    entry = {
        "timestamp": timestamp(),
        "task_name": task_name,
        "steps": steps
    }
    with lock:
        save_json(entry, os.path.join(PROCEDURAL_PATH, f"{task_name}.json"))

# === Codex: Self-Documentation ===
def update_codex(section: str, content: str):
    path = os.path.join(CODEX_PATH, f"{section}.md")
    with lock:
        with open(path, 'a', encoding='utf-8') as f:
            f.write(f"\n## {timestamp()}\n{content}\n")
    index_to_vector_store(section, content, metadata={"type": "codex", "section": section})

def read_codex(section: str) -> str:
    path = os.path.join(CODEX_PATH, f"{section}.md")
    if not os.path.exists(path):
        return ""
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

# === Vector DB (ChromaDB) Setup ===
chroma_client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=VECTOR_INDEX_PATH))
embedding_function = embedding_functions.DefaultEmbeddingFunction()
collection = chroma_client.get_or_create_collection(name="ai_memory", embedding_function=embedding_function)

def index_to_vector_store(topic: str, text: str, metadata: Optional[Dict] = None):
    doc_id = content_hash(f"{topic}:{text}")
    merged_metadata = {
        "topic": topic,
        "timestamp": timestamp()
    }
    if metadata:
        merged_metadata.update(normalize_metadata(metadata))
    with lock:
        collection.add(
            documents=[text],
            metadatas=[merged_metadata],
            ids=[doc_id]
        )

# === Semantic Search ===
def query_memory(query: str, top_k: int = 5) -> List[Dict]:
    results = collection.query(query_texts=[query], n_results=top_k)
    return [
        {"document": doc, "metadata": meta}
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]

# === Summarization & Promotion ===
def summarize_and_promote():
    files = sorted(os.listdir(EPISODIC_PATH))
    grouped = {}

    for fname in files:
        path = os.path.join(EPISODIC_PATH, fname)
        with open(path, 'r', encoding='utf-8') as f:
            entry = json.load(f)
        topic = entry["metadata"].get("module") or "general"
        grouped.setdefault(topic, []).append(entry)

    for topic, events in grouped.items():
        if len(events) < 3:
            continue
        summary = f"Summary of recent events for {topic}:\n"
        summary += "\n".join(f"- {e['event']} @ {e['timestamp']}" for e in events[-5:])
        save_summary(topic, summary)

# === Self-Evaluation & Meta Reflection ===
def reflect_on_memory():
    reflections = [
        "You’ve logged multiple episodes with high frequency; consider summarizing."
    ]
    summarize_and_promote()
    prune_and_merge_semantic_memory()
    update_codex("Self-Reflection", "\n".join(reflections))

# === Semantic Memory Maintenance ===
def prune_and_merge_semantic_memory():
    files = sorted(os.listdir(SEMANTIC_PATH))
    for fname in files:
        path = os.path.join(SEMANTIC_PATH, fname)
        with open(path, 'r', encoding='utf-8') as f:
            entry = json.load(f)

        topic = entry["topic"]
        summary = entry["summary"]

        similar_entries = query_memory(summary, top_k=3)
        unique_docs = {doc["document"] for doc in similar_entries}

        if len(unique_docs) > 1:
            merged_summary = f"Merged insights for {topic}:\n" + "\n".join(f"- {doc}" for doc in unique_docs)
            save_summary(topic, merged_summary)
            update_codex("MemoryMaintenance", f"Merged redundant summaries under topic: {topic}")

# === Module Analysis Integration ===
from recursive_analysis import analyze_project_structure

def analyze_and_log_modules(target_path: Optional[str] = None):
    target_path = target_path or os.getcwd()
    structure, loose_ends = analyze_project_structure(target_path)

    codex_entry = f"# Project Structure Analysis for {target_path}\n"
    codex_entry += "## Directory Tree:\n"
    codex_entry += structure
    codex_entry += "\n\n## Loose Ends & Potential Issues:\n"
    codex_entry += "\n".join(f"- {issue}" for issue in loose_ends)

    update_codex("ModuleAnalysis", codex_entry)

# === Example Usage ===
if __name__ == "__main__":
    log_episode("ModuleBuild", {"module": "1003", "status": "Started"}, priority="high")
    save_summary("DashboardLayout", "The dashboard uses right-pane logic for expanding modules dynamically.")
    save_procedure("CollapseTabLogic", [
        "Detect active module",
        "Trigger hide on current tab",
        "Expand new tab panel"
    ])
    update_codex("SkillLibrary", "Learned reusable tab collapse pattern from dashboard interaction.")
    reflect_on_memory()
    analyze_and_log_modules("C:/Users/User/Desktop/Projects/Assistant")
    print(query_memory("tab collapsing logic"))
