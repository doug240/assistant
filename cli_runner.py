import subprocess
from pathlib import Path
import json
from datetime import datetime

CLI_HISTORY_FILE = Path("./logs/cli_history.jsonl")

def run_command(command, env=None, timeout=15):
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True, env=env, timeout=timeout)
        history = {
            "command": command,
            "output": result.stdout,
            "error": result.stderr,
            "code": result.returncode,
            "timestamp": datetime.utcnow().isoformat()
        }
        with open(CLI_HISTORY_FILE, 'a') as f:
            f.write(json.dumps(history) + "\n")
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Command failed: {e.stderr}"
    except subprocess.TimeoutExpired:
        return f"Command timeout exceeded"
