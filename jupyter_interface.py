from jupyter_client import KernelManager
import json
from pathlib import Path
from datetime import datetime

km = KernelManager()
km.start_kernel()
kc = km.client()
kc.start_channels()

JUPYTER_LOG = Path("./logs/jupyter_log.jsonl")

def run_jupyter_code(code):
    kc.execute(code)
    while True:
        msg = kc.get_iopub_msg()
        msg_type = msg['header']['msg_type']
        if msg_type == 'execute_result':
            _log_jupyter(code, msg['content']['data']['text/plain'])
            return msg['content']['data']['text/plain']
        elif msg_type == 'error':
            _log_jupyter(code, msg['content']['evalue'], error=True)
            return f"Error: {msg['content']['evalue']}"

def _log_jupyter(code, output, error=False):
    with open(JUPYTER_LOG, 'a') as f:
        f.write(json.dumps({
            "code": code,
            "output": output,
            "error": error,
            "timestamp": datetime.utcnow().isoformat()
        }) + "\n")
