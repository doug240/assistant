import subprocess
import time

# Launch fine tuning if needed (optional)
# subprocess.Popen(["python", "fine.py"])

# Launch backend API
backend_process = subprocess.Popen(["python", "app.py"])
time.sleep(5)  # Wait for backend to be ready

# Launch UI
ui_process = subprocess.Popen(["python", "ui.py"])

# Wait for UI to close
ui_process.wait()
backend_process.terminate()
