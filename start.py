import sys
import subprocess
import time
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QLineEdit,
    QPushButton, QLabel, QMessageBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import requests
import threading

# Launch Flask backend in a separate thread
def run_flask():
    subprocess.run([sys.executable, "app.py"])

flask_thread = threading.Thread(target=run_flask, daemon=True)
flask_thread.start()
time.sleep(3)  # Wait for Flask to start

FLASK_API_URL = "http://localhost:5000/infer"

class ApiThread(QThread):
    response_ready = pyqtSignal(str)

    def __init__(self, prompt):
        super().__init__()
        self.prompt = prompt

    def run(self):
        try:
            response = requests.post(FLASK_API_URL, json={"prompt": self.prompt})
            response.raise_for_status()
            data = response.json()
            if "response" in data:
                self.response_ready.emit(data["response"])
            else:
                self.response_ready.emit(f"Error: {data.get('error', 'Unknown error')}")
        except Exception as e:
            self.response_ready.emit(f"Error contacting API: {str(e)}")

class ChatWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GPT Neo 125M Chat with Memory Codex")
        self.setGeometry(300, 300, 700, 550)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet(
            "background-color: #fff; font-family: Consolas; font-size: 15px;"
        )
        self.layout.addWidget(self.chat_display)

        input_layout = QHBoxLayout()
        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText("Type your message here and press Enter...")
        self.user_input.returnPressed.connect(self.send_message)

        send_button = QPushButton("Send")
        send_button.clicked.connect(self.send_message)

        input_layout.addWidget(self.user_input)
        input_layout.addWidget(send_button)
        self.layout.addLayout(input_layout)

        self.status_label = QLabel("")
        self.layout.addWidget(self.status_label)

    def append_message(self, role, text):
        if role == "user":
            self.chat_display.append(
                f'<p style="color: blue; font-weight: bold;">You:</p><p>{text}</p>'
            )
        else:
            self.chat_display.append(
                f'<p style="color: green; font-weight: bold;">Assistant:</p><p>{text}</p>'
            )
        self.chat_display.verticalScrollBar().setValue(
            self.chat_display.verticalScrollBar().maximum()
        )

    def send_message(self):
        user_text = self.user_input.text().strip()
        if not user_text:
            return
        self.append_message("user", user_text)
        self.user_input.clear()

        self.status_label.setText("Assistant is typing...")
        self.user_input.setDisabled(True)

        self.thread = ApiThread(user_text)
        self.thread.response_ready.connect(self.receive_response)
        self.thread.start()

    def receive_response(self, response):
        if response.startswith("Error:"):
            QMessageBox.warning(self, "Error", response)
            self.append_message("assistant", "[Error receiving response]")
        else:
            self.append_message("assistant", response)
        self.status_label.setText("")
        self.user_input.setDisabled(False)
        self.user_input.setFocus()

def main():
    app = QApplication(sys.argv)
    window = ChatWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
