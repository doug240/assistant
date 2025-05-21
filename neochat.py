import sys
import requests
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QLineEdit,
    QPushButton, QLabel, QMessageBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

FLASK_API_URL = "http://localhost:5000/infer"

class ApiThread(QThread):
    response_ready = pyqtSignal(str)

    def __init__(self, conversation):
        super().__init__()
        self.conversation = conversation

    def run(self):
        try:
            response = requests.post(FLASK_API_URL, json={"messages": self.conversation})
            response.raise_for_status()
            data = response.json()
            if "response" in data:
                self.response_ready.emit(data["response"])
            else:
                self.response_ready.emit(f"Error: {data.get('error', 'Unknown error from API')}")
        except Exception as e:
            self.response_ready.emit(f"Error contacting API: {str(e)}")

class ChatWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Local Chat Model (API)")
        self.setGeometry(300, 300, 700, 500)

        self.conversation = []

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet("background-color: #f5f5f5; font-family: Consolas; font-size: 14px;")
        self.layout.addWidget(self.chat_display)

        input_layout = QHBoxLayout()
        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText("Type your message and press Enter...")
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
            self.chat_display.append(f'<b><span style="color: blue;">You:</span></b> {text}')
        else:
            self.chat_display.append(f'<b><span style="color: green;">Assistant:</span></b> {text}')
        self.chat_display.verticalScrollBar().setValue(self.chat_display.verticalScrollBar().maximum())

    def send_message(self):
        user_text = self.user_input.text().strip()
        if not user_text:
            return
        self.append_message("user", user_text)
        self.conversation.append({"role": "user", "content": user_text})
        self.user_input.clear()

        self.status_label.setText("Assistant is typing...")
        self.user_input.setDisabled(True)

        self.thread = ApiThread(self.conversation)
        self.thread.response_ready.connect(self.receive_response)
        self.thread.start()

    def receive_response(self, response):
        if response.startswith("Error:"):
            QMessageBox.warning(self, "Error", response)
            self.append_message("assistant", "[Error receiving response]")
        else:
            self.append_message("assistant", response)
            self.conversation.append({"role": "assistant", "content": response})
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
