import sys
from mode_controller import get_mode
from workflow_executor import execute_task
from interaction_logger import log_interaction
from feedback_manager import ask_for_feedback
from jupyter_interface import run_jupyter_code
from cli_runner import run_command

def assistant_loop():
    print("Assistant is running. Type 'exit' to quit.")
    while True:
        user_input = input(">>> ")
        if user_input.strip().lower() == "exit":
            break

        mode = get_mode()
        output = ""

        if user_input.startswith("!cli "):
            cmd = user_input[len("!cli "):]
            output = run_command(cmd)
        elif user_input.startswith("!code "):
            code = user_input[len("!code "):]
            output = run_jupyter_code(code)
        else:
            output = execute_task(user_input)

        log_interaction(user_input, output, metadata={"mode": mode})
        ask_for_feedback(output)

if __name__ == "__main__":
    try:
        assistant_loop()
    except KeyboardInterrupt:
        sys.exit(0)
