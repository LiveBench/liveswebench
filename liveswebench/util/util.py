from pathlib import Path
import os
import subprocess
import time
import socket

BASE_REPO_PATH = Path(__file__).parent.parent / "repos"
BASE_TASK_PATH = Path(__file__).parent.parent / "data/tasks"
BASE_DATA_PATH = Path(__file__).parent.parent / "data/results"

TOOLS = [
    'cursor',
    'windsurf',
    'github-copilot',
    'navie',
    'codellm',
    'aider',
    'openhands',
    'claude-code',
    'gemini-code-assist',
    'amazon-q',
    'swe-agent'
]

def check_port(port, timeout=30):
    print(f"Checking port {port}")
    time.sleep(1)
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                s.connect(('localhost', port))
                return True
        except (ConnectionRefusedError, socket.timeout):
            time.sleep(0.5)
    return False

def execute_background_command_and_wait(command, wait_string, cwd=None, out_file = None):
    cwd = os.path.join(os.getcwd(), cwd) if cwd is not None else os.getcwd()
    command = "(" + command +  ") 2>&1"
    if out_file:
        command = command + f" | tee -a {out_file}"

    command = "bash -c \"" + command + "\""

    print(f"Executing background command: {command} with cwd: {cwd}")
    process = subprocess.Popen(
        command, 
        cwd=cwd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        shell=True)
    if process.stdout is None:
        raise RuntimeError("Failed to capture process output.")

    print(f"Waiting for ready string: {wait_string}")
    started = False
    start_time = time.time()
    timeout = 300 # seconds
    
    while time.time() - start_time < timeout:
        line = process.stdout.readline()
        if not line:
            time.sleep(0.1)
            continue
            
        line = line.decode("utf-8", errors="replace")
        print("Server Output:", line)
        if wait_string in line:
            print("Found server ready string")
            started = True
            break
            
    if time.time() - start_time >= timeout:
        process.kill()
        raise RuntimeError(f"Server startup timed out after {timeout} seconds: command was {command}")
    if not started:
        raise RuntimeError(f"Server failed to start: command was {command}")
    # check that server process is still alive
    if process.poll() is not None:
        raise RuntimeError(f"Server process terminated unexpectedly: command was {command}")
    
    print("Server started")
    return process


def execute_commands(commands: str | list[str], cwd: str | None=None, output_to_terminal: bool = True, exit_on_fail: bool = False, out_file: str | None = None, no_bash: bool = False):
    if isinstance(commands, str):
        commands = [commands]
    
    if all(isinstance(cmd, str) for cmd in commands):
        commands = [" && ".join(commands)]
    else:
        commands = [" && ".join(cmd) for cmd in commands]

    cwd = os.path.join(os.getcwd(), cwd)

    if len(commands) > 1:
        print("Executing multiple commands:")
        print("\n".join(commands))

    results = []

    for command in commands:

        if not no_bash:
            command = "(" + command +  ") 2>&1"

        if out_file:
            command = command + f" | tee -a {out_file}"

        if not no_bash:
            command = "bash -l -c \"" + command + "\""

        print(f"Executing command: {command} with cwd: {cwd}")

        res = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            text=False,
            executable="/bin/bash",
            capture_output=(not output_to_terminal),
        )

        if res.returncode != 0 and exit_on_fail:
            if res.stdout:
                stdout = res.stdout.decode("utf-8", errors="replace")
                print(f"Error: {stdout}")
            raise RuntimeError(f"Command failed: {command}")

        print("Command completed")
        results.append(res)

    return results