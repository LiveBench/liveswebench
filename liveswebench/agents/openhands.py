from datetime import datetime
import os


def get_openhands_command(prompt_file: str, workspace_base: str, llm_model: str, llm_api_key: str, files_to_edit: list[str] = []) -> list[str]:
    """
    Get OpenHands Docker command for specified parameters.
    
    Args:
        workspace_base (str): Path to workspace base directory
        llm_model (str): LLM model identifier
        llm_api_key (str): API key for LLM

    """
    # Get current timestamp for container name
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    
    # Get current user ID
    user_id = os.getuid()

    prompt = open(prompt_file, 'r').read()

    llm_model = 'anthropic/' + llm_model if 'claude' in llm_model else llm_model
    
    # Construct the docker command
    docker_command = [
        "docker", "run", "-it",
        "--pull=always",
        f"-e", f"SANDBOX_RUNTIME_CONTAINER_IMAGE=docker.all-hands.dev/all-hands-ai/runtime:0.28-nikolaik",
        f"-e", f"SANDBOX_USER_ID={user_id}",
        f"-e", f"WORKSPACE_MOUNT_PATH={workspace_base}",
        f"-e", f"LLM_API_KEY={llm_api_key}",
        f"-e", f"LLM_MODEL={llm_model}",
        f"-e", "LOG_ALL_EVENTS=true",
        f"-v", f"{workspace_base}:/opt/workspace_base",
        f"-v", "/var/run/docker.sock:/var/run/docker.sock",
        f"-v", f"{os.path.expanduser('~')}/.openhands-state:/.openhands-state",
        "--add-host", "host.docker.internal:host-gateway",
        "--name", f"openhands-app-{timestamp}",
        "docker.all-hands.dev/all-hands-ai/openhands:0.28",
        "python", "-m", "openhands.core.main", "-t", f"\"{prompt}\""
    ]

    return docker_command, workspace_base