import os
import argparse
import glob
import shutil
import traceback
from dotenv import load_dotenv

from liveswebench.agents.aider import get_aider_command
from liveswebench.agents.openhands import get_openhands_command
from liveswebench.agents.claude_code.claude_code import get_claude_code_command
from liveswebench.agents.swe_agent import get_swe_agent_command
from liveswebench.harness.prepare import prepare_task
from liveswebench.harness.generate_patch import generate_patch
from liveswebench.util.repo import get_all_repos, get_repo
from liveswebench.util.util import execute_commands
from liveswebench.util.tasks import TaskType, load_tasks, TaskInstance, get_relevant_files_for_task

load_dotenv()

AGENT_COMMANDS = {
    'openhands': get_openhands_command,
    'aider': get_aider_command,
    'claude-code': get_claude_code_command,
    'swe-agent': get_swe_agent_command
}

# Valid task types for agents
VALID_AGENT_TASK_TYPES = [TaskType.AGENT, TaskType.EDIT]

def test_task_with_agent(agent_name: str, task: TaskInstance, task_type: TaskType=TaskType.AGENT, llm_model: str='claude-3-7-sonnet-20250219'):
    # Validate task type is appropriate for agents
    if task_type not in VALID_AGENT_TASK_TYPES:
        raise ValueError(f"Task type {task_type} is not valid for agents. Valid types are: {VALID_AGENT_TASK_TYPES}")
        
    
    # Prepare the task
    prepare_task(task=task, install=False, task_type=task_type)
    
    # Get the repository
    repo = get_repo(task.repo_name)

    did_commit = False
    if agent_name == 'swe-agent':
        # swe-agent needs the repo state to be clean before running, so we commit all the changes so far
        if repo.git_repo.is_dirty():
            print("Repository is dirty, committing all changes")
            repo.git_add(".")
            repo.git_repo.git.commit("-m", "Adding all files to the repository")
            did_commit = True

    # Get the prompt and relevant files from the task object
    prompt = task.get_prompt(task_type)
    relevant_files = get_relevant_files_for_task(task, task_type) or []
    
    # Create a temporary prompt file if needed for EDIT task type
    prompt_file_path: str | None = None
    if task_type == TaskType.EDIT and relevant_files:
        prompt += f'\nYou will be editing the following files: {relevant_files}. No other files should be modified.'
        prompt += '\nDo not commit any changes to the repository.'
        
        agent_edit_prompt_file = task.task_data_path / "agent_edit_prompt.txt"
        with open(agent_edit_prompt_file, 'w') as f:
            f.write(prompt)
        prompt_file_path = str(agent_edit_prompt_file)
    else:
        # For other task types, use the task's prompt path
        if task_type == TaskType.AGENT:
            prompt_file = task.task_data_path / "prompt.md"
            if prompt_file.exists():
                prompt_file_path = str(prompt_file)

    # Get API key for the agent
    api_key_env = os.getenv('ANTHROPIC_API_KEY')
    if not api_key_env and (agent_name == 'openhands' or agent_name == 'claude-code'):
        raise ValueError(f"ANTHROPIC_API_KEY environment variable must be set for {agent_name}")
    
    # Ensure api_key is str type
    api_key: str = api_key_env if api_key_env is not None else ""
    
    # Ensure relevant_files is list[str] type
    files_to_edit: list[str] = relevant_files

    print(f"Launching {agent_name}")
    
    # Prepare the command arguments with proper types
    command_args: dict[str, str | list[str]] = {
        'workspace_base': str(repo.repo_path.resolve()),
        'llm_model': llm_model,
        'llm_api_key': api_key,
        'files_to_edit': files_to_edit,
    }
    
    if prompt_file_path:
        command_args['prompt_file'] = prompt_file_path

    # Get the command to run the agent
    command, cwd = AGENT_COMMANDS[agent_name](**command_args)

    command_list = command if isinstance(command, list) else [command]
    print(f"Running command: {' '.join(command_list)}")
    print(f"Working directory: {cwd}")

    command_str = ' '.join(command_list)

    try:
        # Run the command
        result = execute_commands(
            command_str, 
            cwd=str(cwd), 
            exit_on_fail=True, 
            no_bash=True if agent_name == 'openhands' or agent_name == 'claude-code' else False
        )[0]
    except Exception as e:
        print(f"Error running command: {e}")
        return

    if not result.returncode == 0:
        raise RuntimeError(f"Agent process failed with return code: {result.returncode}")

    print(f"Agent process completed successfully for task {task.task_num}")

    # Clean up aider files if needed
    if agent_name == 'aider':
        aider_files = glob.glob(str(repo.repo_path.resolve() / '.aider.*'))
        # Remove each file
        for path in aider_files:
            try:
                if os.path.isfile(path):
                    os.remove(path)
                elif os.path.isdir(path):
                    shutil.rmtree(path)
                print(f"Removed: {path}")
            except OSError as e:
                print(f"Error removing {path}: {e}")

    # Handle git state for swe-agent
    if agent_name == 'swe-agent' and did_commit:
        # generate_patch assumes that the previous commit didn't happen, so we need to undo it
        repo.git_repo.git.reset("HEAD~")

    if (task.task_data_path / "agent_edit_prompt.txt").exists():
        (task.task_data_path / "agent_edit_prompt.txt").unlink()

    # Generate the patch file for the agent's changes
    generate_patch(task, agent_name, task_type)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent_name', type=str, required=True, choices=AGENT_COMMANDS.keys())
    parser.add_argument('--repo_name', type=str, default=None)
    parser.add_argument('--task_nums', type=int, nargs='+', default=None)
    parser.add_argument('--redo-patches', action='store_true', default=False)
    parser.add_argument('--task_type', type=str, default=TaskType.AGENT.value, 
                       choices=[t.value for t in VALID_AGENT_TASK_TYPES])
    parser.add_argument('--model', type=str, default='claude-3-7-sonnet-20250219')
    args = parser.parse_args()

    # If repo_name is not provided, task_nums should also not be provided
    if args.repo_name is None and args.task_nums is not None:
        raise ValueError("task_nums cannot be specified when repo_name is not provided")

    if args.repo_name is None:
        # Process all repositories
        repos = [repo.name for repo in get_all_repos()]
        repos.sort(reverse=True)
        print(f"Processing all repositories: {repos}")
        for repo_name in repos:
            try:
                process_repo(repo_name, args.agent_name, args.task_nums, args.task_type, args.redo_patches, args.model)
            except Exception as e:
                print(f"An error occurred processing repo {repo_name}: {e}")
                traceback.print_exc()
    else:
        process_repo(args.repo_name, args.agent_name, args.task_nums, args.task_type, args.redo_patches, args.model)

def process_repo(repo_name: str, agent_name: str, task_nums: list[int] | None = None, 
                task_type_str: str = TaskType.AGENT.value, redo_patches: bool = False, 
                llm_model: str = 'claude-3-7-sonnet-20250219'):
    """
    Process a specific repository's tasks.
    
    Args:
        repo_name: Name of the repository to process
        agent_name: Name of the agent to run
        task_nums: Specific task numbers to run, or None to run all
        task_type_str: Task type as string (from TaskType enum)
        redo_patches: Whether to regenerate existing patches
        llm_model: LLM model to use
    """
    # Convert task_type string to TaskType enum
    task_type = TaskType(task_type_str)
    
    # Validate task type is valid for agents
    if task_type not in VALID_AGENT_TASK_TYPES:
        raise ValueError(f"Task type {task_type} is not valid for agents. Valid types are: {VALID_AGENT_TASK_TYPES}")
    
    # Load all tasks
    all_tasks = load_tasks()
    
    # Validate repository exists
    if repo_name not in all_tasks:
        raise ValueError(f"Repository {repo_name} not found")
    
    # Get tasks for this repository
    repo_tasks = all_tasks[repo_name]
    
    # Filter tasks if specific task numbers were requested
    if task_nums:
        filtered_tasks = {task_num: repo_tasks[task_num] for task_num in task_nums if task_num in repo_tasks}
        if not filtered_tasks:
            print(f"No tasks found for repository {repo_name} with task numbers {task_nums}")
            return
        repo_tasks = filtered_tasks
    
    # Sort tasks by task number (newest first)
    task_nums_sorted = sorted(repo_tasks.keys(), reverse=True)
    
    print(f"Found {len(task_nums_sorted)} tasks for repository {repo_name}")
    tasks_to_run: list[TaskInstance] = []
    
    # Determine which tasks need to be run
    for task_num in task_nums_sorted:
        task_instance = repo_tasks[task_num]
        
        # Check if a patch already exists
        tool_dir = task_instance.task_data_path / agent_name
        tool_dir.mkdir(parents=True, exist_ok=True)
        
        patch_files = list(tool_dir.glob(f"*_{task_type}_patch_*.patch"))
        if patch_files and not redo_patches:
            print(f"Skipping task {task_num} ({task_type}) - patch already exists")
            continue
            
        tasks_to_run.append(task_instance)
    
    print(f"Running {len(tasks_to_run)} tasks for repository {repo_name}")
    
    # Run each selected task
    for task_instance in tasks_to_run:
        try:
            print(f"Running task {task_instance.task_num} with type {task_type}")
            test_task_with_agent(agent_name, task_instance, task_type=task_type, llm_model=llm_model)
        except Exception as e:
            print(f"Error running task {task_instance.task_num} with type {task_type}: {e}")
            traceback.print_exc()

if __name__ == '__main__':
    main()
