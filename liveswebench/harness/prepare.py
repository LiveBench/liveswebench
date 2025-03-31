from liveswebench.harness.util import filter_patch_by_operation, extract_hunks_from_patch
from liveswebench.util.repo import get_repo
from liveswebench.util.util import execute_commands, TOOLS
from liveswebench.util.tasks import TaskInstance, get_partial_gold_patch, get_relevant_files_for_task, get_removal_patch_for_task, load_tasks, TaskType

import argparse


def prepare_task(task: TaskInstance, install: bool = False, print_prompt: bool = False, task_type: TaskType = TaskType.AGENT, tool_name: str | None = None, test: bool = False, skip_ignore: bool = False, skip_removals: bool = False):
    print(f"Preparing task {task.task_num} for repository {task.repo_name}")

    repo = get_repo(task.repo_name)

    repo.clean(write_ignore=False)

    # Checkout commit and create/switch to task branch
    branch_name = f"task_{task.task_num}"

    print(f"Checking out commit {task.base_commit} and creating branch {branch_name}")

    if branch_name in repo.git_repo.heads:
        repo.git_checkout(branch_name)
    else:
        repo.git_checkout(task.base_commit)
        repo.git_checkout("-b", branch_name)

    if not skip_ignore:
        repo.write_ignore()

    repo_path = repo.repo_path
            
    partial_gold_patch = get_partial_gold_patch(task, task_type)
    if partial_gold_patch.strip() != '':
        print("Applying partial gold patch for task")
        try:
            repo.apply_patch(partial_gold_patch, '--ignore-whitespace')
        except Exception as e:
            print(f"Partial gold patch failed to apply:\n{partial_gold_patch}")
            raise e

    if task_type == TaskType.EDIT:
        # TODO: recreate edit patches so that they include file creation already
        edit_patch = task.get_ground_truth_patch(task_type)
        relevant_files = get_relevant_files_for_task(task, task_type)
        if relevant_files is None or len(relevant_files) == 0:
            raise RuntimeError(f"No relevant files found for task {task.task_num} in repository {task.repo_name}")
        edit_file = relevant_files[0]
        
        if '--- /dev/null' in edit_patch and not test: 
            # edit patch involves creating a new file
            # we need to create the file first when not testing
            # when testing, the tool patch will include the creation of the file
            with open(repo_path / edit_file, 'w') as f:
                pass
            print(f"Created new file {edit_file}")
    
    removal_patch = get_removal_patch_for_task(task, task_type)
    if removal_patch is not None and not test and not skip_removals:
        # apply the removals (if we're not testing)
        # if testing, the tool patch will include the removals
        if removal_patch.strip() != '':
            print("Applying removal patch for task")
            try:
                repo.apply_patch(removal_patch, '--ignore-whitespace')
            except Exception as e:
                print(f"Failed to apply removal patch:\n{removal_patch}")
                raise e
        else:
            print("Removal patch would be empty, skipping")
        

    # Execute install commands
    if install and repo.install_cmd is not None:
        print(f"Executing install commands for repository {repo.name}")

        cmd = repo.install_cmd

        if repo.name == 'torchtune' and task.task_num < 1591:

            cmd = [s.replace("torchao", "torchao==0.5.0") for s in cmd]
            cmd = [s.replace(" torch ", " \"torch==2.4\" ") for s in cmd]
        elif repo.name == 'torchtune' and task.task_num < 1909:
            cmd = [s.replace("torchao", "torchao==0.5.0") for s in cmd]
        elif repo.name == 'freeCodeCamp' and task.task_num == 54128:
            cmd = [s.replace("pnpm", "npx pnpm@9") for s in cmd]
        elif repo.name == 'freeCodeCamp' and task.task_num <= 54812:
            cmd = [s.replace("pnpm", "npx pnpm@8") for s in cmd]
            #cmd = ["npm install -g pnpm@8"] + cmd

        execute_commands(
            cmd,
            cwd=str(repo_path),
            output_to_terminal=True,
            exit_on_fail=True,
        )

    if print_prompt:
        print('--------------------------------')
        print('TASK PROMPT:')
        
        prompt = task.get_prompt(task_type)
        if task_type == TaskType.EDIT:
            prompt += '\nOnly changes in one file are needed. Do not make changes to other files. Do not make changes to test files.'
        if task_type == TaskType.AGENT or task_type == TaskType.EDIT:
            print(prompt)
            
        if task_type == TaskType.EDIT:
            relevant_files = get_relevant_files_for_task(task, task_type)
            if relevant_files is None or len(relevant_files) == 0:
                raise RuntimeError(f"No relevant files found for task {task.task_num} in repository {task.repo_name}")
            edit_file = relevant_files[0]
            print('--------------------------------')
            print('EDIT FILE:')
            print(repo.repo_path / edit_file)

        if task_type == TaskType.AUTOCOMPLETE:
            autocomplete_prompt_hunks = extract_hunks_from_patch(prompt)
            
            autocomplete_patch = task.get_ground_truth_patch(task_type)
            if not skip_removals:
                autocomplete_patch = filter_patch_by_operation(autocomplete_patch, operation='+')
            else:
                autocomplete_patch = autocomplete_patch
                
            autocomplete_patch_hunks = extract_hunks_from_patch(autocomplete_patch)
            for file_header, file_info in autocomplete_prompt_hunks.items():
                assert file_header in autocomplete_patch_hunks

                for addition_hunk, hunk_prompt in zip(autocomplete_patch_hunks[file_header]['hunks'], file_info['hunks']):
                    print('--------------------------------')
                    print('FILE:')
                    file_path = file_info['file_path']
                    print((repo.repo_path / file_path).resolve())
                    hunk_header = addition_hunk.split('\n')[0]
                    new_line_number = int(hunk_header.split(' ')[2].split(',')[0].replace('+', ''))
                    print(f'ORIGINAL ADDITION (Near line {new_line_number}):')
                    print('\n'.join(addition_hunk.split('\n')[1:]))
                    print('PROMPT:')
                    hunk_prompt = '\n'.join(hunk_prompt.split('\n')[1:]) # remove hunk header from prompt
                    
                    # Format each line with the appropriate comment syntax
                    formatted_prompt = '\n'.join([
                        f"{line}" if line.strip() else line 
                        for line in hunk_prompt.strip().split('\n')
                    ])
                    
                    print(formatted_prompt)

        print('--------------------------------')

    if tool_name is not None:
        tool_name_command_mapping = {
            'cursor': 'cursor',
            'github-copilot': 'code-insiders',
            'amazon-q': 'code',
            'windsurf': 'windsurf',
            'codellm': 'codellm',
            'navie': 'code',
            'gemini-code-assist': 'code'
        }
        if tool_name not in tool_name_command_mapping:
            raise ValueError(f"Tool name {tool_name} not supported")
        tool_command = tool_name_command_mapping[tool_name]
        command = f"vscli open --behavior force-classic --command {tool_command} {repo_path}"
        print(f"Setting up {tool_name} for repository {repo.name}")
        try:
            import subprocess
            
            # Run the command on any operating system
            process = subprocess.Popen(
                command,
                shell=True,
                cwd=repo_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                universal_newlines=True
            )
            stdout, stderr = process.communicate()
            
            # Print output to terminal
            if stdout:
                print(stdout)
            if stderr:
                print(stderr)
                
            if process.returncode != 0:
                print(f"Command exited with non-zero status: {process.returncode}")
        except Exception as e:
            print("Couldn't automatically open the tool, please open it manually")
            return
        
        relevant_files = get_relevant_files_for_task(task, task_type)
        if relevant_files is not None and len(relevant_files) > 0:
            print(f"Opening relevant files in {tool_name}")
            for file in relevant_files:
                command = f"vscli open --behavior force-classic --command {tool_command} {repo.repo_path / file} -- -r" # -r opens the file in the window that was opened in the previous command
                try:
                    # Run the command on any operating system
                    process = subprocess.Popen(
                        command,
                        shell=True,
                        cwd=repo_path,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        universal_newlines=True
                    )
                    stdout, stderr = process.communicate()
                    
                    # Print output to terminal
                    if stdout:
                        print(stdout)
                    if stderr:
                        print(stderr)
                        
                    if process.returncode != 0:
                        print(f"Command exited with non-zero status: {process.returncode}")
                except Exception as e:
                    print(f"Couldn't open file {file} in {tool_name}, please open it manually")

    print(f"Successfully prepared task {task.task_num} for repository {repo.name}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LiveSWEBench Task Preparation")
    parser.add_argument('--repo_name', type=str, help='Name of the repository', required=True)
    parser.add_argument('--task_num', type=int, help='Task number', required=True)
    parser.add_argument('--install', action='store_true', help='Whether to install repository dependencies after preparing')
    parser.add_argument('--print-prompt', action='store_true', help='Whether to print the prompt for the task after preparing')
    parser.add_argument('--task_type', help='Type of task to prepare', choices=list(TaskType), default=TaskType.AGENT, type=TaskType)
    parser.add_argument('--tool_name', type=str, help='Name of the tool to prepare', default=None, choices=TOOLS)
    parser.add_argument('--skip_removals', action='store_true', help='Whether to skip the removal of the removals in the autocomplete patch')
    parser.add_argument('--task_source', type=str, choices=['local', 'huggingface'], default='huggingface',
                      help='Source of tasks, either local or from huggingface')

    args = parser.parse_args()

    tasks = load_tasks(task_source=args.task_source)
    task = tasks[args.repo_name][args.task_num]
    prepare_task(task=task, install=args.install, print_prompt=args.print_prompt, task_type=args.task_type, tool_name=args.tool_name, skip_removals=args.skip_removals)