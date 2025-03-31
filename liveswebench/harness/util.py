from pathlib import Path
from liveswebench.util.util import execute_commands, execute_background_command_and_wait
import subprocess
import re
from liveswebench.util.repo import get_repo, Repo

def execute_test_command(command_config: list[str] | dict[str, list[str] | dict[str, list[str]]], repo_name: str, task_num: int, repo_path: str, out_file: str):
    """
    Execute test commands, either directly or with a server process.
    
    Args:
        command_config: Either a list of commands or a dict with server and test configurations
        repo_name: Name of the repository
        task_num: Task number
        repo_path: Path to the repository
        out_file: Output file for logs
    """
    # Handle the case when command_config is a dictionary (server-client pattern)
    if isinstance(command_config, dict):
        server_cmd = command_config["server"]["command"]
        server_ready_string = command_config["server"]["ready_string"]
        test_cmd = command_config["test"]
        
        # Apply pnpm version specific modifications
        if repo_name == 'freeCodeCamp' and task_num == 54128:
            server_cmd = server_cmd.replace("pnpm", "npx pnpm@9")
            test_cmd = [s.replace("pnpm", "npx pnpm@9") for s in test_cmd]
        elif repo_name == 'freeCodeCamp' and task_num <= 54812:
            server_cmd = server_cmd.replace("pnpm", "npx pnpm@8")
            test_cmd = [s.replace("pnpm", "npx pnpm@8") for s in test_cmd]
        
        # Start server, run tests, then terminate server
        server_process = execute_background_command_and_wait(server_cmd, server_ready_string, cwd=repo_path, out_file=out_file)
        execute_commands(test_cmd, cwd=repo_path, out_file=out_file)
        print("Terminating server")
        server_process.terminate()
        try:
            server_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server_process.kill()
            server_process.wait()
    # Handle the case when command_config is a list (direct commands)
    else:
        command = command_config
        if repo_name == 'freeCodeCamp' and task_num == 54128:
            command = [s.replace("pnpm", "npx pnpm@9") for s in command]
        elif repo_name == 'freeCodeCamp' and task_num <= 54812:
            command = [s.replace("pnpm", "npx pnpm@8") for s in command]
        execute_commands(command, cwd=repo_path, out_file=out_file)

def run_tests(repo_name: str, task_num: int, out_file: str):
    repo = get_repo(repo_name)
    task_path = repo.task_path / str(task_num)
    test_patch = open(task_path / "test_patch.patch", "r").read()

    pre_test_cmd = repo.pre_test_cmd
    if pre_test_cmd:
        print("Running pre-test commands...")
        if repo_name == 'freeCodeCamp' and task_num == 54128:
            pre_test_cmd = [s.replace("pnpm", "npx pnpm@9") for s in pre_test_cmd]
        elif repo_name == 'freeCodeCamp' and task_num <= 54812:
            pre_test_cmd = [s.replace("pnpm", "npx pnpm@8") for s in pre_test_cmd]
        execute_commands(pre_test_cmd, cwd=str(repo.repo_path), out_file=out_file)

    print("Running test commands...")

    if "default" in repo.test_cmd:
        print("Running default test commands...")
        execute_commands(repo.test_cmd["default"], cwd=repo.repo_path, out_file=out_file)

    for identifier_str in repo.test_cmd:
        if identifier_str != "default" and identifier_str in test_patch:
            print(f"Running test commands for {identifier_str}...")
            execute_test_command(repo.test_cmd[identifier_str], repo_name, task_num, repo.repo_path, out_file)
    
    if repo.test_regex_cmd:
        for regex in repo.test_regex_cmd:
            print(f"Running test commands for regex {regex}...")
            matches: list[str] = re.findall(regex, test_patch)
            if len(matches) > 0:
                groups: str = ' '.join(matches)
                if isinstance(repo.test_regex_cmd[regex], dict):
                    command_config = {}
                    command_config["server"] = repo.test_regex_cmd[regex]["server"]
                    command_config["test"] = [s.format(groups=groups) for s in repo.test_regex_cmd[regex]["test"]]
                    execute_test_command(command_config, repo_name, task_num, repo.repo_path, out_file)
                else:
                    test_cmd = [s.format(groups=groups) for s in repo.test_regex_cmd[regex]]
                    execute_test_command(test_cmd, repo_name, task_num, repo.repo_path, out_file)

def extract_hunks_from_patch(patch_content: str) -> dict[str, dict[str, str | bool | list[str]]]:
    """
    Extract all hunks from a git patch file without any filtering.
    
    Args:
        patch_content (str): Content of the git patch file
        
    Returns:
        dict: Maps file headers to dictionaries containing file_path, is_new_file, and hunks list
    """
    result: dict[str, dict[str, str | bool |list[str]]] = {}
    lines = patch_content.splitlines()

    i = 0
    while i < len(lines):
        # Find start of file section
        if not lines[i].startswith('diff --git'):
            i += 1
            continue

        # Extract file header
        file_header_start = i
        while i < len(lines) and not lines[i].startswith('@@'):
            i += 1

        if i >= len(lines):
            break

        file_header_end = i
        file_header = '\n'.join(lines[file_header_start:file_header_end])
        
        # Extract file path and new file status for metadata
        file_path = None
        is_new_file = False
        is_renamed_file = False
        for line in lines[file_header_start:file_header_end]:
            if line.startswith('--- /dev/null'):
                is_new_file = True
            if line.startswith('+++ b/'):
                file_path = line[6:]
                break
            if line.startswith('rename from '):
                is_renamed_file = True

        if file_path is None:
            raise ValueError(f"File path not found in patch for file header: {file_header}")
        
        # Process all hunks for this file without filtering
        hunks: list[str] = []

        while i < len(lines) and not lines[i].startswith('diff --git'):
            if lines[i].startswith('@@'):
                hunk_start = i

                # Find end of the hunk
                i += 1
                while i < len(lines) and not lines[i].startswith('@@') and not lines[i].startswith('diff --git'):
                    i += 1

                hunk_text = '\n'.join(lines[hunk_start:i])
                hunks.append(hunk_text)
            else:
                i += 1

        # Add to results if there are any hunks
        if hunks:
            result[file_header] = {
                "file_path": file_path,
                "is_new_file": is_new_file,
                "is_renamed_file": is_renamed_file,
                "hunks": hunks
            }
    return result

def construct_partial_patch(original_patch: Path | str, exclude_patch: Path | str) -> str | None:
    """
    Construct a partial patch from an original patch and an exclude patch.
    The exclude patch is a patch that contains the changes to be excluded from the original patch.

    original_patch and exclude_patch are either Path objects, pointing to patch files, or strings containing the patch content.
    """
    if isinstance(original_patch, Path):
        if not original_patch.exists():
            print(f"Original patch file {original_patch} does not exist")
            return None
        original_patch = open(original_patch, "r", encoding="utf-8").read()
    if isinstance(exclude_patch, Path):
        if not exclude_patch.exists():
            print(f"Exclude patch file {exclude_patch} does not exist")
            return None
        exclude_patch = open(exclude_patch, "r", encoding="utf-8").read()

    original_patch_hunks = extract_hunks_from_patch(original_patch)
    exclude_patch_hunks = extract_hunks_from_patch(exclude_patch)

    for file_header, file_info in original_patch_hunks.items():
        if file_header in exclude_patch_hunks:
            exclude_hunks = [hunk.strip() for hunk in exclude_patch_hunks[file_header]["hunks"]]
            file_info["hunks"] = [hunk for hunk in file_info["hunks"] if hunk.strip() not in exclude_hunks]

    partial_patch = ""
    for file_header, file_info in original_patch_hunks.items():
        if len(file_info["hunks"]) > 0:
            partial_patch += file_header
            if not partial_patch.endswith("\n"):
                partial_patch += "\n"
            for hunk in file_info["hunks"]:
                partial_patch += hunk
                if not partial_patch.endswith("\n"):
                    partial_patch += "\n"

    return partial_patch

def filter_patch_by_operation(patch_content: str, operation: str) -> str:
    """
    Takes a git patch and an operation ('+' or '-'), and returns a new patch
    containing only changes of the specified type.

    Args:
        patch_content (str): The content of the git patch
        operation (str): Either '+' or '-' to indicate additions or deletions

    Returns:
        str: A new patch with only the specified operation changes
    """
    import re

    if operation not in ['+', '-']:
        raise ValueError("Operation must be either '+' or '-'")

    # Parse the patch using the provided function
    file_patches = extract_hunks_from_patch(patch_content)

    # Build the new patch
    new_patch_lines: list[str] = []

    for file_header, file_info in file_patches.items():
        # Skip files with no relevant operations
        if not any(any(line.startswith(operation) for line in hunk.split('\n'))
                  for hunk in file_info['hunks']):
            continue

        # Add file header to new patch
        new_patch_lines.append(file_header)

        for hunk in file_info['hunks']:
            hunk_lines = hunk.split('\n')

            # Skip hunks without our operation
            if not any(line.startswith(operation) for line in hunk_lines):
                continue

            # Parse the hunk header
            header = hunk_lines[0]
            match = re.match(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$', header)
            if not match:
                continue

            old_start = int(match.group(1))
            new_start = int(match.group(3))
            header_tail = match.group(5)

            # Filter lines to keep context and our operation
            filtered_lines: list[str] = []
            for line in hunk_lines[1:]:
                if not line:
                    continue
                if not line.startswith('+') and not line.startswith('-'):  # Context line
                    filtered_lines.append(line)
                elif line.startswith(operation):  # Our operation
                    filtered_lines.append(line)

            # Count lines by type
            context_count = sum(1 for line in filtered_lines if not line.startswith('+') and not line.startswith('-'))
            operation_count = sum(1 for line in filtered_lines if line.startswith(operation))

            # Calculate new header values based on operation type
            if operation == '+':
                # For additions: old file has context lines, new file has context + additions
                new_old_count = context_count
                new_new_count = context_count + operation_count
            else:  # operation == '-'
                # For deletions: old file has context + deletions, new file has context
                new_old_count = context_count + operation_count
                new_new_count = context_count

            # Create new header
            new_old_range = f"{old_start},{new_old_count}" if new_old_count != 1 else str(old_start)
            new_new_range = f"{new_start},{new_new_count}" if new_new_count != 1 else str(new_start)
            new_header = f"@@ -{new_old_range} +{new_new_range} @@{header_tail}"

            # Add the new hunk to the patch
            new_patch_lines.append(new_header)
            new_patch_lines.extend(filtered_lines)

    return '\n'.join(new_patch_lines) + '\n'

def check_and_revert_patch(patch_content: str, repo: Repo) -> None:
    """
    Check if a patch has been applied and revert it if it has.
    
    Args:
        patch_content: String content of the patch
        repo: Repo object to use for checking and reverting the patch
    """
    print("Checking if patch has been applied...")
    # Check if patch can be reversed (meaning it was already applied)
    try:
        repo.apply_patch(patch_content, '--reverse', '--ignore-whitespace', '--check')
    except:
        print("Patch was not previously applied, skipping revert...")
        return
    
    print("Patch was previously applied, now reverting...")
    # Patch was previously applied, now revert it
    repo.apply_patch(patch_content, '--reverse', '--ignore-whitespace')