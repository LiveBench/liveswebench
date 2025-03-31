import json
from typing import Any

from liveswebench.util.util import BASE_TASK_PATH, BASE_DATA_PATH
from liveswebench.util.tasks import TaskType


def collect_jsonl() -> None:
    """
    Collect contents of all task files into a single JSONL file.
    
    For each task directory at BASE_TASK_PATH/repo_name/task_num/, collect all files
    except problem_statement.md and agent_edit_prompt.txt. Use the filename as the key
    in the JSON object, and also include repo_name and task_num keys.
    
    The JSONL file is stored at the root of BASE_TASK_PATH.
    """
    output_file = BASE_TASK_PATH / "test.jsonl"
    
    # Ensure the parent directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        # Iterate through repo directories
        for repo_dir in BASE_TASK_PATH.iterdir():
            if not repo_dir.is_dir():
                continue
                
            repo_name = repo_dir.name
            
            # Iterate through task directories
            for task_dir in repo_dir.iterdir():
                if not task_dir.is_dir():
                    continue
                    
                task_num = int(task_dir.name)
                
                # Create JSON object for this task
                task_data: dict[str, Any] = {
                    "repo_name": repo_name,
                    "task_num": task_num
                }
                
                # Collect all files except problem_statement.md and agent_edit_prompt.txt
                for file_path in task_dir.iterdir():
                    if file_path.is_file() and file_path.name not in ["problem_statement.md", "agent_edit_prompt.txt", "classifications.json"]:
                        try:
                            # Use the filename without extension as the key in the JSON object
                            file_key = file_path.stem
                            task_data[file_key] = file_path.read_text(encoding="utf-8")
                        except Exception as e:
                            print(f"Error reading file {file_path}: {e}")
                
                # Write task data as JSON line to the output file
                f.write(json.dumps(task_data) + "\n")
    
    print(f"JSONL file created at {output_file}")


def collect_tool_patches() -> None:
    """
    Collect patch files for each tool and task type combination.
    
    For each tool that has generated patches, create a JSONL file for each task type
    (agent, edit, autocomplete) containing patch data. Each JSONL object will have
    repo_name, task_num, and patch fields.
    
    If multiple patch files exist for a combination of tool, task, and task type,
    use the one with the latest timestamp.
    
    The JSONL files are stored at the root of BASE_DATA_PATH with names like:
    <tool>_<task_type>_preds.jsonl (e.g., cursor_agent_preds.jsonl)
    
    Tool patch files are located at: BASE_DATA_PATH/<repo_name>/<task_num>/<tool_name>/
    """
    # Map to store (tool, task_type) -> list of objects to write to JSONL
    tool_task_type_data: dict[tuple[str, str], list[dict[str, Any]]] = {}
    
    # Iterate through repo directories in the results directory
    for repo_dir in BASE_DATA_PATH.iterdir():
        if not repo_dir.is_dir():
            continue
            
        repo_name = repo_dir.name
        
        # Iterate through task directories
        for task_dir in repo_dir.iterdir():
            if not task_dir.is_dir():
                continue
                
            try:
                task_num = int(task_dir.name)
            except ValueError:
                # Skip directories that are not task numbers
                continue
                
            # Iterate through tool directories
            for tool_dir in task_dir.iterdir():
                if not tool_dir.is_dir():
                    continue
                    
                tool_name = tool_dir.name
                
                # Find patch files for each task type (agent, edit, autocomplete)
                for task_type in [t.value for t in TaskType]:
                    # Look for patch files with the task type in the filename
                    patch_files = list(tool_dir.glob(f"*_{task_type}*.patch"))
                    
                    if not patch_files:
                        continue
                    
                    # Sort patch files by timestamp (assuming timestamp is in filename)
                    # This is a simple approach that assumes newer files are modified more recently
                    patch_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                    
                    # Get the latest patch file
                    latest_patch_file = patch_files[0]
                    
                    try:
                        # Read patch content
                        patch_content = latest_patch_file.read_text(encoding="utf-8", errors="replace")
                        
                        # Create tuple key for tool and task type
                        key = (tool_name, task_type)
                        if key not in tool_task_type_data:
                            tool_task_type_data[key] = []
                        
                        # Add data for this patch
                        tool_task_type_data[key].append({
                            "repo_name": repo_name,
                            "task_num": task_num,
                            "patch": patch_content
                        })
                    except Exception as e:
                        print(f"Error reading patch file {latest_patch_file}: {e}")
    
    # Write JSONL files for each tool/task type combination
    for (tool_name, task_type), data_list in tool_task_type_data.items():
        output_file = BASE_DATA_PATH / f"{tool_name}_{task_type}_preds.jsonl"
        
        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Write data to JSONL file
        with open(output_file, "w", encoding="utf-8") as f:
            for data in data_list:
                f.write(json.dumps(data) + "\n")
        
        print(f"Created {output_file} with {len(data_list)} entries")
    
    print(f"Tool patch JSONL files created at {BASE_DATA_PATH}")


if __name__ == "__main__":
    collect_jsonl()
    collect_tool_patches()
