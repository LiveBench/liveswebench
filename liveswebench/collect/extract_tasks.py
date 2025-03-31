from pathlib import Path
import json
import os

PROMPT_TEMPLATE = """The following is the text of a github issue and related comments for this repository:

{problem_statement}

Resolve the issue described above. Do not modify tests or user code. The issue can be resolved by modifying the actual library/application code. Make the minimal changes to resolve the issue while adhering to the specification and coding style of the codebase."""

def load_json_or_jsonl(file_path):
    """Load either JSON file containing a list or JSONL file with one object per line."""
    with open(file_path) as f:
        # Try reading as regular JSON first
        try:
            data = json.load(f)
            # If it's not a list, wrap it in a list
            return data if isinstance(data, list) else [data]
        except json.JSONDecodeError:
            # If regular JSON parsing fails, try JSONL
            f.seek(0)  # Reset file pointer to beginning
            return [json.loads(line) for line in f if line.strip()]

def extract_tasks(source_task_instances_path, dest_dir):
    task_instances = load_json_or_jsonl(source_task_instances_path)

    os.makedirs(dest_dir, exist_ok=True)

    for task in task_instances:
        instance_id = task['instance_id']
        base_commit = task['base_commit']

        instance_path = os.path.join(dest_dir, instance_id.split('-')[-1])
        if os.path.exists(instance_path):
            print(f"Instance {instance_id} already exists. Skipping.")
            continue

        os.makedirs(instance_path, exist_ok=False)

        instance_path = Path(instance_path)

        with open(instance_path / 'commit.txt', 'w') as f:
            f.write(base_commit)

        with open(instance_path / 'gold_patch.patch', 'w') as f:
            f.write(task['patch'])

        with open(instance_path / 'test_patch.patch', 'w') as f:
            f.write(task['test_patch'])

        with open(instance_path / 'problem_statement.md', 'w') as f:
            f.write(task['problem_statement'])

        prompt = PROMPT_TEMPLATE.format(problem_statement=task['problem_statement'])
        with open(instance_path / "prompt.md", "w") as f:
            f.write(prompt)