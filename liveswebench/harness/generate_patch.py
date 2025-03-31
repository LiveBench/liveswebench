from liveswebench.harness.util import check_and_revert_patch
from liveswebench.util.util import TOOLS
from liveswebench.util.repo import get_repo
from liveswebench.util.tasks import get_patch_path, load_tasks, TaskInstance, TaskType, get_partial_gold_patch
import argparse
import git

def generate_patch(task: TaskInstance, tool_name: str, task_type: TaskType):

    print(f"Generating patch for {task.repo_name} task {task.task_num} for tool {tool_name}")

    repo = get_repo(task.repo_name)
    repo.clean_ignore()

    task_data_path = task.task_data_path / tool_name
    task_data_path.mkdir(parents=True, exist_ok=True)

    patch_file = get_patch_path(task, task_type, tool_name)

    if patch_file.exists():
        print(f"Patch file {patch_file} already exists, would you like to overwrite it? (y/n)")
        if input() != 'y':
            print("Aborting")
            exit()
    
    partial_gold_patch = get_partial_gold_patch(task, task_type)

    if partial_gold_patch.strip() != '':
        print("Reverting partial gold patch for task")
        try:
            check_and_revert_patch(partial_gold_patch, repo=repo)
        except git.GitCommandError:
            raise RuntimeError(f"Partial gold patch for task {task.task_num} for repository {task.repo_name} was not previously applied or could not be reverted")

    repo.git_add(".")
    diff = repo.git_diff("HEAD")
    if diff.strip() != "":
        with open(patch_file, "w", encoding="utf-8") as f:
            f.write(diff)
    else:
        print(f"No changes to commit for task {task.task_num} for repository {task.repo_name}")
        with open(patch_file, "w", encoding="utf-8") as f:
            f.write("")
    print(f"Generated patch file: {patch_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LiveSWEBench Task Preparation")
    parser.add_argument('--repo_name', type=str, help='Name of the repository')
    parser.add_argument('--task_num', type=int, help='Task number')
    parser.add_argument('--tool_name', type=str, help='Name of the tool', required=True, choices=TOOLS)
    parser.add_argument('--task_type', type=TaskType, help='Type of the task', choices=list(TaskType), default=TaskType.AGENT)
    parser.add_argument('--task_source', type=str, choices=['local', 'huggingface'], default='huggingface',
                      help='Source of tasks, either local or from huggingface')

    args = parser.parse_args()

    tasks = load_tasks(task_source=args.task_source)
    task = tasks[args.repo_name][args.task_num]
    generate_patch(task=task, tool_name=args.tool_name, task_type=args.task_type)