import argparse
import os
import traceback
from pathlib import Path
from typing import Literal

from liveswebench.harness.prepare import prepare_task
from liveswebench.harness.util import check_and_revert_patch, run_tests
from liveswebench.util.repo import get_repo
from liveswebench.util.tasks import find_task_patches, get_log_path, load_tasks, TaskInstance, TaskType


def patch_and_test(task: TaskInstance, tool_name: str, patch_file: str | Path | None = None, task_type: TaskType = TaskType.AGENT, skip_test_patch: bool = False, 
                  out_file_name: str | Path | None = None):
    """Test a specific patch for a task."""
    print(f"Testing task {task.task_num} for repository {task.repo_name} for tool {tool_name}" + 
          (f" with patch file {patch_file}" if patch_file else ""))

    if patch_file and not Path(patch_file).exists():
        raise ValueError(f"Patch file {patch_file} does not exist")

    # Get task instance
    repo = get_repo(task.repo_name)

    # check that repo is on the correct branch (task_{task_num})
    if repo.git_repo.active_branch.name != f"task_{task.task_num}":
        raise RuntimeError(f"Repository {task.repo_name} is not on the correct branch (task_{task.task_num}). Make sure to run prepare_task() before running test_task()")

    if not skip_test_patch:
        # Revert test patch if previously applied
        check_and_revert_patch(task.test_patch, repo=repo)

        # Apply test patch
        print(f"Applying test patch: {task.test_patch}")
        try:
            repo.apply_patch(task.test_patch)
        except Exception as e:
            print(f"Could not apply test patch: {e}")
            raise

    # Generate output file path
    if out_file_name:
        # Use provided output file name
        out_file = out_file_name if isinstance(out_file_name, Path) else Path(out_file_name)
    else:
        # Generate output file name based on patch file
        if patch_file:
            patch_path = Path(patch_file)
            patch_timestamp = patch_path.name.split('_')[-1].replace('.patch', '')
            out_file_name = get_log_path(task, task_type, tool_name, timestamp=patch_timestamp)
        else:
            # Default file name if no patch is provided
            out_file_name = get_log_path(task, task_type, tool_name, omit_timestamp=True)
        
        tool_dir = task.task_data_path / tool_name
        tool_dir.mkdir(parents=True, exist_ok=True)
        out_file = tool_dir / out_file_name

    if out_file.exists():
        print(f"Removing existing log file: {out_file}")
        os.remove(out_file)

    # Apply solution patch if provided
    if patch_file:
        print(f"Applying solution patch: {patch_file}")
        try:
            repo.apply_patch(Path(patch_file).resolve())
        except Exception as e:
            with open(patch_file, "a") as f:
                f.write("\n")
            try:
                repo.apply_patch(Path(patch_file).resolve())
            except Exception as e:
                print(f"Patch apply error: {e}")
                try:
                    repo.apply_patch(Path(patch_file).resolve(), '--reject')
                    print(f"Patch with --reject applied successfully")
                except Exception as e:
                    print(f"Patch with --reject apply had errors")
                    print(f"Continuing with testing...")

    # print current status
    print(repo.git_repo.git.status())

    run_tests(task.repo_name, task.task_num, str(out_file.resolve()))

    print(f"Finished testing task {task.task_num} for repository {task.repo_name}")


def test_tasks(tasks: list[TaskInstance], tool_name: str | None = None, skip_gpu_tests: bool = False, retest: bool = False, task_type: TaskType | Literal['all'] = 'all', test_all_patches: bool = False):
    """Test tasks for a repository."""
    # Sort tasks by number
    tasks.sort(key=lambda t: t.task_num)
    
    print(f"Testing {len(tasks)} tasks for repository {tasks[0].repo_name}")

    task_types_to_test = [task_type] if task_type != 'all' else list(TaskType)
    print(f"Testing task types: {task_types_to_test}")
    
    # Process each task
    for task in tasks:
        # Skip GPU tests if requested
        test_patch = task.test_patch
        if "@gpu_test" in test_patch and skip_gpu_tests:
            print(f"Skipping GPU task {task.task_num} for repository {task.repo_name}")
            continue
        for t_type in task_types_to_test:
            
            # Get all patches to test - if tool_name is None, this gets patches from all tools
            patches = find_task_patches(task=task, tool_name=tool_name, include_all=test_all_patches, 
                                    only_unevaluated=(not retest), task_type=t_type)
            
            if not patches:
                print(f"No patches to test for task {task.task_num} ({t_type})" +
                    (f" with tool {tool_name}" if tool_name else ""))
                continue
            
            print(f"Testing {len(patches)} patches for task {task.task_num} ({t_type})")
            
            for patch in patches:
                # Extract tool name from patch filename
                patch_path = Path(patch)
                current_tool = patch_path.parent.name
                
                try:
                    print(f"Testing task {task.task_num} ({t_type}) with tool: {current_tool} and patch: {patch_path.name}")
                    prepare_task(task=task, install=True, task_type=t_type, test=True)
                    patch_and_test(task=task, tool_name=current_tool, patch_file=patch, task_type=t_type)
                except Exception as e:
                    print(f"Task {task.task_num} ({t_type}) with tool {current_tool} test failed: {e}")
                    traceback.print_exc()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LiveSWEBench Task Testing")
    parser.add_argument('--repo_name', type=str, nargs="*", help='Name of the repository or repositories (if none provided, all repos will be processed)')
    parser.add_argument('--task_num', default=None, type=int, help='Task number')
    parser.add_argument('--tool_name', default=None, type=str, help='Tool name')
    parser.add_argument('--skip-gpu-tests', action='store_true')
    parser.add_argument('--retest', action='store_true')
    parser.add_argument('--task_type', default=TaskType.AGENT, choices=list(TaskType), type=TaskType)
    parser.add_argument('--test-all-patches', action='store_true', help='Test all patches for each tool, not just the latest')
    parser.add_argument('--task_source', type=str, choices=['local', 'huggingface'], default='huggingface',
                      help='Source of tasks, either local or from huggingface')

    args = parser.parse_args()

    all_tasks = load_tasks(task_source=args.task_source)


    tasks_to_test: list[TaskInstance] = []

    if args.task_num is not None:
        assert args.repo_name is not None and len(args.repo_name) == 1
        tasks_to_test.append(all_tasks[args.repo_name[0]][args.task_num])
    else:
        # Extract repo_names and other args for test_tasks
        repo_names = args.__dict__.pop('repo_name')
        
        # If no repos specified, use all dirs in BASE_TASK_PATH
        if not repo_names:
            print("No repositories specified. Processing all repositories.")
            repo_names = list(all_tasks.keys())
            if not repo_names:
                print("No repositories found.")
                exit(1)
            print(f"Found repositories: {repo_names}")

        for repo_name in repo_names:
            tasks_to_test.extend(all_tasks[repo_name].values())
        
    test_tasks(tasks=tasks_to_test, tool_name=args.tool_name, skip_gpu_tests=args.skip_gpu_tests, retest=args.retest, task_type=args.task_type, test_all_patches=args.test_all_patches)