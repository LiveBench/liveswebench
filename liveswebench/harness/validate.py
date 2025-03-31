import os
from liveswebench.harness.prepare import prepare_task
from liveswebench.util.tasks import TaskInstance, TaskType, get_log_path, load_tasks
from liveswebench.harness.test import patch_and_test
import traceback
import argparse

def validate_tasks(
    tasks: list[TaskInstance], baseline: bool = True, baseline_with_test_patch: bool = True, gold: bool = True, revalidate: bool = False, skip_gpu_tests: bool = False
):

    for task in tasks:
        print(f"Validating task {task.task_num} for repository {task.repo_name}")

        # Create task data directory if it doesn't exist
        task_data_path = task.task_data_path / "baseline"
        task_data_path.mkdir(parents=True, exist_ok=True)

        test_patch = task.test_patch
        if "@gpu_test" in test_patch and skip_gpu_tests:
            print(f"Skipping GPU task {task.task_num} for repository {task.repo_name}")
            continue

        baseline_log = task_data_path / get_log_path(task, TaskType.AGENT, "baseline", omit_timestamp=True)
        baseline_test_log = task_data_path / get_log_path(task, TaskType.AGENT, "baseline_with_test_patch", omit_timestamp=True)
        gold_log = task_data_path / get_log_path(task, TaskType.AGENT, "gold", omit_timestamp=True)

        has_baseline = os.path.exists(baseline_log)
        has_baseline_with_test_patch = os.path.exists(baseline_test_log)
        has_gold = os.path.exists(gold_log)

        if baseline and (not has_baseline or revalidate):
            if has_baseline:
                os.remove(baseline_log)

            print("Baseline run")
            prepare_task(task=task, install=True, test=True)
            
            # Run tests without patches
            try:
                patch_and_test(
                    task=task,
                    tool_name="baseline",
                    skip_test_patch=True, # skip applying test patch
                    out_file_name=baseline_log
                )
            except Exception as e:
                print(f"Error running baseline tests for task {task.task_num} for repository {task.repo_name}: {e}")
                traceback.print_exc()

        if baseline_with_test_patch and (not has_baseline_with_test_patch or revalidate):
            if has_baseline_with_test_patch:
                os.remove(baseline_test_log)

            print("Baseline run with test patch")
            prepare_task(task=task, install=True, test=True)

            # Run tests with only test patch
            try:
                patch_and_test(
                    task=task,
                    tool_name="baseline",
                    out_file_name=baseline_test_log
                )
            except Exception as e:
                print(f"Error running baseline_with_test_patch tests for task {task.task_num} for repository {task.repo_name}: {e}")
                traceback.print_exc()

        if gold and (not has_gold or revalidate):
            if has_gold:
                os.remove(gold_log)

            print("Gold test run")
            prepare_task(task=task, install=True, test=True)

            # Run tests with test patch and gold patch
            try:
                patch_and_test(
                    task=task,
                    tool_name="baseline",
                    patch_file=task.gold_patch,
                    out_file_name=gold_log
                )
            except Exception as e:
                print(f"Error running gold tests for task {task.task_num} for repository {task.repo_name}: {e}")
                traceback.print_exc()

        print(f"Finished validating task {task.task_num} for repository {task.repo_name}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LiveSWEBench Task Validation")
    parser.add_argument('--repo_name', nargs='+', type=str, default=None, help='Name of repository (if not specified, will validate all repositories)')
    parser.add_argument('--task_type', type=TaskType, choices=list(TaskType), default=TaskType.AGENT)
    parser.add_argument('--task_source', type=str, choices=['local', 'huggingface'], default='huggingface',
                      help='Source of tasks, either local or from huggingface')
    parser.add_argument('--skip-gpu-tests', action='store_true')
    parser.add_argument('--revalidate', action='store_true')
    parser.add_argument('--skip-baseline', action='store_true', help='Skip running baseline validation')
    parser.add_argument('--skip-baseline-with-test-patch', action='store_true', help='Skip running baseline with test patch validation')
    parser.add_argument('--skip-gold', action='store_true', help='Skip running gold validation')

    args = parser.parse_args()

    all_tasks = load_tasks(task_source=args.task_source)

    tasks_to_validate: list[TaskInstance] = []

    if args.task_num is not None:
        assert args.repo_name is not None and len(args.repo_name) == 1
        tasks_to_validate.append(all_tasks[args.repo_name[0]][args.task_num])
    else:
        # Extract repo_names and other args for test_tasks
        repo_names = args.__dict__.pop('repo_name')
        
        if not repo_names:
            print("No repositories specified. Processing all repositories.")
            repo_names = list(all_tasks.keys())
        
        for repo_name in repo_names:
            tasks_to_validate.extend(all_tasks[repo_name].values())
        
    validate_tasks(tasks=tasks_to_validate, baseline=not args.skip_baseline, baseline_with_test_patch=not args.skip_baseline_with_test_patch, gold=not args.skip_gold, revalidate=args.revalidate, skip_gpu_tests=args.skip_gpu_tests)