import glob
import os
import datetime
from pathlib import Path
from typing_extensions import override
from liveswebench.util.repo import get_repo
from dataclasses import dataclass, field
from enum import Enum
from datasets import load_dataset

from liveswebench.util.util import BASE_TASK_PATH

class TaskType(Enum):
    AGENT = 'agent'
    EDIT = 'edit'
    AUTOCOMPLETE = 'autocomplete'

    @override
    def __str__(self) -> str:
        return self.value

@dataclass
class TaskInstance:
    repo_name: str
    task_num: int
    base_commit: str
    gold_patch: str
    test_patch: str
    agent_prompt: str
    edit_patch: str | None
    edit_prompt: str | None
    autocomplete_patch: str | None
    autocomplete_prompts: str | None
    task_data_path: Path = field(init=False)

    def __post_init__(self):
        self.task_data_path = get_repo(self.repo_name).task_data_path / str(self.task_num)

    def get_ground_truth_patch(self, task_type: TaskType) -> str:
        if task_type == TaskType.AGENT:
            res = self.gold_patch
        elif task_type == TaskType.EDIT:
            res = self.edit_patch
        elif task_type == TaskType.AUTOCOMPLETE:
            res = self.autocomplete_patch
        if res is None:
            raise ValueError(f"Ground truth patch for task {self.task_num} ({task_type}) in repository {self.repo_name} could not be constructed")
        return res

    def get_prompt(self, task_type: TaskType) -> str:
        if task_type == TaskType.AGENT:
            res = self.agent_prompt
        elif task_type == TaskType.EDIT:
            res = self.edit_prompt
        elif task_type == TaskType.AUTOCOMPLETE:
            res = self.autocomplete_prompts
        if res is None:
            raise ValueError(f"Prompt for task {self.task_num} ({task_type}) in repository {self.repo_name} could not be constructed")
        return res
    
def get_patch_path(task: TaskInstance, task_type: TaskType, tool_name: str, omit_timestamp: bool = False) -> Path:
    """
    Get the patch file path for the task instance with the given tool name.
    If omit_timestamp is True, the timestamp is omitted.
    """
    timestamp = "" if omit_timestamp else f"_{datetime.datetime.now().strftime('%Y%m%d')}"
    return task.task_data_path / tool_name / f"{tool_name}_{task_type}_patch{timestamp}.patch"

def get_log_path(task: TaskInstance, task_type: TaskType, tool_name: str, timestamp: str | None = None, omit_timestamp: bool = False) -> Path:
    """
    Get the log file path for the task instance with the given tool name.
    If timestamp is provided, it is used instead of generating a new one.
    If omit_timestamp is True, the timestamp is omitted.
    """
    if timestamp is None:
        ts = "" if omit_timestamp else f"_{datetime.datetime.now().strftime('%Y%m%d')}"
    else:
        ts = f"_{timestamp}"
    return task.task_data_path / tool_name / f"{tool_name}_{task_type}{ts}.log"

def get_partial_gold_patch(task: TaskInstance, task_type: TaskType) -> str:
    if task_type == TaskType.AGENT:
        return task.gold_patch
    
    exclude_patch = task.get_ground_truth_patch(task_type)
    
    from liveswebench.harness.util import construct_partial_patch
    partial_gold_patch = construct_partial_patch(
        task.gold_patch,
        exclude_patch
    )
    if partial_gold_patch is None:
        raise ValueError(f"Partial gold patch for task {task.task_num} in repository {task.repo_name} could not be constructed")
    return partial_gold_patch

def get_removal_patch_for_task(task: TaskInstance, task_type: TaskType) -> str | None:
    if task_type == TaskType.AGENT:
        return None
    
    ground_truth_patch = task.get_ground_truth_patch(task_type)
    from liveswebench.harness.util import filter_patch_by_operation
    removal_patch = filter_patch_by_operation(
        ground_truth_patch,
        operation='-'
    )

    return removal_patch
    
def get_relevant_files_for_task(task: TaskInstance, task_type: TaskType) -> list[str] | None:
    ground_truth_patch = task.get_ground_truth_patch(task_type)
    from liveswebench.harness.util import extract_hunks_from_patch
    hunks = extract_hunks_from_patch(ground_truth_patch)
    if not hunks:
        return None
    return [hunk['file_path'] for hunk in hunks.values() if 'file_path' in hunk]

def find_task_patches(task: TaskInstance, task_type: TaskType, tool_name: str | None = None, include_all: bool = False, only_unevaluated: bool = False) -> list[str]:
    """
    Find all patches for the given task instance and task type.
    If tool_name is provided, only patches for the given tool are returned.
    If include_all is False, only the latest patch is returned.
    If only_unevaluated is True, only unevaluated patches are returned.
    """
    if tool_name is None:
        data_paths = [task.task_data_path / path for path in os.listdir(task.task_data_path)]
    else:
        data_paths = [task.task_data_path / tool_name]

    patch_files = []
    for data_path in data_paths:
        patch_files.extend(glob.glob(os.path.join(data_path, f"*_{task_type}_patch_*.patch")))
    patch_files.sort(key=lambda x: os.path.basename(x).split('_')[-1].replace('.patch', '')) # sort by timestamp

    tool_to_patch_files = {}
    for patch_file in patch_files:
        tool_name = os.path.basename(patch_file).split('_')[0]
        if tool_name not in tool_to_patch_files:
            tool_to_patch_files[tool_name] = []
        tool_to_patch_files[tool_name].append(patch_file)

    patch_files = []
    for tool_name in tool_to_patch_files:
        tool_to_patch_files[tool_name].sort(key=lambda x: os.path.basename(x).split('_')[-1].replace('.patch', '')) # sort by timestamp
        if not include_all:
            tool_to_patch_files[tool_name] = [tool_to_patch_files[tool_name][-1]]
        patch_files.extend(tool_to_patch_files[tool_name])

    if len(patch_files) == 0:
        return []

    patches = []
    if only_unevaluated:
        for patch_file in patch_files:
            tool_name = os.path.basename(patch_file).split('_')[0]
            timestamp = os.path.basename(patch_file).split('_')[-1].replace('.patch', '')
            logs = find_task_logs(task, task_type, tool_name=tool_name, timestamp=timestamp)
            if len(logs) == 0:
                patches.append(patch_file)
    else:
        patches = patch_files
    return patches

def find_task_logs(task: TaskInstance, task_type: TaskType, tool_name: str | None = None, timestamp: str | None = None, include_all: bool = False) -> list[str]:
    """
    Find all logs for the given task instance and task type.
    If tool_name is provided, only logs for the given tool are returned.
    If include_all is False, only the latest log is returned.
    If timestamp is provided, only logs with the given timestamp are returned.
    """
    data_path = task.task_data_path if tool_name is None else task.task_data_path / tool_name
    if timestamp is None:
        log_glob = f"*_{task_type}_*.log"
    else:
        log_glob = f"*_{task_type}_{timestamp}.log"
    log_files = glob.glob(os.path.join(data_path, log_glob))
    log_files.sort(key=lambda x: os.path.basename(x).split('_')[-1].replace('.log', '')) # sort by timestamp
    if len(log_files) == 0:
        return []
    if include_all:
        return log_files
    return [log_files[-1]]

def load_tasks(task_source: str = 'huggingface'):
    """
    Load all tasks from the task data path.
    
    Args:
        task_source: The source of the tasks. Either 'local' or 'huggingface'.
    """
    tasks: dict[str, dict[int, TaskInstance]] = {}
    if task_source == 'local':
        for repo_dir in BASE_TASK_PATH.iterdir():
            if not repo_dir.is_dir():
                continue
            for task_dir in repo_dir.iterdir():
                if not task_dir.is_dir():
                    continue
                repo_name = repo_dir.name
                task_num = int(task_dir.name)
                base_commit = (task_dir / 'commit.txt').read_text().strip()
                gold_patch = (task_dir / 'gold_patch.patch').read_text()
                test_patch = (task_dir / 'test_patch.patch').read_text()
                agent_prompt = (task_dir / 'prompt.md').read_text()
                edit_patch = (task_dir / 'edit_patch.patch').read_text() if (task_dir / 'edit_patch.patch').exists() else None
                edit_prompt = (task_dir / 'edit_prompt.txt').read_text() if (task_dir / 'edit_prompt.txt').exists() else None
                autocomplete_patch = (task_dir / 'autocomplete_patch.patch').read_text() if (task_dir / 'autocomplete_patch.patch').exists() else None
                autocomplete_prompts = (task_dir / 'autocomplete_prompts.txt').read_text() if (task_dir / 'autocomplete_prompts.txt').exists() else None

                if repo_name not in tasks:
                    tasks[repo_name] = {}

                tasks[repo_name][task_num] = TaskInstance(repo_name=repo_name, task_num=task_num, base_commit=base_commit, gold_patch=gold_patch, test_patch=test_patch, agent_prompt=agent_prompt, edit_patch=edit_patch, edit_prompt=edit_prompt, autocomplete_patch=autocomplete_patch, autocomplete_prompts=autocomplete_prompts)
    elif task_source == 'huggingface':
        dataset = load_dataset('livebench/liveswebench', split='test')
        for task in dataset:
            if task['repo_name'] not in tasks:
                tasks[task['repo_name']] = {}
            tasks[task['repo_name']][task['task_num']] = TaskInstance(repo_name=task['repo_name'], task_num=task['task_num'], base_commit=task['commit'], gold_patch=task['gold_patch'], test_patch=task['test_patch'], agent_prompt=task['prompt'], edit_patch=task['edit_patch'], edit_prompt=task['edit_prompt'], autocomplete_patch=task['autocomplete_patch'], autocomplete_prompts=task['autocomplete_prompts'])
    else:
        raise ValueError(f"Unknown task source: {task_source}. Must be 'local' or 'huggingface'.")
               
    return tasks
