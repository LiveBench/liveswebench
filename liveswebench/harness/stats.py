import argparse
import re
import os
from pathlib import Path
import sys
import statistics
from collections import defaultdict
from typing import Literal

from liveswebench.util.repo import get_all_repos
from liveswebench.harness.util import extract_hunks_from_patch
from liveswebench.util.tasks import TaskInstance, TaskType, load_tasks

def count_lines_changed(patch_content: str) -> int:
    """Count the number of lines added and removed in a patch.
    
    This function simply counts the total number of lines that start with '+' or '-'
    in the patch content, excluding the lines that are part of the diff header (--- and +++ lines).
    """
    total_changes = 0
    
    for line in patch_content.splitlines():
        # Skip diff header lines
        if line.startswith('---') or line.startswith('+++') or line.startswith('@@'):
            continue
        
        # Count lines that start with + or -
        if line.startswith('+') or line.startswith('-'):
            total_changes += 1
    
    return total_changes

def count_hunks_in_patch(patch_content: str) -> tuple[int, float, list[int]]:
    """Count the number of hunks in a patch and calculate the average line additions per hunk."""
    hunks = re.findall(r'@@\s+-\d+,\d+\s+\+\d+,\d+\s+@@', patch_content)
    if not hunks:
        return 0, 0, []

    file_hunks = extract_hunks_from_patch(patch_content)
    total_additions = 0
    additions_per_hunk: list[int] = []
    
    for file_info in file_hunks.values():
        for hunk in file_info["hunks"]:
            lines = hunk.splitlines()
            # Count only meaningful addition lines (starting with +)
            # Skip lines that are part of the diff header (+++), blank lines, or comment lines
            additions = []
            for line in lines:
                if line.startswith('+') and not line.startswith('+++'):
                    # Extract the content of the line (without the '+')
                    content = line[1:].strip()
                    # Skip if the line is blank or is a comment
                    if content and not content.startswith('//') and not content.startswith('#'):
                        additions.append(line)
            
            total_additions += len(additions)
            additions_per_hunk.append(len(additions))
    
    num_hunks = len(hunks)
    avg_additions_per_hunk = total_additions / num_hunks if num_hunks > 0 else 0
    
    return num_hunks, avg_additions_per_hunk, additions_per_hunk

def get_autocomplete_prompts_avg_length(prompt_content: str) -> float:
    """Calculate the average length of prompts in the autocomplete_prompts.txt file."""

    # Split by --- separator typically used between prompts
    prompts = re.split(r'-{3,}', prompt_content)
    prompts = [p.strip() for p in prompts if p.strip()]
    
    if not prompts:
        return 0
        
    total_length = sum(len(prompt) for prompt in prompts)
    return total_length / len(prompts)

def analyze_patch(patch_content: str) -> dict[str, int | str | list[str] | float | list[int] | None] | None:
    """Analyze a patch file and return statistics."""
    
    lines_changed = count_lines_changed(patch_content)
    
    # Use extract_hunks_from_patch utility to get file information and hunks
    file_hunks = extract_hunks_from_patch(patch_content)
    
    # Extract modified files from the file_hunks data
    modified_files = []
    for file_info in file_hunks.values():
        if file_info["file_path"] and file_info["file_path"] not in modified_files:
            modified_files.append(file_info["file_path"])
    
    # Extract edit locations from the hunk headers
    edit_locations: set[str] = set()
    for file_info in file_hunks.values():
        filename = Path(file_info["file_path"]).stem if file_info["file_path"] else ""
        
        for hunk in file_info["hunks"]:
            # Extract the context from the hunk header
            hunk_header_match = re.search(r'^@@ -\d+,\d+ \+\d+,\d+ @@(.*)$', hunk, re.MULTILINE)
            if hunk_header_match:
                context = hunk_header_match.group(1).strip()
                if context:
                    edit_locations.add(context)
                elif filename:
                    edit_locations.add(filename)
                    
        # If no edit locations were found for this file, use the filename
        if not any(loc in edit_locations for loc in [filename]) and filename:
            edit_locations.add(filename)
    
    # Count hunks and average line additions per hunk
    num_hunks, avg_additions_per_hunk, additions_per_hunk = count_hunks_in_patch(patch_content)
    
    return {
        'lines_changed': lines_changed,
        'files_modified': len(modified_files),
        'modified_files': modified_files,
        'edit_locations': len(edit_locations),
        'locations': sorted(list(edit_locations)),
        'num_hunks': num_hunks,
        'avg_additions_per_hunk': avg_additions_per_hunk,
        'additions_per_hunk': additions_per_hunk
    }

def analyze_single_task(task: TaskInstance, task_type: TaskType = TaskType.AGENT, print_stats: bool = True) -> dict[str, int | list[str] | float | list[int] | None] | None:
    """Analyze a single task and print its statistics.
    
    Args:
        repo_name (str): Name of the repository
        task_num (int): Task number
        task_type (TaskType): Type of task to analyze
        print_stats (bool): Whether to print individual task statistics
        
    Returns:
        dict: The statistics for the task, or None if analysis failed
    """
    try:
        
        if task_type == TaskType.AGENT:
            patch_path = task.gold_patch
            # Print removed: no longer print each patch file path
            stats = analyze_patch(patch_path)
            
            if stats:
                # Add issue text length
                issue_text_length = len(task.agent_prompt)
                stats['issue_text_length'] = issue_text_length
                stats['task_type'] = 'agent'
                stats['task_id'] = f"{task.repo_name}/{task.task_num}"
                
                if print_stats:
                    print(f"\nAgent Task Statistics for {task.repo_name}/{task.task_num}:")
                    print(f"Lines of code changed: {stats['lines_changed']}")
                    print(f"Files modified: {stats['files_modified']}")
                    print(f"Locations edited: {stats['edit_locations']}")
                    print(f"Issue Text: {stats['issue_text_length']} characters")
                    
                    print("\nModified files:")
                    for file in stats['modified_files']:
                        print(f"  - {file}")
                    
                    if stats['locations']:
                        print("\nEdited locations:")
                        for loc in stats['locations']:
                            print(f"  - {loc}")
            
        elif task_type == TaskType.EDIT:
            if task.edit_patch is None or task.edit_prompt is None:
                print(f"Edit patch or edit prompt not found for task {task.repo_name}/{task.task_num}")
                return None
                
            # Print removed: no longer print each patch file path
            edit_stats = analyze_patch(task.edit_patch)
            
            # Also analyze gold patch to determine additional files modified
            gold_stats = analyze_patch(task.gold_patch)
            
            if edit_stats and gold_stats:
                # Calculate additional files modified in gold patch
                edit_files = set(edit_stats['modified_files'])
                gold_files = set(gold_stats['modified_files'])
                additional_files = gold_files - edit_files
                
                # Add to edit stats
                edit_stats['issue_text_length'] = len(task.edit_prompt)
                edit_stats['additional_files_in_gold'] = len(additional_files)
                edit_stats['additional_files_list'] = sorted(list(additional_files))
                edit_stats['task_type'] = 'edit'
                edit_stats['task_id'] = f"{task.repo_name}/{task.task_num}"
                
                if print_stats:
                    print(f"\nEdit Task Statistics for {task.repo_name}/{task.task_num}:")
                    print(f"Lines of code changed: {edit_stats['lines_changed']}")
                    print(f"Locations edited: {edit_stats['edit_locations']}")
                    print(f"Additional files in gold patch: {edit_stats['additional_files_in_gold']}")
                    print(f"Edit Prompt: {edit_stats['issue_text_length']} characters")
                    
                    if edit_stats['additional_files_list']:
                        print("\nAdditional files in gold patch:")
                        for file in edit_stats['additional_files_list']:
                            print(f"  - {file}")
                    
                    if edit_stats['locations']:
                        print("\nEdited locations:")
                        for loc in edit_stats['locations']:
                            print(f"  - {loc}")
                
                stats = edit_stats
            else:
                stats = None
                
        elif task_type == TaskType.AUTOCOMPLETE:
            if task.autocomplete_patch is None or task.autocomplete_prompts is None:
                print(f"Autocomplete patch or autocomplete prompts not found for task {task.repo_name}/{task.task_num}")
                return None
            
            stats = analyze_patch(task.autocomplete_patch)
            
            # Add avg prompt length from autocomplete_prompts.txt
            avg_prompt_length = get_autocomplete_prompts_avg_length(task.autocomplete_prompts)
            
            stats['task_type'] = 'autocomplete'
            stats['avg_prompt_length'] = avg_prompt_length
            
            print(f"\nAutocomplete Task Statistics for {task.repo_name}/{task.task_num}:")
            print(f"Lines of code changed: {stats['lines_changed']}")
            print(f"Files modified: {stats['files_modified']}")
            print(f"Number of hunks: {stats['num_hunks']}")
            print(f"Avg line additions per hunk: {stats['avg_additions_per_hunk']:.2f}")
            print(f"Avg Prompt Length: {stats['avg_prompt_length']:.2f} characters")
            
            print("\nModified files:")
            for file in stats['modified_files']:
                print(f"  - {file}")
            
            if stats['locations']:
                print("\nEdited locations:")
                for loc in stats['locations']:
                    print(f"  - {loc}")
                    
            return stats
        
        if not stats:
            print("Failed to analyze the task.")
        
        return stats
    except Exception as e:
        print(f"Error analyzing task {task.repo_name}/{task.task_num}: {e}")
        return None

def analyze_repo_tasks(tasks: dict[int, TaskInstance], repo_name: str, task_type: TaskType | Literal['all'] = TaskType.AGENT, print_repo_stats: bool = True) -> dict[str, list[dict[str, int | list[str] | float | list[int] | None]] | set[str] | dict[str, dict[str, int | list[str] | float | list[int] | None]] | list[tuple[int, str]] | list[tuple[int, str]] | list[str] | None] | None:
    """Analyze all tasks in a repository and generate aggregate statistics.
    
    Args:
        repo_name (str): Name of the repository
        task_type (str): Type of task to analyze ('agent', 'edit', 'prompted_autocomplete', 'autocomplete', or 'all')
        print_repo_stats (bool): Whether to print repository-level statistics
        
    Returns:
        dict: A dictionary mapping task types to their statistics information
    """
    tasks_to_analyze = [t for t in tasks.values() if t.repo_name == repo_name]

    if task_type == 'all':
        task_types = list(TaskType)
        result = {}
        
        for t_type in task_types:
            stats_result = analyze_repo_tasks(tasks, repo_name, t_type, print_repo_stats)
            if stats_result:
                result[t_type] = stats_result
        
        return result
    
    # Collect statistics for all tasks
    all_stats = []
    all_files_modified = set()
    all_locations = set()
    
    # Track individual metrics for aggregate statistics
    metrics = defaultdict(list)
    
    # For autocomplete tasks, collect all additions per hunk with task info
    all_additions_per_hunk = []
    
    # Collect issue text lengths with task info
    all_issue_text_lengths = []
    
    # Track which task has min/max values for each metric
    task_stats = {}
    
    # Track successful and failed task numbers
    successful_tasks = []
    failed_tasks = []
    
    print(f"Analyzing {task_type} tasks in repository: {repo_name}")
    
    if not tasks_to_analyze:
        print(f"No {task_type} tasks found in repository {repo_name}.")
        return None
    
    # Loop through all tasks
    for task in sorted(tasks_to_analyze, key=lambda t: t.task_num):

        if task_type != TaskType.AUTOCOMPLETE and task_type != TaskType.PROMPTED_AUTOCOMPLETE and task.task_num == 55444:
            continue
        
        stats = analyze_single_task(task, task_type, print_stats=False)
        
        if stats:
            all_stats.append(stats)
            task_id = f"{repo_name}/{task.task_num}"
            successful_tasks.append(str(task.task_num))
            
            if 'modified_files' in stats:
                all_files_modified.update(stats['modified_files'])
            
            if 'locations' in stats:
                all_locations.update(stats['locations'])
            
            # Store stats with task number for tracking min/max
            task_stats[str(task.task_num)] = stats
            
            # Collect metrics for aggregate statistics based on task type
            metrics['lines_changed'].append(stats['lines_changed'])
            metrics['edit_locations'].append(stats['edit_locations'])
            
            # Collect issue text length with task info
            if task_type != TaskType.AUTOCOMPLETE and task_type != TaskType.PROMPTED_AUTOCOMPLETE and 'issue_text_length' in stats:
                all_issue_text_lengths.append((stats['issue_text_length'], task_id))
            
            if task_type == TaskType.AGENT:
                metrics['files_modified'].append(stats['files_modified'])
            elif task_type == TaskType.EDIT:
                metrics['additional_files_in_gold'].append(stats['additional_files_in_gold'])
            elif task_type == TaskType.AUTOCOMPLETE:
                metrics['files_modified'].append(stats['files_modified'])
                metrics['num_hunks'].append(stats['num_hunks'])
                # Collect all additions per hunk with task info
                if 'additions_per_hunk' in stats:
                    for addition in stats['additions_per_hunk']:
                        all_additions_per_hunk.append((addition, task_id))
        else:
            failed_tasks.append(str(task.task_num))
    
    if not all_stats:
        print(f"No {task_type} tasks were successfully analyzed in repository {repo_name}.")
        return None
    
    # Print the task numbers that were analyzed
    if successful_tasks:
        print(f"Successfully analyzed {len(successful_tasks)} {task_type} tasks: {', '.join(successful_tasks)}")
    if failed_tasks:
        print(f"Failed to analyze {len(failed_tasks)} {task_type} tasks: {', '.join(failed_tasks)}")
    
    if print_repo_stats:
        # For autocomplete tasks, add additions_per_hunk to metrics for direct statistics
        if task_type == TaskType.AUTOCOMPLETE and all_additions_per_hunk:
            metrics['additions_per_hunk_with_task'] = all_additions_per_hunk
            metrics['additions_per_hunk'] = [a[0] for a in all_additions_per_hunk]
            
        # Add raw issue text lengths for direct statistics
        metrics['issue_text_length_with_task'] = all_issue_text_lengths
        metrics['issue_text_length'] = [t[0] for t in all_issue_text_lengths]
            
        print_aggregate_statistics(repo_name, task_type, all_stats, all_files_modified, all_locations, task_stats, metrics)
    
    return {
        'all_stats': all_stats,
        'all_files_modified': all_files_modified,
        'all_locations': all_locations,
        'task_stats': task_stats,
        'metrics': metrics,
        'all_additions_per_hunk': all_additions_per_hunk if task_type in [TaskType.AUTOCOMPLETE] else [],
        'all_issue_text_lengths': all_issue_text_lengths,
        'successful_tasks': successful_tasks,
        'failed_tasks': failed_tasks
    }

def analyze_tasks(tasks: dict[str, dict[int, TaskInstance]], task_type=TaskType.AGENT):
    """Analyze all tasks in the given tasks dictionary.
    
    Args:
        tasks (dict[str, dict[int, TaskInstance]]): A dictionary of repository names to task numbers to TaskInstances
        task_type (TaskType): Type of task to analyze, or 'all' to analyze all tasks
        
    Returns:
        dict: A dictionary mapping task types to their combined statistics
    """
    if task_type == 'all':
        task_types = list(TaskType)
        result = {}
        
        for t_type in task_types:
            stats_result = analyze_tasks(tasks, t_type)
            if stats_result:
                result[t_type] = stats_result
        
        return result
    
    print(f"Analyzing all tasks for {task_type} tasks")
    
    # Get all repository directories
    repo_names = list(tasks.keys())
    
    if not repo_names:
        print("No repositories found in tasks dictionary.")
        return None
    
    print(f"Found {len(repo_names)} repositories: {', '.join(repo_names)}")
    
    # Combined statistics across all repositories
    combined_stats = []
    combined_files = set()
    combined_locations = set()
    combined_task_stats = {}
    combined_metrics = defaultdict(list)
    
    # For autocomplete tasks, collect all additions per hunk across all repos and tasks
    all_additions_per_hunk = []
    
    # Collect all issue text lengths across all repositories
    all_issue_text_lengths = []
    
    # Track tasks by repository
    repo_tasks = {}
    
    # Process each repository
    for repo_name in sorted(repo_names):
        repo_stats = analyze_repo_tasks(tasks[repo_name], repo_name, task_type, print_repo_stats=False)
        
        if repo_stats:
            all_stats = repo_stats['all_stats']
            all_files = repo_stats['all_files_modified']
            all_locations = repo_stats['all_locations']
            task_stats = repo_stats['task_stats']
            metrics = repo_stats['metrics']
            successful_tasks = repo_stats['successful_tasks']
            
            # Store successful tasks by repository
            repo_tasks[repo_name] = successful_tasks
            
            combined_stats.extend(all_stats)
            combined_files.update(all_files)
            combined_locations.update(all_locations)
            
            # Add repository prefix to task IDs to make them unique
            for task_num, stats in task_stats.items():
                combined_task_stats[f"{repo_name}/{task_num}"] = stats
            
            # Combine metrics
            for metric_name, values in metrics.items():
                if metric_name not in ['additions_per_hunk_with_task', 'issue_text_length_with_task']:
                    combined_metrics[metric_name].extend(values)
            
            # For autocomplete tasks, collect all additions per hunk
            if task_type == TaskType.AUTOCOMPLETE and 'all_additions_per_hunk' in repo_stats:
                all_additions_per_hunk.extend(repo_stats['all_additions_per_hunk'])
                
            # Collect all issue text lengths
            if 'all_issue_text_lengths' in repo_stats:
                all_issue_text_lengths.extend(repo_stats['all_issue_text_lengths'])
    
    if not combined_stats:
        print(f"No {task_type} tasks were successfully analyzed across all repositories.")
        return None
    
    # Print summary of tasks analyzed by repository
    print("\nTasks analyzed by repository:")
    total_tasks = 0
    for repo_name, tasks in repo_tasks.items():
        print(f"  {repo_name}: {len(tasks)} tasks ({', '.join(tasks)})")
        total_tasks += len(tasks)
    print(f"\nTotal {task_type} tasks analyzed: {total_tasks}")
    
    # For autocomplete tasks, add the collected additions_per_hunk to metrics
    if task_type in [TaskType.AUTOCOMPLETE] and all_additions_per_hunk:
        combined_metrics['additions_per_hunk_with_task'] = all_additions_per_hunk
        combined_metrics['additions_per_hunk'] = [a[0] for a in all_additions_per_hunk]
        
    # Add collected issue text lengths to metrics
    if all_issue_text_lengths:
        combined_metrics['issue_text_length_with_task'] = all_issue_text_lengths
        combined_metrics['issue_text_length'] = [t[0] for t in all_issue_text_lengths]
    
    # Print combined statistics
    print_aggregate_statistics(None, task_type, combined_stats, combined_files, combined_locations, combined_task_stats, combined_metrics)
    
    return {
        'all_stats': combined_stats,
        'all_files_modified': combined_files,
        'all_locations': combined_locations,
        'task_stats': combined_task_stats,
        'metrics': combined_metrics,
        'all_additions_per_hunk': all_additions_per_hunk if task_type in [TaskType.AUTOCOMPLETE] else [],
        'all_issue_text_lengths': all_issue_text_lengths,
        'repo_tasks': repo_tasks
    }

def print_aggregate_statistics(repo_name, task_type, all_stats, all_files_modified, all_locations, task_stats, metrics):
    """Print aggregate statistics for a repository or the entire task set.
    
    Args:
        repo_name (str or None): Name of the repository, or None for all repositories
        task_type (str): Type of task being analyzed
        all_stats (list): List of statistics for all tasks
        all_files_modified (set): Set of all modified files
        all_locations (set): Set of all edit locations
        task_stats (dict): Dictionary mapping task numbers to their statistics
        metrics (dict): Dictionary of metric names to lists of values
    """
    # Find tasks with min/max values for each metric
    min_tasks = {}
    max_tasks = {}
    
    # Define cross-task metrics that are tracked separately with task info
    cross_task_metrics = ['additions_per_hunk', 'issue_text_length']
    cross_task_metrics_with_info = {
        'additions_per_hunk': 'additions_per_hunk_with_task',
        'issue_text_length': 'issue_text_length_with_task'
    }
    
    for metric_name in metrics:
        if not metrics[metric_name] or metric_name in cross_task_metrics_with_info.values():
            continue
            
        min_val = min(metrics[metric_name])
        max_val = max(metrics[metric_name])
        
        min_task = None
        max_task = None
        
        # For cross-task metrics, look up the task ID from the _with_task version
        if metric_name in cross_task_metrics:
            info_metric = cross_task_metrics_with_info[metric_name]
            if info_metric in metrics:
                # Find the task with the min/max value
                for val, task_id in metrics[info_metric]:
                    if val == min_val:
                        min_task = task_id
                    if val == max_val:
                        max_task = task_id
        else:
            # For regular metrics, look up from task_stats
            for task_id, stats in task_stats.items():
                if metric_name in stats and stats[metric_name] == min_val:
                    min_task = task_id
                if metric_name in stats and stats[metric_name] == max_val:
                    max_task = task_id
        
        min_tasks[metric_name] = min_task
        max_tasks[metric_name] = max_task
    
    # Print header for aggregate statistics
    if repo_name:
        print(f"\nAggregate Statistics for {repo_name} {task_type.title()} Tasks ({len(all_stats)} tasks):")
    else:
        print(f"\nAggregate Statistics for All Repositories {task_type.title()} Tasks ({len(all_stats)} tasks):")
    
    # Print statistics for each metric
    for metric_name, values in metrics.items():
        # Skip the _with_task metrics as they're just for internal tracking
        if not values or metric_name in cross_task_metrics_with_info.values():
            continue
            
        min_val = min(values)
        max_val = max(values)
        mean_val = statistics.mean(values)
        median_val = statistics.median(values)
        
        # Format values based on their type
        min_val_str = f"{min_val:.2f}" if isinstance(min_val, float) else str(min_val)
        max_val_str = f"{max_val:.2f}" if isinstance(max_val, float) else str(max_val)
        
        print(f"\n{metric_name.replace('_', ' ').title()}:")
        
        min_task_str = f"(Task: {min_tasks[metric_name]})" if min_tasks.get(metric_name) else ""
        max_task_str = f"(Task: {max_tasks[metric_name]})" if max_tasks.get(metric_name) else ""
        
        print(f"  Minimum: {min_val_str} {min_task_str}")
        print(f"  Median: {median_val:.2f}")
        print(f"  Mean: {mean_val:.2f}")
        print(f"  Maximum: {max_val_str} {max_task_str}")
    
    # Print total number of hunks for autocomplete and autocomplete_completion tasks
    if task_type in [TaskType.AUTOCOMPLETE] and 'num_hunks' in metrics:
        total_hunks = sum(metrics['num_hunks'])
        print(f"\nTotal Num Hunks: {total_hunks}")
    
    # Print unique files and locations
    if task_type in [TaskType.AGENT, TaskType.AUTOCOMPLETE]:
        print(f"\nUnique files modified across all tasks: {len(all_files_modified)}")
    print(f"Unique edit locations across all tasks: {len(all_locations)}")

def count_repo_files_and_lines(repo_name: str) -> tuple[int, int]:
    """Count the number of non-test files and lines of code in a repository.
    
    Excludes:
    - Test files and directories
    - Hidden files and directories
    - Documentation files (.txt, .md, etc.)
    
    Args:
        repo_name (str): Name of the repository
        
    Returns:
        tuple: (file_count, line_count) - number of non-test files and total lines of code
    """
    repo_path = Path(f"liveswebench/repos/{repo_name}")
    if not repo_path.exists():
        print(f"Repository directory not found: {repo_path}")
        return 0, 0
    
    test_patterns = ['test', 'tests', 'e2e', 'testing']
    doc_extensions = ['.txt', '.md', '.rst', '.markdown', '.doc', '.docx', '.pdf']
    file_count = 0
    line_count = 0
    
    for root, dirs, files in os.walk(repo_path):
        # Skip hidden directories and test directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and 
                  not any(pattern in d.lower() for pattern in test_patterns)]
        
        rel_path = os.path.relpath(root, repo_path)
        
        # Skip test directories at any level
        if any(pattern in rel_path.lower() for pattern in test_patterns):
            continue
            
        for file in files:
            file_path = os.path.join(root, file)
            
            # Skip hidden files
            if file.startswith('.'):
                continue
                
            # Skip test files
            if any(pattern in file.lower() for pattern in test_patterns):
                continue
                
            # Skip documentation files
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext in doc_extensions:
                continue
                
            file_count += 1
            
            # Count lines in the file
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    file_lines = sum(1 for _ in f)
                    line_count += file_lines
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
    
    return file_count, line_count

def get_repo_size_stats():
    """Calculate size statistics for all repositories.
    
    Returns:
        dict: Repository size statistics
    """
    repo_names = [r.name for r in get_all_repos()]
    repo_stats = {}
    
    file_counts = []
    line_counts = []
    
    print("\nAnalyzing repository sizes...")
    for repo_name in sorted(repo_names):
        file_count, line_count = count_repo_files_and_lines(repo_name)
        repo_stats[repo_name] = {
            'files': file_count,
            'lines': line_count
        }
        file_counts.append(file_count)
        line_counts.append(line_count)
        print(f"  {repo_name}: {file_count} files, {line_count} lines of code")
    
    # Calculate aggregate statistics
    if file_counts:
        min_files = min(file_counts)
        max_files = max(file_counts)
        mean_files = statistics.mean(file_counts)
        median_files = statistics.median(file_counts)
        
        min_lines = min(line_counts)
        max_lines = max(line_counts)
        mean_lines = statistics.mean(line_counts)
        median_lines = statistics.median(line_counts)
        
        # Find repos with min/max values
        min_files_repo = next(repo for repo, stats in repo_stats.items() if stats['files'] == min_files)
        max_files_repo = next(repo for repo, stats in repo_stats.items() if stats['files'] == max_files)
        min_lines_repo = next(repo for repo, stats in repo_stats.items() if stats['lines'] == min_lines)
        max_lines_repo = next(repo for repo, stats in repo_stats.items() if stats['lines'] == max_lines)
        
        print("\nRepository Size Statistics:")
        print("\nNon-Test Files Count:")
        print(f"  Minimum: {min_files} files (Repo: {min_files_repo})")
        print(f"  Median: {median_files:.2f} files")
        print(f"  Mean: {mean_files:.2f} files")
        print(f"  Maximum: {max_files} files (Repo: {max_files_repo})")
        
        print("\nLines (Non-Test):")
        print(f"  Minimum: {min_lines} lines (Repo: {min_lines_repo})")
        print(f"  Median: {median_lines:.2f} lines")
        print(f"  Mean: {mean_lines:.2f} lines")
        print(f"  Maximum: {max_lines} lines (Repo: {max_lines_repo})")
    
    return {
        'repo_stats': repo_stats,
        'aggregates': {
            'files': {
                'min': min_files if file_counts else 0,
                'max': max_files if file_counts else 0,
                'mean': mean_files if file_counts else 0,
                'median': median_files if file_counts else 0,
                'min_repo': min_files_repo if file_counts else None,
                'max_repo': max_files_repo if file_counts else None
            },
            'lines': {
                'min': min_lines if line_counts else 0,
                'max': max_lines if line_counts else 0,
                'mean': mean_lines if line_counts else 0,
                'median': median_lines if line_counts else 0,
                'min_repo': min_lines_repo if line_counts else None,
                'max_repo': max_lines_repo if line_counts else None
            }
        }
    }

def main():
    parser = argparse.ArgumentParser(description="Analyze task patch files and generate statistics")
    parser.add_argument('--repo_name', type=str, help='Name of the repository (optional, if not provided will analyze all repositories)')
    parser.add_argument('--task_num', type=int, help='Task number (optional, if not provided will analyze all tasks)')
    parser.add_argument('--task_type', choices=list(TaskType), default=TaskType.AGENT, type=TaskType,
                      help='Type of task to analyze (agent, edit, autocomplete, or all)')
    parser.add_argument('--repo_stats', action='store_true',
                      help='Include repository size statistics (counts files and lines)')
    parser.add_argument('--task_source', type=str, choices=['local', 'huggingface'], default='huggingface',
                      help='Source of tasks, either local or from huggingface')
    
    args = parser.parse_args()
    
    # Always calculate repository statistics if specifically requested
    if args.repo_stats:
        repo_size_stats = get_repo_size_stats()

    tasks = load_tasks(task_source=args.task_source)
    
    if args.repo_name and args.task_num:
        # Analyze a single task
        if args.task_type == 'all':
            # Analyze all task types for this task
            for task_type in list(TaskType):
                stats = analyze_single_task(tasks[args.repo_name][args.task_num], task_type)
                if not stats:
                    print(f"No {task_type} task found for {args.repo_name}/{args.task_num}")
        else:
            # Analyze a specific task type
            stats = analyze_single_task(tasks[args.repo_name][args.task_num], args.task_type)
            if not stats:
                sys.exit(1)
    elif args.repo_name:
        # Analyze all tasks in the specified repository
        results = analyze_repo_tasks(tasks[args.repo_name], args.task_type)
        
        # Calculate repository statistics if analyzing a single repo
        if args.repo_stats or (not args.task_num and not args.repo_stats):
            file_count, line_count = count_repo_files_and_lines(args.repo_name)
            print(f"\nRepository Size - {args.repo_name}:")
            print(f"  Non-Test Files: {file_count}")
            print(f"  Lines of Code: {line_count}")
            
        if not results:
            sys.exit(1)
    else:
        # Analyze all tasks in all repositories
        results = analyze_tasks(tasks, args.task_type)
        
        repo_size_stats = get_repo_size_stats()
            
        if not results:
            sys.exit(1)

if __name__ == "__main__":
    main()
