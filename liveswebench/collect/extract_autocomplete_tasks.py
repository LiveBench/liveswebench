import argparse
import json
import os
import re
from anthropic import Anthropic
from liveswebench.collect.process_patch import extract_qualifying_hunks
from dotenv import load_dotenv
from liveswebench.harness.util import extract_hunks_from_patch
from liveswebench.util.tasks import TaskInstance, TaskType, load_tasks
from liveswebench.util.repo import get_repo, get_all_repos
from liveswebench.util.util import BASE_TASK_PATH

load_dotenv()

HUNK_CLASSIFICATION_PROMPT = """You will be analyzing a git patch and classifying an individual hunk based on specific criteria. You will be provided with two pieces of information :

1. A full git patch containing changes to multiple files across multiple hunks:
<full_patch>
{full_patch}
</full_patch>

2. An individual hunk from the patch:
<individual_hunk>
{individual_hunk}
</individual_hunk>

Your task is to classify the individual hunk based on the following criteria:

1. Well-definedness of variable definitions and usages:
   - A variable definition is considered well-defined if it is strongly implied by its name (e.g., `const isMacOS = navigator.userAgent.includes('Mac OS')` is well-defined, while `const x = navigator.userAgent.includes('Mac OS')` is not).
   - A variable usage is considered well-defined if, given the context of where the variable is defined and knowledge about approximately where it is being used, it is clearly implied by the variable's name and definition.

Note: "variable" here includes functions. In other words, if the hunk contains a function definition, then well-definedness means that the function parameters and definition are well-implied by the function name. Similarly, function usages should be well-implied by function names.

2. Location of needed information:
   - The individual hunk itself
   - The current file within the full patch
   - Multiple files within the full patch

Note: when discussing needed information, we're specifically talking about *changes* made in all the above listed locations. e.g. if you classify a hunk as file-context, you are saying that other *changes* (additions) made in the patch in the same file are relevant. If the information needed for the hunk comes from other parts of the current file that are unchanged in the patch, this would not move the hunk to the file-context category. In other words, you're asking the following questions:
1. would i need to apply the changes made to the current file in the patch before being able to implement this hunk?
2. would i need to apply the changes made to all files in the patch before being able to implement this hunk?
Think from the perspective of if you were making the change to the new hunk. Would you need to directly reference the changes made to other parts of the current file or other files in the patch?

Consider carefully whether the changes in other locations are TRULY NECESSARY to implement the change in the current hunk. It's about whether the changes in the current hunk could not be recreated without the context of changes made in the other locations. 

For instance, if the full patch contains a variable definition for a variable that was not previously present, and that variable is used in the current patch, then the change made in the full patch is necessary context; without having applied the full patch there'd be no way of knowing the variable name. In contrast, if the full patch simply modifies the variable definition without changing its name, then that change may not be necessary context for the current hunk (it would depend on whether the change to the usage of the variable in the current hunk is directly related to the change to the variable's definition).

Similarly, if the full patch updates a method to potentially have a different return value, and the current hunk contains an update to be able to handle this new return value, then the full patch becomes necessary context, as it would not be possible to know what the new return value could be without referencing the changes made in the full patch.

As another example, you can consider the separation between a C++ header and implementation file. Headers tend to only include function prototypes, without implementations, while the implementation files contain the actual implementations. However, assuming the function is well-named, it wouldn't be necessary to see the actual implementation in order to write the prototype, so changes made to the implementation would not be necessary context. In other words, it would not be necessary to apply changes to the implementation file in order to be able to make the changes in the current hunk. In contrast, if the changes made to the implementation file and the current hunk constitute a renaming of the function, then the implementation file would indeed be necessary context, as a developer implementing the change to the current hunk would need to reference the new name from the implementation file to know the new name of the function.

You can see how the question of which changes must be applied to be able to implement the current hunk is a useful proxy for determining which context is necessary.

If, given all context possible, it would still not be obvious how to make the change (without being told what the change is), then the hunk should be classified as having undefined definitions or usages. For instance, if the hunk contains a change to make a variable value optional or update some magic string, and no other changes in the full patch make the optionality of the variable or the new magic string obvious, then the hunk should be classified as having an undefined definition.

To complete this task, follow these steps:

1. Carefully examine the individual hunk provided.

2. Determine if the variable definitions and usages in the hunk are well-defined based on the criteria provided above.

3. If the variables are well-defined, identify where the needed information to implement the change is located (individual hunk or full patch).

4. Use the following classification system to categorize the hunk:
   a. Self-contained: The hunk contains all needed information with well-defined variables.
   d. File context: Well-defined, but requires context from the current file in the full patch.
   e. Multi-file context: Well-defined, but requires context from multiple files in the full patch.
   f. Undefined definitions: Variable definitions are not well-defined.
   g. Undefined usages: Variable usages are not well-defined.

5. Provide a justification for your classification, explaining your reasoning and citing specific examples from the hunk and, if necessary, the full patch.

6. State your final classification.

Present your analysis in the following format:

<analysis>
<justification>[Provide your detailed justification here]</justification>

<classification>[State the final classification here]</classification>
<relevant_hunks>[Provide the headers of the relevant hunks]</relevant_hunks>
</analysis>

When providing hunk headers you do not need to include the file header; just include the line of the form "@@ -x,y +a,b @@ c" for each hunk besides the INDIVIDUAL_HUNK that contains changes that you believe are necessary in order to be able to understand and implement the INDIVIDUAL_HUNK. Provide the headers in a comma-separated list, enclosing each in quotation marks.

Remember, non-obvious variable definitions and usages override other classifications. If the hunk contains poorly defined variables, that should be your primary classification regardless of the location of needed information.
Note that changes to strings generally would cause a classification of undefined definitions, as the specific value of the string is not obvious."""

CLASSIFICATIONS = [
    "self-contained",
    "file context",
    "multi-file context",
    "undefined definitions",
    "undefined usages"
]

def classify_hunk(full_patch: str, individual_hunk: str) -> dict[str, str]:
    anthropic = Anthropic()
    response = anthropic.messages.create(
        model="claude-3-5-sonnet-20240620",
        messages=[
            {"role": "user", "content": HUNK_CLASSIFICATION_PROMPT.format(full_patch=full_patch, individual_hunk=individual_hunk)}
        ],
        max_tokens=1000
    )

    text = response.content[0].text

    justification = re.search(r'<justification>(.*?)</justification>', text, re.DOTALL).group(1)
    classification = re.search(r'<classification>(.*?)</classification>', text, re.DOTALL).group(1).lower()
    assert classification in CLASSIFICATIONS
    relevant_hunks = re.search(r'<relevant_hunks>(.*?)</relevant_hunks>', text, re.DOTALL).group(1)

    return {
        "justification": justification,
        "classification": classification,
        "relevant_hunks": relevant_hunks
    }

def create_filtered_patch(task: TaskInstance, classifications: dict[str, dict[str, dict[str, str]]]) -> str:
    """
    Create a filtered patch file containing only self-contained and file context hunks.
    
    Args:
        task: The task object
        classifications: Dictionary of classifications for each hunk
        
    Returns:
        String containing the filtered patch
    """
    autocomplete_patch = task.get_ground_truth_patch(TaskType.PROMPTED_AUTOCOMPLETE)
    autocomplete_patch_files_to_hunks = extract_hunks_from_patch(autocomplete_patch)
    
    filtered_patch = ""
    
    for file_header in autocomplete_patch_files_to_hunks:
        file_info = autocomplete_patch_files_to_hunks[file_header]
        file_path = file_info["file_path"]
        hunks = file_info["hunks"]
        
        # Check if any hunks in this file are self-contained or file context
        valid_hunks = []
        for hunk in hunks:
            hunk_header = hunk.split('\n')[0]
            if hunk_header in classifications.get(file_path, {}) and \
               classifications[file_path][hunk_header]["classification"] in ["self-contained", "file context"]:
                valid_hunks.append(hunk)
        
        # If we have valid hunks, add the file header and hunks to the patch
        if valid_hunks:
            filtered_patch += file_header + '\n'
            for hunk in valid_hunks:
                filtered_patch += hunk + '\n'
    
    return filtered_patch

def extract_autocomplete_tasks(task: TaskInstance, model: str = "claude-3-5-sonnet-20241022") -> dict[str, dict[str, dict[str, str]]] | None:
    substantive_full_patch_hunks = extract_qualifying_hunks(task.gold_patch, max_additions=None, require_contiguous_additions=False, llm_check=False)
    substantive_full_patch = ''
    for file_header, hunks in substantive_full_patch_hunks.items():
        substantive_full_patch += file_header + '\n'
        for hunk in hunks:
            substantive_full_patch += hunk + '\n'

    autocomplete_patch = task.prompted_autocomplete_patch
    if autocomplete_patch is None:
        print(f"No prompted autocomplete patch found for task {task.task_num} in repo {task.repo_name}")
        return
    autocomplete_patch_files_to_hunks = extract_hunks_from_patch(autocomplete_patch)

    classifications = {}

    for file_header in autocomplete_patch_files_to_hunks:
        file_info = autocomplete_patch_files_to_hunks[file_header]
        file_path = file_info["file_path"]
        hunks = file_info["hunks"]
        classifications[file_path] = {}
        print(f'Classifying {len(hunks)} hunks in {file_path}')
        for hunk in hunks:
            hunk_header = hunk.split('\n')[0]
            classification = classify_hunk(substantive_full_patch, hunk)
            classifications[file_path][hunk_header] = classification
    
    return classifications

def main(tasks: dict[str, dict[int, TaskInstance]], repo_name: str | None = None, task_num: int | None = None, model: str = "claude-3-5-sonnet-20241022", reclassify: bool = False, skip_classification: bool = False):
    assert repo_name is not None or task_num is None # can't specify task num without repo name

    if repo_name is None:
        repos = get_all_repos()
    else:
        repos = [get_repo(repo_name)]
    
    if not skip_classification:
        tasks_to_classify: list[TaskInstance] = []
        if task_num is not None:
            assert repo_name is not None
            tasks_to_classify.append(tasks[repo_name][task_num])
        else:
            for repo in repos:
                tasks_to_classify.extend([t for t in tasks[repo.name].values()])

        if len(tasks_to_classify) == 0:
            print(f"No tasks found")
        
        for task in tasks_to_classify:
            classifications_path = BASE_TASK_PATH / task.repo_name / str(task.task_num) / 'classifications.json'
            if not reclassify and os.path.exists(classifications_path):
                print(f"Skipping classification for task {task.task_num} in repo {task.repo_name} because it already has classifications")
                continue

            print(f"Extracting autocomplete tasks for task {task.task_num} in repo {task.repo_name}")
            classifications = extract_autocomplete_tasks(task.repo_name, task.task_num, model)
            if classifications is None:
                print(f"No classifications created for task {task.task_num} in repo {task.repo_name}")
                continue
            classification_counts = {}
            for file_path in classifications:
                for hunk_header in classifications[file_path]:
                    classification = classifications[file_path][hunk_header]["classification"]
                    classification_counts[classification] = classification_counts.get(classification, 0) + 1
            print(f"Task {task.task_num} in repo {task.repo_name}:")
            for classification in classification_counts:
                print(f"  {classification}: {classification_counts[classification]}")
            print()

            task_path = BASE_TASK_PATH / task.repo_name / str(task.task_num)
            with open(classifications_path, 'w') as f:
                json.dump(classifications, f)

    # Generate filtered patch files for each task
    for repo in repos:
        for task in tasks[repo.name].values():
            task_path = BASE_TASK_PATH / task.repo_name / str(task.task_num)
            classifications_path = task_path / 'classifications.json'
            if not os.path.exists(classifications_path):
                continue
                
            # Read the classifications
            with open(classifications_path, 'r') as f:
                classifications = json.load(f)
                
            # Create filtered patch
            filtered_patch = create_filtered_patch(task, classifications)
            if filtered_patch == "":
                print(f"No valid hunks found for task {task.task_num} in repo {task.repo_name}")
                continue
            
            # Write the filtered patch to a file
            with open(task_path / 'autocomplete_patch.patch', 'w') as f:
                f.write(filtered_patch)
                
            print(f"Created filtered patch for task {task.task_num} in repo {task.repo_name}")

    full_classification_counts = {}
    full_task_count = 0
    for repo in repos:
        repo_classification_counts = {}
        repo_task_count = 0
        for task in tasks[repo.name].values():
            valid = False
            task_path = BASE_TASK_PATH / task.repo_name / str(task.task_num)
            classifications_path = task_path / 'classifications.json'
            if not os.path.exists(classifications_path):
                continue
            with open(classifications_path, 'r') as f:
                classifications = json.load(f)
            for file_path in classifications:
                for hunk_header in classifications[file_path]:
                    classification = classifications[file_path][hunk_header]["classification"]
                    repo_classification_counts[classification] = repo_classification_counts.get(classification, 0) + 1
                    full_classification_counts[classification] = full_classification_counts.get(classification, 0) + 1
                    if classification in ["self-contained", "file context"]:
                        valid = True
            if valid:
                repo_task_count += 1
                full_task_count += 1
        if len(repos) > 1:
            print(f"Repo {repo.name}:")
            for classification in repo_classification_counts:
                print(f"  {classification}: {repo_classification_counts[classification]}")
            print(f"  {repo_task_count} / {full_task_count} tasks valid")
            print()

    print("Full classification counts:")
    for classification in full_classification_counts:
        print(f"  {classification}: {full_classification_counts[classification]}")
    print(f"  {full_task_count} tasks valid")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract autocomplete tasks")
    parser.add_argument("--repo_name", type=str, required=False)
    parser.add_argument("--task_num", type=int, required=False)
    parser.add_argument("--reclassify", action="store_true")
    parser.add_argument("--skip-classification", action="store_true")
    parser.add_argument("--model", type=str, required=False, default="claude-3-5-sonnet-20241022")
    args = parser.parse_args()
    
    tasks = load_tasks()
    
    main(tasks, args.repo_name, args.task_num, args.model, args.reclassify, args.skip_classification)

    print(f"Completed extraction of autocomplete tasks")