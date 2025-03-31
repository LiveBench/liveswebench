import argparse
import os
from pathlib import Path
from liveswebench.util.util import BASE_TASK_PATH
import openai
import anthropic
from dotenv import load_dotenv

load_dotenv()

DIFF_DESCRIPTION_PROMPT = """
You are an expert software engineer. You will be given a Github issue description and a patch for the issue.
You will then be given a section of the patch.
Given the issue description and the overall patch, your job is to write a concise description of the changes made in the section of the patch.

Think about it this way: given the original issue description, and what has been done so far (i.e. everything but the section of the patch), what (at a high level) is left to do?

Write your description as a command that could be given to another person to implement the diff. Your description should be at most three sentences, but less is better.

Assume that all of the overall patch has been applied EXCEPT for the section of the patch you are given.
So, a programmer who reads your description should be able to finish fixing the issue.

example:

<github_issue>
For MacOS, the submit button of the completion modal should display Command + Enter
## Description

The submit button on the completion modal always displays Ctrl + Enter, but it should be Command + Enter if the user is on MacOS.

https://github.com/freeCodeCamp/freeCodeCamp/blob/c7d3b1303e6badfbdabc16de8c95e9f334b2a4b5/client/src/templates/Challenges/components/completion-modal.tsx#L202

The modal can be found on one of these pages:
- https://www.freecodecamp.org/learn/a2-english-for-developers/learn-greetings-in-your-first-day-at-the-office/task-3
- https://www.freecodecamp.org/learn/javascript-algorithms-and-data-structures/basic-javascript/declare-javascript-variables

## Requirements

- Update the `completion-modal.tsx` file to have the key name displayed based on user device
- Update tests in `completion-modal-spec.ts` file to cover the button text display

https://github.com/freeCodeCamp/freeCodeCamp/pull/54276 can be used as reference.

## Additional context

We recently made a similar change to the submit button in the lower jaw: https://github.com/freeCodeCamp/freeCodeCamp/issues/54270.
</github_issue>

<overall_patch>
diff --git a/client/src/templates/Challenges/components/completion-modal.tsx b/client/src/templates/Challenges/components/completion-modal.tsx
index ecdb17835e3925..f3c2f76ba159ba 100644
--- a/client/src/templates/Challenges/components/completion-modal.tsx
+++ b/client/src/templates/Challenges/components/completion-modal.tsx
@@ -164,6 +164,8 @@ class CompletionModal extends Component<
       submitChallenge
     } = this.props;
 
+    const isMacOS = navigator.userAgent.includes('Mac OS');
+
     return (
       <Modal
         onClose={close}
@@ -199,7 +201,9 @@ class CompletionModal extends Component<
             onClick={() => submitChallenge()}
           >
             {isSignedIn ? t('buttons.submit-and-go') : t('buttons.go-to-next')}
-            <span className='hidden-xs'> (Ctrl + Enter)</span>
+            <span className='hidden-xs'>
+              {isMacOS ? ' (Command + Enter)' : ' (Ctrl + Enter)'}
+            </span>
           </Button>
           <Spacer size='xxSmall' />
           {this.state.downloadURL ? (
</overall_patch>

<section_of_patch>
@@ -199,7 +201,9 @@ class CompletionModal extends Component<
             onClick={() => submitChallenge()}
           >
             {isSignedIn ? t('buttons.submit-and-go') : t('buttons.go-to-next')}
-            <span className='hidden-xs'> (Ctrl + Enter)</span>
+            <span className='hidden-xs'>
+              {isMacOS ? ' (Command + Enter)' : ' (Ctrl + Enter)'}
+            </span>
           </Button>
           <Spacer size='xxSmall' />
           {this.state.downloadURL ? (
</section_of_patch>

response:
Show " (Command + Enter)" if the user is on MacOS, otherwise show " (Ctrl + Enter)"

explanation:
In this case, the specific text shown depending on the user's OS is important, so it is included in the description.
However, the specific way in which the code should check for the user's OS is not important, so it is not included in the description.
Notably, it's not important to specify the name of the variable isMacOS, because the programmer will be able to see it as the other code in the patch will have already been applied.

example 2:

<github_issue>
For MacOS, the submit button of the completion modal should display Command + Enter
## Description

The submit button on the completion modal always displays Ctrl + Enter, but it should be Command + Enter if the user is on MacOS.

https://github.com/freeCodeCamp/freeCodeCamp/blob/c7d3b1303e6badfbdabc16de8c95e9f334b2a4b5/client/src/templates/Challenges/components/completion-modal.tsx#L202

The modal can be found on one of these pages:
- https://www.freecodecamp.org/learn/a2-english-for-developers/learn-greetings-in-your-first-day-at-the-office/task-3
- https://www.freecodecamp.org/learn/javascript-algorithms-and-data-structures/basic-javascript/declare-javascript-variables

## Requirements

- Update the `completion-modal.tsx` file to have the key name displayed based on user device
- Update tests in `completion-modal-spec.ts` file to cover the button text display

https://github.com/freeCodeCamp/freeCodeCamp/pull/54276 can be used as reference.

## Additional context

We recently made a similar change to the submit button in the lower jaw: https://github.com/freeCodeCamp/freeCodeCamp/issues/54270.
</github_issue>

<overall_patch>
diff --git a/client/src/templates/Challenges/components/completion-modal.tsx b/client/src/templates/Challenges/components/completion-modal.tsx
index ecdb17835e3925..f3c2f76ba159ba 100644
--- a/client/src/templates/Challenges/components/completion-modal.tsx
+++ b/client/src/templates/Challenges/components/completion-modal.tsx
@@ -164,6 +164,8 @@ class CompletionModal extends Component<
       submitChallenge
     } = this.props;
 
+    const isMacOS = navigator.userAgent.includes('Mac OS');
+
     return (
       <Modal
         onClose={close}
@@ -199,7 +201,9 @@ class CompletionModal extends Component<
             onClick={() => submitChallenge()}
           >
             {isSignedIn ? t('buttons.submit-and-go') : t('buttons.go-to-next')}
-            <span className='hidden-xs'> (Ctrl + Enter)</span>
+            <span className='hidden-xs'>
+              {isMacOS ? ' (Command + Enter)' : ' (Ctrl + Enter)'}
+            </span>
           </Button>
           <Spacer size='xxSmall' />
           {this.state.downloadURL ? (
</overall_patch>

<section_of_patch>
@@ -164,6 +164,8 @@ class CompletionModal extends Component<
       submitChallenge
     } = this.props;
 
+    const isMacOS = navigator.userAgent.includes('Mac OS');
+
     return (
       <Modal
         onClose={close}
</section_of_patch>

response:
Store whether the user is on MacOS in a variable called `isMacOS`.

explanation:
In this case, the name of the variable is relevant for other code in the patch, so it is important to include it in the description.
However, the specific way in which the code should check for the user's OS is not important, so it is not included in the description.

Make sure to exactly include any "magic strings" that could be output during function execution (e.g. error messages).
Make sure to specify any values that should be used exactly in the patch.
However, do not specify exactly how to implement the patch (e.g. control flow or specific operations).
Specify enough information to implement the patch, but not so much that it is obvious what the patch is.
Your description should be somewhat more precise than the original issue description without being so specific that it is obvious what the patch is.
Your description should specify actual code snippets, variable names, or operators ONLY IF ABSOLUTELY NECESSARY, if there is absolutely no way to describe the changes without mentioning specific code.
For instance, if a method is being added in the section of the patch and used elsewhere in the overall patch, you should mention the method name so that its usages will be valid.
Similarly, if the added method is overriding something, you should mention the name of the overridden method, even if no usages are being changed in the overall patch.
Otherwise, describe the changes in natural language in general terms. Pretend like you don't actually know how the issue was fixed but are giving an educated suggestion as to what should be done. 
For instance, rather than saying "use the modulus operator to prevent x from exceeding the maximum", you should just say "wrap to zero to prevent exceeding the maximum".
As another example, if the change being made was to add a specific class to an HTML element under certain conditions, you should use the exact class name but describe the conditions in natural language.
So if during a dragover event an element is given the class "activate", your description should just say "add the class 'activate' during a dragover event" rather than formally specifying the code to check for the event.
This way, another programmer can implement a functionally equivalent patch.

Do not mention filenames or function names in your description.

If the overall patch includes changes to documentation that fully describe the changes in the section of the patch, you could just repeat those changes in your description.
If the overall patch is equal to the section of the patch, describe the entire patch.

<github_issue>
ISSUE_TEXT
</github_issue>

<overall_patch>
OVERALL_PATCH
</overall_patch>

<section_of_patch>
SECTION_OF_PATCH
</section_of_patch>
"""

def generate_edit_prompt(issue_text: str, overall_patch: str, section_of_patch: str, model: str, base_url: str | None) -> str:
    prompt = DIFF_DESCRIPTION_PROMPT.replace("ISSUE_TEXT", issue_text).replace("OVERALL_PATCH", overall_patch).replace("SECTION_OF_PATCH", section_of_patch)
    if 'claude' in model:
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        response = client.messages.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200
        )
        res = response.content[0].text
    else:
        if base_url is not None:
            client = openai.OpenAI(base_url=base_url, api_key="BLANK")
        else:
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        if response is None or response.choices is None or len(response.choices) == 0:
            raise Exception("No response from OpenAI")
        res = response.choices[0].message.content

    return res.split('Explanation:')[0].strip()
    

def generate_edit_task(repo_name: str, task_num: int, model: str, base_url: str | None):
    task_path = BASE_TASK_PATH / repo_name / str(task_num)
    task_problem_statement_file = task_path / "problem_statement.md"
    if not task_problem_statement_file.exists():
        print(f"Problem statement file not found for task {task_num} in repository {repo_name}")
        return
    with open(task_problem_statement_file, "r") as f:
        issue_text = f.read()
    gold_patch_file = task_path / "gold_patch.patch"
    if not gold_patch_file.exists():
        print(f"Gold patch file not found for task {task_num} in repository {repo_name}")
        return
    with open(gold_patch_file, "r") as f:
        gold_patch = f.read()
    edit_patch_file = task_path / "edit_patch.patch"
    if not edit_patch_file.exists():
        print(f"Edit patch file not found for task {task_num} in repository {repo_name}")
        return
    with open(edit_patch_file, "r") as f:
        edit_patch = f.read()
    prompt = generate_edit_prompt(issue_text, gold_patch, edit_patch, model, base_url)

    output_file = task_path / "edit_prompt.txt"
    with open(output_file, "w") as f:
        f.write(prompt)
    print(f"Generated edit prompt for task {task_num} in repository {repo_name}")

def main(repo_name: str | None = None, task_num: int | None = None, model: str = "claude-3-5-sonnet-20241022", base_url: str | None = None, regenerate: bool = False):
    if repo_name is None:
        for repo_name in os.listdir(BASE_TASK_PATH):
            main(repo_name, task_num, model, base_url, regenerate)
        return
    tasks_dir = BASE_TASK_PATH / repo_name
    if not os.path.exists(tasks_dir):
        print(f"No tasks directory found for repository {repo_name}")
        return
    
    if task_num is not None:
        if not os.path.exists(Path(tasks_dir) / str(task_num)):
            print(f"Invalid task {task_num}")
            return
        task_nums = [task_num]
    else:
        task_nums = [int(task_dir) for task_dir in os.listdir(tasks_dir) if os.path.isdir(Path(tasks_dir) / task_dir)]

    task_nums = [n for n in task_nums if os.path.exists(BASE_TASK_PATH / repo_name / str(n) / "edit_patch.patch")]

    if not regenerate:
        task_nums = [n for n in task_nums if not os.path.exists(BASE_TASK_PATH / repo_name / str(n) / "edit_prompt.txt")]

    print(f"Generating edit prompts for tasks {task_nums} in repository {repo_name}")

    for task_num in task_nums:
        generate_edit_task(repo_name, task_num, model, base_url)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_name", type=str, required=False)
    parser.add_argument("--task_num", type=int, required=False)
    parser.add_argument("--model", type=str, required=False, default="claude-3-5-sonnet-20241022")
    parser.add_argument("--base_url", type=str, required=False)
    parser.add_argument("--regenerate", action="store_true")
    args = parser.parse_args()
    main(args.repo_name, args.task_num, args.model, args.base_url, args.regenerate)