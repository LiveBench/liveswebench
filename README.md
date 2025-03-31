# LiveSWEBench

LiveSWEBench is a benchmark for evaluating the utility of AI coding assistants in real-world software engineering tasks, at varying levels of developer involvement. Given a real-world codebase and issue, we investigate the following questions:
 - How useful are AI coding assistants at completing tasks with no developer involvement?
 - How useful are AI coding assistants at completing tasks with some developer involvement (i.e. writing prompts)?
 - How useful are AI coding assistants at aiding in the completion of tasks with high developer involvement (i.e. writing code)?

Specifically, we evaluate the following task types:
 - Autonomous agentic engineering, where the AI assistant is given the raw issue description and must complete the task fully autonomously (á la SWE-Bench)
 - Prompted editing, where the AI assistant is told the file to edit and given a natural language description of the changes to make (specifically in that file)
 - Autocomplete, where the AI assistant reduces the amount of code the developer needs to write by suggesting inline completions

These tasks cover the most common use cases for AI coding assistants and provide a representative result for the utility of each tool.

## Setup

```bash
git clone https://github.com/liveswebench/liveswebench.git
cd liveswebench
python3 -m venv .venv # recommended
source .venv/bin/activate # recommended
pip install -e .
```

If you'd like to be able to automatically launch the relevant tool when preparing a task, follow the instructions to install [vscli](https://github.com/michidk/vscli).

## Usage

<!-- ### Task Generation -->

### Agentic Tasks

#### IDE Agents

These instructions are for agents that are integrated into an IDE and must be prompted manually (e.g. Cursor, Github Copilot).
Agents that can be run from the command line are discussed in the next section.

To evaluate an agentic task, first prepare the task by running
```bash
python -m liveswebench.harness.prepare --repo_name <repo_name> --task_num <task_num> --print-prompt
```
This will clone the relevant repository to the `liveswebench/repos` directory, check out the relevant commit, and create a branch for it.
The `--print-prompt` flag means it will also print the prompt that should be used for the task.

When this is done, open the IDE to the repository. **Note: You must open the IDE to a new window and set that window to the repository root, NOT the `liveswebench` directory.** Otherwise, the agent will get confused or will be able to peek at the ground truth task solution.

If you have `vscli` installed, you can add the argument `--tool_name <tool_name>`; this will cause the tool (e.g. cursor) to be automatically opened to the correct folder once task preparation is complete.

Activate the agent chat in the IDE and paste the prompt into the chat. Let the agent do its thing, granting permission to edit files or run terminal commands when necessary. When it is finished, accept all the suggested edits. If the agent asks for anything, respond along the lines of "Please complete the task to the best of your ability."

Once it is finished, you can run the following command to generate a patch file of the changes:
```bash
python -m liveswebench.harness.generate_patch --repo_name <repo_name> --task_num <task_num> --tool_name <tool_name>
```
This will generate a patch file at `liveswebench/data/results/<repo_name>/<task_num>/<tool_name>/<tool_name>_patch_<timestamp>.patch` 
(`<timestamp>` is the current date in YYYYMMDD format). Check over this file to make sure it is correct, i.e. that the changes align with what was output by the agent as it was running.

Finally, to evaluate the solution, run
```bash
python -m liveswebench.harness.test --repo_name <repo_name> --task_num <task_num> --tool_name <tool_name>
```
This will rerun the prepare script but additionally install the repository dependencies. Then, it will apply the patch file and run the repository unit tests.
The standard output and error of the test will be printed to the terminal and also saved to `liveswebench/data/results/<repo_name>/<task_num>/<tool_name>/<tool_name>_<timestamp>.log`. The `<timestamp>` in the log filename will correspond to the timestamp of the patch file that was used for the evaluation.

Currently, there's no automated method to identify test failures from the logs. So, at this point, it is necessary to manually check the logs to see whether the task was completed successfully. You can compare with the `liveswebench/data/results/<repo_name>/<task_num>/baseline/gold.log` file to see what the test output was for the gold solution (the actual patch from the pull request).

#### CLI / Headless Agents

We have provided implementations of evaluation code for several headless agents in the `liveswebench/agents` directory. Follow the instructions there to run those agents and generate patches. Once the patch is generated, the rest of the process is the same as the IDE agents.

### Prompted Editing

The process for the prompted editing tasks is effectively the same as the agentic tasks. The main adjustment is that scripts should be run with the `--task_type edit` flag.

To prepare the task, run
```bash
python -m liveswebench.harness.prepare --repo_name <repo_name> --task_num <task_num> --task_type edit --print-prompt
```

This will print the edit prompt as well as the name of the file that should be edited. When you open the IDE, you should open up the file that is specified. Make sure that this file is added as context in the assistant chat window. Then, copy and paste the edit prompt into the chat window. Notably, the "Agent" mode of whatever tool is being evaluated should still be used for this task; the edit prompts do not include all potentially relevant context from other files, so it will still be necessary for the tool to discover context automatically. It would have been unfair to compare tools with no tool use capability (e.g. Cursor in its "Edit" mode) with full agents (e.g. OpenHands) on this task. If you'd like to evaluate the non-agentic mode of a tool, you may do so with this task, but should not expect comparable results to the leaderboard.

To generate a patch, run
```bash
python -m liveswebench.harness.generate_patch --repo_name <repo_name> --task_num <task_num> --tool_name <tool_name> --task_type edit
```

This will generate a patch file at `liveswebench/data/results/<repo_name>/<task_num>/<tool_name>/<tool_name>_edit_patch_<timestamp>.patch`. Check over this file to make sure it is correct, i.e. that the changes align with what was output by the agent as it was running.

To evaluate the solution, run
```bash
python -m liveswebench.harness.test --repo_name <repo_name> --task_num <task_num> --tool_name <tool_name> --task_type edit
```

Again, when the test is completed you should inspect the logs to see if the task was completed successfully.

### Autocomplete

The process for the autocomplete task is similar to that of the edit task. Again, a primary adjustment is the use of the `--task_type autocomplete` flag:

To prepare the task, run
```bash
python -m liveswebench.harness.prepare --repo_name <repo_name> --task_num <task_num> --task_type autocomplete --print-prompt
```

This will prepare the repository state for the task. The autocomplete task is unique in the way the repository state is prepared. The file `autocomplete_patch.patch` for the task contains all the hunks of the gold patch that will be recreated using the autocomplete tool. Each hunk contains additions and deletions, and the goal of the task is for the tool to reconstruct the additions. So, all the deletions are applied during this prepare phase (so that the former code does not confuse the autocomplete model). In addition, changes outside of the autocomplete hunks will be applied automatically.

There is also a file `autocomplete_prompts.txt` which contains a prompt for each hunk in `autocomplete_patch.patch`.

When the `--print-prompt` flag is passed to the prepare script, the addition hunks and their prompts will be printed out in the following format:

```
FILE:
/Users/gabriel/liveswebench/liveswebench/repos/freeCodeCamp/api/src/schemas/certificate/cert-slug.ts
ORIGINAL ADDITION (Near line 85):
         certSlug: Type.Enum(Certification),
         certTitle: Type.String(),
         username: Type.String(),
+        name: Type.Optional(Type.String()),
         date: Type.Number(),
         completionTime: Type.Number()
       }),
PROMPT:
name:
```

FILE is an absolute path to the file being edited. ORIGINAL ADDITION is the original hunk from the gold patch, with the removals already applied (so only the added lines are shown). PROMPT is a snippet or set of line snippets, one for each addition or group of additions from ORIGINAL ADDITION (in general, it will be one snippet in PROMPT for each *statement* (e.g. variable assginment or method call) in ORIGINAL ADDITION). To evaluate the task, the file should be opened in the tool. Then, you can scroll to find the place where the addition should be. Place the cursor in the line prior to the addition in ORIGINAL ADDITION and press enter. Then, paste the snippet from PROMPT and press tab/enter to accept suggestions. If there are mulitple snippets in PROMPT, do this for each of them. Sometimes, the autocomplete model will predict the next snippets in one shot; in such cases you can then skip pasting those in.

Once the evaluation is finished, run 
```bash
python -m liveswebench.harness.generate_patch --repo_name <repo_name> --task_num <task_num> --tool_name <tool_name> --task_type autocomplete
```
to generate the patch file of the form `<tool_name>_autocomplete_patch_<timestamp>.patch`.

To evaluate the solution, run
```bash
python -m liveswebench.harness.test --repo_name <repo_name> --task_num <task_num> --tool_name <tool_name> --task_type autocomplete
```

Note: some repo test scripts rely on Docker for test runs. This means that in order to run tests, you should have the docker daemon running.

## Tool-Specific Instructions

### Github Copilot

Github Copilot's agent mode is in preview and currently only available on the VSCode Insiders build, so you will need to install that to use this mode.

### Cursor

Be sure to enable codebase indexing in the settings. When you open up a repository, wait to send the prompt until the codebase has been indexed.