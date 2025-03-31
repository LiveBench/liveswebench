# LiveSWEBench Agents

This subpackage contains code for evaluating LiveSWEBench using command-line agent tools. It contains a barebones implementation of evaluating agent and edit tasks with Aider, OpenHands, SWE-Agent, and Claude Code. See the claude_code folder for instructions about setting up and using Claude Code.

With the agent set up, you can run it using `python3 -m liveswebench.agents.run_agent --repo_name <repo_name> --task_num <task_num> --task_type <task_type> --agent_name <agent_name>`.

## SWE-Agent Notes

SWE-Agent requires some additional setup. First, Python 3.11 must be used. Additionally, you must initialize the SWE-agent git submodule using `git submodule update --init` in the root of this repository. Then, you should setup a new virtual environment within the SWE-agent folder and install dependencies.