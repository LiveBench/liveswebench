from pathlib import Path


def get_swe_agent_command(prompt_file, workspace_base, llm_model, llm_api_key, files_to_edit=None):

    if 'claude' in llm_model:
        llm_model = 'anthropic/' + llm_model

    command = [
        "source",
        ".venv/bin/activate",
        "&&",
        "sweagent",
        "run",
        "--agent.model.name=" + llm_model,
        "--agent.model.api_key=" + llm_api_key,
        "--env.repo.path=" + workspace_base,
        "--problem_statement.path=" + prompt_file,
        "--actions.apply_patch_locally=true",
    ]

    return command, Path(__file__).parent / 'SWE-agent'