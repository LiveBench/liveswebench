def get_aider_command(prompt_file: str, workspace_base: str, llm_model: str, llm_api_key: str, files_to_edit: list[str] = []) -> list[str]:

    command = [
        'aider',
        '--message-file',
        prompt_file,
        '--no-auto-commits',
        '--model',
        llm_model,
        '--anthropic-api-key',
        llm_api_key,
        '--no-gitignore',
        '--no-detect-urls',
        '--yes-always'
    ]

    if len(files_to_edit) > 0:
        for file in files_to_edit:
            command.append('--file')
            command.append(file)

    return command, workspace_base