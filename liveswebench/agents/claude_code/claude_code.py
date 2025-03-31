def get_claude_code_command(prompt_file, workspace_base, llm_model, llm_api_key, files_to_edit=None):
    """
    Generate the command to run Claude Code agent
    
    Args:
        repo_path: Path to the repository
        model: Model to use (e.g., 'claude-3-7-sonnet-20250219')
        api_key: Anthropic API key
        prompt: Task prompt
        files_to_edit: List of files to edit (for edit tasks)
        
    Returns:
        List of command arguments to run the Claude Code agent
    """
    
    # Set up the command to run Claude Code
    command = [
        "cat",
        prompt_file,
        "|",
        "claude",
        "-p",
        f"\"complete the task\"",
        "--dangerously-skip-permissions",
        "--verbose"
    ]
    
    return command, workspace_base
