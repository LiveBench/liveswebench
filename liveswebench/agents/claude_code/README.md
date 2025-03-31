# LiveSWEBench Claude Code Setup

To enable uninterrupted operation of Claude Code, it is necessary to run the agent within a firewall-limited Docker container.

1. Run `python3 -m liveswebench.agents.claude_code.start_docker` to setup and start the container
2. Open the container shell with `docker exec -it claude-code-dev bash`
3. Run `source docker/.venv/bin/activate` to activate the virtual environment within the container
4. Run `sudo /usr/local/bin/init-firewall.sh` to initialize the firewall (blocking all network destinations except those necessary for Claude)
5. Run `claude` once to authenticate with your Anthropic account
6. Run `claude --dangerously-skip-permissions` to accept the agreement
7. Finally, you can run `python3 -m liveswebench.agents.run_agent --agent_name claude-code`