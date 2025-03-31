#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path
import time


def main():
    
    # Get the default workspace path (two directories up from the current script)
    workspace_path = Path(__file__).resolve().parent.parent.parent.parent
    
    # Ensure the workspace path exists
    workspace_path = Path(workspace_path).resolve()
    if not workspace_path.exists() or not workspace_path.is_dir():
        print(f"Error: Workspace path '{workspace_path}' does not exist or is not a directory", file=sys.stderr)
        sys.exit(1)
    
    print(f"Using workspace: {workspace_path}")
    
    # Docker container name
    container_name = "claude-code-dev"
    
    # Check if container already exists
    result = subprocess.run(
        ["docker", "ps", "-a", "--filter", f"name={container_name}", "--format", "{{.Names}}"],
        capture_output=True,
        text=True
    )
    
    if container_name in result.stdout:
        print(f"Container '{container_name}' already exists. Stopping and removing...")
        subprocess.run(["docker", "rm", "-f", container_name], check=True)
    
    # Build the Docker image
    dockerfile_path = Path(__file__).resolve().parent / "Dockerfile"
    image_name = "claude-code-image"

    print("Building Docker image...")
    build_command = ["docker", "build", "-t", image_name, "-f", str(dockerfile_path), "."]
    
    try:
        subprocess.run(
            build_command,
            cwd=str(Path(__file__).resolve().parent),
            check=True
        )
    except subprocess.CalledProcessError as e:
        print("\n======= Docker Build Failed =======", file=sys.stderr)
        print("The Docker build process failed. This might be due to network connectivity issues.", file=sys.stderr)
        print("\nDetailed error:", file=sys.stderr)
        print(f"{e}", file=sys.stderr)
        sys.exit(1)
    
    # Run the Docker container
    print("Starting Docker container...")
    try:
        subprocess.run([
            "docker", "run", 
            "-d",                              # Run in detached mode
            "-t",                              # Allocate a pseudo-TTY to keep container alive
            "--restart=unless-stopped",        # Automatically restart the container unless explicitly stopped
            "--name", container_name,         
            "-v", f"{workspace_path}:/workspace",  # Mount the workspace
            "--cap-add=NET_ADMIN",            # Required for firewall setup
            image_name
        ], check=True)
        
        # Verify the container is actually running
        print("Verifying container is running...")
        # Add a small delay to give Docker time to start or fail the container
        time.sleep(2)
        
        # First check if the container exists with any status
        container_check = subprocess.run(
            ["docker", "ps", "-a", "--filter", f"name=^{container_name}$", "--format", "{{.Status}}"],
            capture_output=True,
            text=True,
            check=True
        )
        
        status_output = container_check.stdout.strip()
        print(f"Container status: '{status_output}'")
        
        if not status_output:
            print("\n======= Container Not Found =======", file=sys.stderr)
            print(f"Container '{container_name}' does not exist after creation attempt.", file=sys.stderr)
            print("This might indicate a Docker daemon issue.", file=sys.stderr)
            sys.exit(1)
        
        # Check specifically for running containers with exact name match
        running_check = subprocess.run(
            ["docker", "ps", "--filter", f"name=^{container_name}$", "--filter", "status=running", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            check=True
        )
        
        running_output = running_check.stdout.strip()
        print(f"Running container check result: '{running_output}'")
        
        if running_output == container_name:
            # Container is running - print useful commands
            print("\n" + "="*50)
            print("Container started successfully!")
            print("="*50 + "\n")
            
            print("To enter the container's shell:")
            print(f"  docker exec -it {container_name} zsh\n")
            
            print("To stop and destroy the container:")
            print(f"  docker rm -f {container_name}\n")
        else:
            print("\n======= Container Failed to Start =======", file=sys.stderr)
            print("The container was created but is not running.", file=sys.stderr)
            
            # Automatically fetch and display container logs
            print("\nContainer logs:", file=sys.stderr)
            print("-" * 50, file=sys.stderr)
            try:
                logs = subprocess.run(
                    ["docker", "logs", container_name],
                    capture_output=True,
                    text=True
                )
                if logs.stdout.strip() or logs.stderr.strip():
                    if logs.stdout.strip():
                        print(logs.stdout, file=sys.stderr)
                    if logs.stderr.strip():
                        print(logs.stderr, file=sys.stderr)
                else:
                    print("No logs available. Container may have failed to start completely.", file=sys.stderr)
            except Exception as log_e:
                print(f"Could not retrieve logs: {log_e}", file=sys.stderr)
            
            print("-" * 50, file=sys.stderr)
            print("\nAdditional troubleshooting commands:", file=sys.stderr)
            print(f"  docker inspect {container_name}", file=sys.stderr)
            print(f"  docker rm {container_name}", file=sys.stderr)
            sys.exit(1)
        
    except subprocess.CalledProcessError as e:
        print("\n======= Failed to Start Container =======", file=sys.stderr)
        print(f"Error starting container: {e}", file=sys.stderr)
        print("Check Docker logs for more details: docker logs", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
