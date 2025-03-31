import os
import time
from openai import OpenAI
from dotenv import load_dotenv

from liveswebench.harness.util import extract_hunks_from_patch

# Load environment variables from .env file
load_dotenv()

# OpenAI API configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

def chat_completion_openai(prompt, model="gpt-4o-mini"):
    """
    Call an LLM with a prompt and return the response.
    
    Args:
        prompt (str): The prompt to send to the model
        model (str): The model to use (default: "gpt-4o-mini")
        
    Returns:
        str: The response from the model
    """
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200
    )
    return response.choices[0].message.content.strip()

IS_CODE_CHANGE_PROMPT = """
You are a code analysis assistant. Examine the git patch hunk between <HUNK></HUNK> tags and determine if it contains actual code changes or only changes to comments/documentation/imports.

For the given hunk:
1. Identify the programming language based on syntax
2. Analyze lines marked with + or - (added or removed)
3. Determine if these changes affect functional code or only comments/documentation/imports

Respond with:
- Explanation: Brief justification for your classification
- Changed Elements: List the specific code elements modified (if applicable)
- Classification: "YES" or "NO", enclosed in <CLASSIFICATION></CLASSIFICATION> tags

Example hunk:
<HUNK>
@@ -23,7 +23,7 @@ def calculate_total(items):
     # Loop through all items
-    # and sum their values
+    # and return the sum of their values
     return sum(item.value for item in items)
</HUNK>

Response:
The hunk is classified as "NO" because it only modifies a comment, not actual code.
<CLASSIFICATION>NO</CLASSIFICATION>

<HUNK>
{hunk}
</HUNK>
"""


def is_actual_code_change(hunk_text, model="gpt-4o-mini"):
    """
    Use the model to determine if a hunk contains actual code changes.
    
    Args:
        hunk_text (str): The hunk to check
        model (str): The model to use (default: "gpt-4o-mini")
        
    Returns:
        bool: True if the hunk contains actual code changes, False otherwise
    """
    prompt = IS_CODE_CHANGE_PROMPT.format(hunk=hunk_text)
    
    # Add retry logic for API calls
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = chat_completion_openai(prompt, model)
            if not '<CLASSIFICATION>' in response or not '</CLASSIFICATION>' in response:
                raise Exception(f"No classification found in response: {response}")
            # Parse response - looking for YES or NO
            classification = response.split('<CLASSIFICATION>')[1].split('</CLASSIFICATION>')[0].strip()
            if classification.upper() == "YES":
                return True
            return False
        except Exception as e:
            if attempt < max_retries - 1:
                # Exponential backoff: sleep 2^attempt seconds before retrying
                time.sleep(2 ** attempt)
                continue
            print(f"Failed to check hunk after {max_retries} attempts: {e}")
            return False  # Default to False in case of persistent errors

def has_contiguous_additions(hunk_lines):
    """
    Check if additions in a hunk are contiguous after applying all deletions.
    
    Args:
        hunk_lines (list): Lines of the hunk
        
    Returns:
        bool: True if additions are contiguous (possibly separated by whitespace), False otherwise
    """
    # Skip hunk header lines
    content_lines = []
    for line in hunk_lines:
        if not line.startswith('@@') and not line.startswith('---') and not line.startswith('+++'):
            # Keep context lines and addition lines, remove deletion lines
            if not line.startswith('-'):
                content_lines.append(line)
    
    addition_indices = []
    for i, line in enumerate(content_lines):
        if line.startswith('+'):
            addition_indices.append(i)
    
    if not addition_indices:
        return False
    
    # Check if additions are contiguous (or separated by whitespace)
    for i in range(len(addition_indices) - 1):
        has_non_whitespace = False
        for j in range(addition_indices[i] + 1, addition_indices[i + 1]):
            # If there's a non-whitespace line between additions, they're not contiguous
            if j < len(content_lines) and not content_lines[j].startswith('+'):
                line_content = content_lines[j][1:] if content_lines[j].startswith(' ') else content_lines[j]
                if line_content.strip():  # If there's non-whitespace content
                    has_non_whitespace = True
                    break
        
        if has_non_whitespace:
            return False
    
    return True



def extract_qualifying_hunks(patch_content, max_additions=None, require_contiguous_additions=True, llm_check=True):
    """
    Extract qualifying hunks from a git patch file.

    A qualifying hunk is both substantive and has fewer than max_additions additions.
    - Substantive: Changes in code files (not .txt or .md), not purely comments or imports
    - Additions must be contiguous after applying deletions

    Args:
        patch_content (str): Content of the git patch file
        max_additions (int): Maximum number of additions allowed (optional)

    Returns:
        dict: Maps file headers to lists of qualifying hunks
    """
    # First extract all hunks without filtering
    all_hunks = extract_hunks_from_patch(patch_content)
    
    # Now apply filters to the hunks
    filtered_result = {}
    
    for file_header, file_info in all_hunks.items():
        file_path = file_info["file_path"]
        is_new_file = file_info["is_new_file"]
        is_renamed_file = file_info["is_renamed_file"]
        qualifying_hunks = []
        
        # Skip new or renamed files
        if is_new_file or is_renamed_file:
            continue
            
        # Skip non-code files
        if not file_path or is_non_code_file(file_path):
            continue
            
        # Skip specific files (nlohmann/json.hpp)
        if 'single_include/nlohmann/json.hpp' in file_path:
            continue
        
        for hunk_text in file_info["hunks"]:
            # Convert hunk text to lines for filtering checks
            hunk_lines = hunk_text.splitlines()
            
            # Check if hunk qualifies
            additions = 0
            is_substantive = False

            for line in hunk_lines:
                if line.startswith('+') and not line.startswith('@@') and not line.startswith('+++'):
                    code_line = line[1:].strip()
                    if code_line and not is_comment_or_import(code_line):
                        additions += 1
                        is_substantive = True

            # Only check with GPT-4o-mini if the hunk passes initial filters
            if is_substantive and (max_additions is None or additions <= max_additions) and (not require_contiguous_additions or has_contiguous_additions(hunk_lines)):
                # Check with GPT-4o-mini if it contains actual code changes
                if llm_check and is_actual_code_change(hunk_text):
                    qualifying_hunks.append(hunk_text)
                elif not llm_check:
                    qualifying_hunks.append(hunk_text)
        
        # Add to results if there are qualifying hunks
        if qualifying_hunks:
            filtered_result[file_header] = qualifying_hunks
    
    return filtered_result

def is_non_code_file(file_path):
    """Check if a file is not a code file based on its extension."""
    non_code_extensions = ['.txt', '.md', '.jpg', '.png', '.pdf', '.doc', '.docx', '.adoc', '.json', '.yaml', '.yml', '.toml', '.lock', '.lockb', '.lock.json', '.lock.yaml', '.lock.yml', '.lock.toml', '.rst']
    return any(file_path.endswith(ext) for ext in non_code_extensions)

def is_comment_or_import(line):
    """Check if a line is a comment or import statement."""
    line = line.strip()

    # Empty lines
    if not line:
        return True

    # Comments in various languages
    if (line.startswith('#') or
        line.startswith('//') or
        line.startswith('/*') or
        line.startswith('*') or
        line.startswith('"""') or
        line.startswith("'''") or
        line.endswith('*/')):
        return True

    # Imports in various languages
    if (line.startswith('import ') or
        line.startswith('from ') or
        line.startswith('require ') or
        line.startswith('include ') or
        line.startswith('using ') or
        line.startswith('package ') or
        line.startswith('exports ') or
        line.startswith('uses ') or
        line.startswith('#include')):
        return True

    return False