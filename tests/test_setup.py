import pytest
from setuptools import find_packages
import os
import re
from pathlib import Path

# We'll read setup.py as text instead of importing it to avoid execution issues
setup_path = Path(__file__).parent.parent / 'setup.py'
setup_content = setup_path.read_text()

# Extract key information using regular expressions
def extract_value(pattern, text):
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

name = extract_value(r"name=['\"]([^'\"]+)['\"]", setup_content)
author = extract_value(r"author=['\"]([^'\"]+)['\"]", setup_content)
description = extract_value(r"description=['\"]([^'\"]+)['\"]", setup_content)
long_description_content_type = extract_value(r"long_description_content_type=['\"]([^'\"]+)['\"]", setup_content)
python_requires = extract_value(r"python_requires=['\"]([^'\"]+)['\"]", setup_content)

# Extract install_requires list
install_requires_match = re.search(r"install_requires=\[(.*?)\]", setup_content, re.DOTALL)
install_requires = []
if install_requires_match:
    packages_text = install_requires_match.group(1)
    # Extract quoted package names
    install_requires = re.findall(r"['\"]([^'\"]+)['\"]", packages_text)

# Extract extras_require dict
extras_require_match = re.search(r"extras_require=\{(.*?)\}", setup_content, re.DOTALL)
extras_require = {}
if extras_require_match:
    extras_text = extras_require_match.group(1)
    # Extract each group's name and packages
    groups = re.findall(r"['\"]([^'\"]+)['\"]:\s*\[(.*?)\]", extras_text, re.DOTALL)
    for group_name, packages_text in groups:
        packages = re.findall(r"['\"]([^'\"]+)['\"]", packages_text)
        extras_require[group_name] = packages

def test_package_name():
    assert name is not None
    assert name == "liveswebench"

def test_package_metadata():
    assert author is not None
    assert description is not None
    assert long_description_content_type is not None
    assert author == "Abacus.AI"
    assert description is not None and "LiveSWEBench" in description
    assert long_description_content_type == "text/markdown"

def test_python_version():
    assert python_requires is not None
    assert python_requires >= "3.9"

def test_required_dependencies():
    required_packages = [
        'GitPython',
        'python-dotenv',
        'requests', 
        'ghapi',
        'fastcore',
        'beautifulsoup4',
        'unidiff',
        'tqdm',
        'rich',
        'datasets',
        'chardet',
        'openai',
    ]
    for package in required_packages:
        assert package in install_requires

def test_extras_require():
    assert 'base' in extras_require
    assert 'GitPython' in extras_require['base']
    
    # Now also test for dev dependencies including pytest
    assert 'dev' in extras_require
    assert any('pytest' in pkg for pkg in extras_require['dev'])

def test_package_discovery():
    packages = find_packages()
    assert 'liveswebench' in packages
