from dataclasses import dataclass, field
import os
from pathlib import Path
import shutil
import git
from git import Repo as GitRepo

from liveswebench.util.util import BASE_DATA_PATH, BASE_REPO_PATH, BASE_TASK_PATH, execute_commands

@dataclass
class Repo:
    name: str
    github_path: str
    test_cmd: dict[str, list[str] | dict[str, list[str] | dict[str, list[str]]]]
    install_cmd: list[str] | None = None
    pre_test_cmd: list[str] | None = None
    test_regex_cmd: dict[str, list[str] | dict[str, dict[str, str | list[str]]]] | None = None
    clean_cmd: dict[str, str] | None = None
    ignore: str | None = None
    build_path: list[str] | None = None
    repo_path: Path = field(init=False)
    task_path: Path = field(init=False)
    task_data_path: Path = field(init=False)
    git_repo: GitRepo = field(init=False)

    def clone(self):
        if not self.repo_path.exists():
            url = f"https://github.com/{self.github_path}"
            print(f"Cloning repository from {url}")
            self.git_repo = GitRepo.clone_from(url, self.repo_path)
            # some agents automatically ignore all files in .gitignore, even in a nested repo
            # so we use the .git/info/exclude to ignore the repos instead of .gitignore
            with open(Path(__file__).parent.parent / '../.git/info/exclude', 'a') as f:
                f.write(f"liveswebench/repos/{self.name}\n")
        else:
            self.git_repo = GitRepo(self.repo_path)

    def __post_init__(self):
        self.task_path = BASE_TASK_PATH / self.name
        self.task_data_path = BASE_DATA_PATH / self.name
        self.repo_path = BASE_REPO_PATH / self.name
        

    def write_ignore(self):
        self.clone()
        if self.ignore:
            print(f"Writing ignore files for repository {self.name}")
            with open(self.repo_path / '.gitignore', 'a') as f:
                f.write(self.ignore)
            with open(self.repo_path / '.cursorignore', 'w') as f:
                f.write(self.ignore)
            with open(self.repo_path / '.codeiumignore', 'w') as f:
                f.write(self.ignore)
            with open(self.repo_path / '.aiderignore', 'w') as f:
                f.write(self.ignore)
                

    def clean_ignore(self):
        self.clone()
        if self.ignore:
            if (self.repo_path / '.cursorignore').exists():
                os.remove(self.repo_path / '.cursorignore')
            if (self.repo_path / '.codeiumignore').exists():
                os.remove(self.repo_path / '.codeiumignore')
            if (self.repo_path / '.aiderignore').exists():
                os.remove(self.repo_path / '.aiderignore')
            if (self.repo_path / '.vscode').exists():
                shutil.rmtree(self.repo_path / '.vscode')
            self.git_checkout("HEAD", "--", ".gitignore")
                

    def clean(self, write_ignore: bool = True):
        """Clean up any local changes in the repository"""
        self.clone()
        try:
            # Discard all local changes
            self.git_repo.git.reset("--hard")
            # Clean untracked files and directories
            self.git_repo.git.clean("-fdx")
            if write_ignore:
                self.write_ignore()
        except git.GitCommandError as e:
            raise RuntimeError(f"Error cleaning repository state: {e}")

        if self.build_path is not None:
            if isinstance(self.build_path, list):
                build_path = [self.repo_path / path for path in self.build_path]
            else:
                build_path = [self.repo_path / self.build_path]
            for path in build_path:
                if path.exists():
                    print(f"Removing build path {path}")
                    shutil.rmtree(path)

        if self.clean_cmd is not None:
            for check in self.clean_cmd:
                if os.path.exists(self.repo_path / check):
                    print(f"Executing clean commands for repository {self.name}")
                    execute_commands(
                        self.clean_cmd[check],
                        cwd=str(self.repo_path),
                        output_to_terminal=False,
                        exit_on_fail=True,
                    )
        
    def git_checkout(self, *args: str, **kwargs: str) -> None:
        self.clone()
        self.git_repo.git.checkout(*args, **kwargs)

    def git_apply(self, *args: str, **kwargs: str) -> None:
        self.clone()
        self.git_repo.git.apply(*args, **kwargs)

    def git_add(self, *args: str, **kwargs: str) -> None:
        self.clone()
        self.git_repo.git.add(*args, **kwargs)

    def git_diff(self, *args: str, **kwargs: str) -> str:
        self.clone()
        return self.git_repo.git.diff(*args, **kwargs)
    
    def git_status(self) -> str:
        self.clone()
        return self.git_repo.git.status()
    
    def apply_patch(self, patch_source: str | Path, *args: str, **kwargs: str) -> None:
        """
        Apply a patch from either a file path or a string.
        
        Args:
            patch_source: Either a file path (string or Path) or patch contents as a string
            *args, **kwargs: Additional arguments to pass to git_apply
            
        Returns:
            The output of the git_apply command
        """
        self.clone()
        temp_file_path = None
        try:
            # If patch_source is a string and not a file path, write it to a temporary file
            if isinstance(patch_source, str) and not os.path.exists(patch_source):
                import time
                timestamp = int(time.time() * 1000)
                temp_file_path = self.repo_path / f"tmp_patch_{timestamp}.patch"
                
                with open(temp_file_path, 'w', encoding='utf-8') as f:
                    f.write(patch_source)
                    if not patch_source.endswith('\n'):
                        f.write('\n')
                
                self.git_apply(str(temp_file_path), *args, **kwargs)
            else:
                # Otherwise, assume it's a file path
                self.git_apply(patch_source, *args, **kwargs)
        finally:
            # Clean up the temporary file if one was created
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

Json = Repo(
    name='json',
    github_path='nlohmann/json',
    test_cmd={
        'default': ['rm -rf build', 'mkdir build', 'cd build', 'cmake .. -DJSON_BuildTests=ON', 'cmake --build .', 'ctest --output-on-failure']
    },
    build_path=['build'],
    ignore='single_include/'
)

TorchTune = Repo(
    name='torchtune',
    github_path='pytorch/torchtune',
    install_cmd=['python3 -m venv .venv', 'source .venv/bin/activate', 'pip install torchao torch torchvision lm-eval==0.4.5', "pip install -e \".[dev]\""],
    test_cmd={
        'default': ['source .venv/bin/activate', 'pytest tests'],
        'mark.integration_test': ['source .venv/bin/activate', 'pytest tests -m integration_test'],
        'mark.slow_integration_test': ['source .venv/bin/activate', 'pytest tests -m slow_integration_test']
    },
    build_path=[".venv"]
)

FreeCodeCamp = Repo(
    name='freeCodeCamp',
    github_path='freeCodeCamp/freeCodeCamp',
    install_cmd=['cp sample.env .env', 'nvm install 20', 'pnpm install', 'pnpm run create:shared', 'pnpm run seed', 'pnpm run seed:certified-user'],
    pre_test_cmd=['pnpm run build:curriculum', 'pnpm run build-workers','cd client', 'pnpm run build:scripts'],
    test_cmd={
        'diff --git a/curriculum': ['nvm install 20', "NODE_OPTIONS='--max-old-space-size=8192' pnpm run test-curriculum-full-output"],
        'diff --git a/api/': ['nvm install 20', "NODE_OPTIONS='--max-old-space-size=8192' pnpm run test:api"],
        'diff --git a/client': ['nvm install 20', "NODE_OPTIONS='--max-old-space-size=8192' pnpm run test-client"],
        'diff --git a/tools': ['nvm install 20', "NODE_OPTIONS='--max-old-space-size=8192' pnpm run test-tools"]
    },
    test_regex_cmd={
        'diff --git a\/e2e\/(?:mobile\/)?([^\/]+\\.spec\\.ts)': {
            'server': {
                'command': "NODE_OPTIONS='--max-old-space-size=8192' pnpm run develop",
                'ready_string': 'You can now view @freecodecamp/client in the browser',
                'port': 8000
            },
            'test': ['nvm install 20', 'pnpm run playwright:install-build-tools', "PW_TEST_HTML_REPORT_OPEN='never' PLAYWRIGHT_HTML_OPEN='never' npx playwright test {groups}"]
        }
    },
    clean_cmd={'node_modules': 'pnpm run clean'},
    ignore='.devcontainer/*\n.git/*\n.github/*\n.husky/*\ncurriculum/challenges/*\n!curriculum/challenges/english/*\ncurriculum/dictionaries/*\ncurriculum/schema/*\ndocker/*'
)

Wagtail = Repo(
    name='wagtail',
    github_path='wagtail/wagtail',
    install_cmd=['nvm install 22', 'python -m venv .venv', 'source .venv/bin/activate', 'pip install -e .\"[testing,docs]\" -U', 'npm ci'],
    pre_test_cmd=['npm run build'],
    test_cmd={
        'diff --git a/wagtail': ['source .venv/bin/activate', 'python runtests.py wagtail'],
        'diff --git a/client/src': ['source .venv/bin/activate', 'nvm install 22', 'npm run test:unit']
    },
    build_path=[".venv", "client/build", "node_modules", "wagtail.egg-info"]
)

JUnit5 = Repo(
    name='junit5',
    github_path='junit-team/junit5',
    test_cmd={
        'default': ['export JAVA_HOME=`/usr/libexec/java_home -v 21`', './gradlew clean test']
    }
)

def get_repo(name: str):
    if name == 'json':
        return Json
    elif name == 'torchtune':
        return TorchTune
    elif name == 'freeCodeCamp':
        return FreeCodeCamp
    elif name == 'wagtail':
        return Wagtail
    elif name == 'junit5':
        return JUnit5
    else:
        raise ValueError(f"Repository {name} not found")

def get_all_repos():
    return [Json, TorchTune, FreeCodeCamp, Wagtail, JUnit5]