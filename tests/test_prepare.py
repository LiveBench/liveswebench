import pytest
from unittest.mock import patch, MagicMock, call
from pathlib import Path

from liveswebench.harness.prepare import prepare_task
from liveswebench.util.tasks import TaskInstance, TaskType


@pytest.fixture
def mock_repo():
    """Create a mock repository with necessary methods"""
    repo = MagicMock()
    repo.name = "mock_repo"
    repo.repo_path = Path("/fake/repo/path")
    repo.clean = MagicMock()
    repo.git_checkout = MagicMock()
    repo.write_ignore = MagicMock()
    repo.apply_patch = MagicMock()
    repo.install_cmd = ["pip install -e ."]
    
    # Setup git_repo mock
    repo.git_repo = MagicMock()
    repo.git_repo.heads = []
    
    return repo


@pytest.fixture
def mock_task_instance():
    """Create a mock task instance for testing"""
    task = MagicMock(spec=TaskInstance)
    task.repo_name = "mock_repo"
    task.task_num = 123
    task.base_commit = "abc123"
    task.test_patch = "mock test patch content"
    task.task_data_path = Path("/fake/task/data/path/123")
    task.get_prompt = MagicMock(return_value="Test prompt for task 123")
    task.get_ground_truth_patch = MagicMock(return_value="mock ground truth patch")
    
    return task


@patch("liveswebench.harness.prepare.get_repo")
@patch("liveswebench.harness.prepare.get_partial_gold_patch")
@patch("liveswebench.harness.prepare.get_relevant_files_for_task")
@patch("liveswebench.harness.prepare.execute_commands")
class TestPrepareAgent:
    """Test prepare_task function for AGENT task type"""

    def test_prepare_agent_task(self, mock_execute, mock_get_relevant_files, 
                              mock_get_partial_gold, mock_get_repo, 
                              mock_repo, mock_task_instance):
        """Test basic agent task preparation"""
        mock_get_repo.return_value = mock_repo
        mock_get_partial_gold.return_value = "mock gold patch content"
        mock_get_relevant_files.return_value = ["file1.py", "file2.py"]
        
        # Call the function
        prepare_task(
            task=mock_task_instance,
            install=True,
            print_prompt=True,
            task_type=TaskType.AGENT
        )
        
        # Verify repository is cloned and configured correctly
        mock_repo.clean.assert_called_once()
        mock_repo.git_checkout.assert_called()
        mock_repo.write_ignore.assert_called_once()
        
        # Verify partial gold patch is applied correctly
        mock_get_partial_gold.assert_called_once_with(mock_task_instance, TaskType.AGENT)
        mock_repo.apply_patch.assert_called_once_with("mock gold patch content", '--ignore-whitespace')
        
        # Verify install command is used
        mock_execute.assert_called_once_with(
            mock_repo.install_cmd,
            cwd=str(mock_repo.repo_path),
            output_to_terminal=True,
            exit_on_fail=True,
        )


@patch("liveswebench.harness.prepare.get_repo")
@patch("liveswebench.harness.prepare.get_partial_gold_patch")
@patch("liveswebench.harness.prepare.get_relevant_files_for_task")
@patch("liveswebench.harness.prepare.execute_commands")
class TestPrepareEdit:
    """Test prepare_task function for EDIT task type"""

    def test_prepare_edit_task(self, mock_execute, mock_get_relevant_files, 
                             mock_get_partial_gold, mock_get_repo, 
                             mock_repo, mock_task_instance):
        """Test edit task preparation"""
        mock_get_repo.return_value = mock_repo
        mock_get_partial_gold.return_value = "mock partial gold patch content"
        mock_get_relevant_files.return_value = ["edit_file.py"]
        
        # Call the function
        prepare_task(
            task=mock_task_instance,
            install=True,
            print_prompt=True,
            task_type=TaskType.EDIT
        )
        
        # Verify repository is cloned and configured correctly
        mock_repo.clean.assert_called_once()
        mock_repo.git_checkout.assert_called()
        
        # Verify partial gold patch is applied correctly
        mock_get_partial_gold.assert_called_once_with(mock_task_instance, TaskType.EDIT)
        mock_repo.apply_patch.assert_called_once_with("mock partial gold patch content", '--ignore-whitespace')
        
        # Verify relevant files are retrieved
        mock_get_relevant_files.assert_called_with(mock_task_instance, TaskType.EDIT)
        
        # Verify install command is used
        mock_execute.assert_called_once()


@patch("liveswebench.harness.prepare.get_repo")
@patch("liveswebench.harness.prepare.get_partial_gold_patch")
@patch("liveswebench.harness.prepare.get_relevant_files_for_task")
@patch("liveswebench.harness.prepare.get_removal_patch_for_task")
@patch("liveswebench.harness.prepare.execute_commands")
class TestPrepareAutocomplete:
    """Test prepare_task function for AUTOCOMPLETE task type"""

    def test_prepare_autocomplete_task(self, mock_execute, mock_get_removal_patch,
                                     mock_get_relevant_files, mock_get_partial_gold, 
                                     mock_get_repo, mock_repo, mock_task_instance):
        """Test autocomplete task preparation"""
        mock_get_repo.return_value = mock_repo
        mock_get_partial_gold.return_value = "mock partial gold patch content"
        mock_get_removal_patch.return_value = "mock removal patch content"
        mock_get_relevant_files.return_value = ["autocomplete_file.py"]
        
        # Call the function
        prepare_task(
            task=mock_task_instance,
            install=True,
            print_prompt=True,
            task_type=TaskType.AUTOCOMPLETE
        )
        
        # Verify repository is cloned and configured correctly
        mock_repo.clean.assert_called_once()
        mock_repo.git_checkout.assert_called()
        
        # Verify partial gold patch is applied correctly
        mock_get_partial_gold.assert_called_once_with(mock_task_instance, TaskType.AUTOCOMPLETE)
        
        # Verify removal patch is applied correctly
        mock_get_removal_patch.assert_called_once_with(mock_task_instance, TaskType.AUTOCOMPLETE)
        
        # Repo.apply_patch should be called twice - once for partial gold patch, once for removal patch
        assert mock_repo.apply_patch.call_count == 2
        
        # Verify install command is used
        mock_execute.assert_called_once()


class TestPrepareTask:
    """Test prepare_task function"""
    
    @patch("pathlib.Path.resolve")
    @patch("builtins.open", new_callable=MagicMock)
    @patch("liveswebench.harness.prepare.get_relevant_files_for_task")
    @patch("liveswebench.harness.prepare.get_removal_patch_for_task")
    @patch("liveswebench.harness.prepare.execute_commands")
    @patch("liveswebench.harness.prepare.get_partial_gold_patch")
    @patch("liveswebench.harness.prepare.get_repo")
    def test_prepare_task_basic(self, mock_get_repo, mock_partial_gold, 
                             mock_execute_commands, mock_removal_patch, mock_relevant_files,
                             mock_open, mock_resolve, mock_repo, mock_task_instance):
        """Test basic preparation of a task"""
        mock_get_repo.return_value = mock_repo
        mock_partial_gold.return_value = "mock partial gold patch"
        mock_removal_patch.return_value = None
        mock_relevant_files.return_value = None
        mock_resolve.return_value = Path("/fake/repo/path/resolved")
        
        # Call the function
        prepare_task(
            task=mock_task_instance,
            install=False,
            print_prompt=False,
            task_type=TaskType.AGENT
        )
        
        # Verify repo was cleaned
        mock_repo.clean.assert_called_once()
        
        # Verify branch creation/checkout
        mock_repo.git_checkout.assert_has_calls([
            call(mock_task_instance.base_commit),
            call("-b", f"task_{mock_task_instance.task_num}")
        ])
        
        # Verify ignore file was written
        mock_repo.write_ignore.assert_called_once()
        
        # Verify partial gold patch was applied
        mock_partial_gold.assert_called_once_with(mock_task_instance, TaskType.AGENT)
        mock_repo.apply_patch.assert_called_once_with("mock partial gold patch", '--ignore-whitespace')
        
        # Verify install command was not executed
        mock_execute_commands.assert_not_called()
    
    @patch("pathlib.Path.resolve")
    @patch("builtins.open", new_callable=MagicMock)
    @patch("liveswebench.harness.prepare.get_relevant_files_for_task")
    @patch("liveswebench.harness.prepare.get_removal_patch_for_task")
    @patch("liveswebench.harness.prepare.execute_commands")
    @patch("liveswebench.harness.prepare.get_partial_gold_patch")
    @patch("liveswebench.harness.prepare.get_repo")
    def test_prepare_task_with_install(self, mock_get_repo, mock_partial_gold, 
                                    mock_execute_commands, mock_removal_patch, mock_relevant_files,
                                    mock_open, mock_resolve, mock_repo, mock_task_instance):
        """Test preparation with install flag"""
        mock_get_repo.return_value = mock_repo
        mock_partial_gold.return_value = "mock partial gold patch"
        mock_removal_patch.return_value = None
        mock_relevant_files.return_value = None
        
        # Call the function with install=True
        prepare_task(
            task=mock_task_instance,
            install=True,
            print_prompt=False,
            task_type=TaskType.AGENT
        )
        
        # Verify install command was executed
        mock_execute_commands.assert_called_once_with(
            ["pip install -e ."],
            cwd=str(mock_repo.repo_path),
            output_to_terminal=True,
            exit_on_fail=True
        )
    
    @patch("pathlib.Path.resolve")
    @patch("builtins.open", new_callable=MagicMock)
    @patch("liveswebench.harness.prepare.get_relevant_files_for_task")
    @patch("liveswebench.harness.prepare.get_removal_patch_for_task")
    @patch("liveswebench.harness.prepare.execute_commands")
    @patch("liveswebench.harness.prepare.get_partial_gold_patch")
    @patch("liveswebench.harness.prepare.get_repo")
    def test_prepare_task_print_prompt(self, mock_get_repo, mock_partial_gold, 
                                     mock_execute_commands, mock_removal_patch, mock_relevant_files,
                                     mock_open, mock_resolve, mock_repo, mock_task_instance):
        """Test preparation with print_prompt flag"""
        mock_get_repo.return_value = mock_repo
        mock_partial_gold.return_value = "mock partial gold patch"
        mock_removal_patch.return_value = None
        mock_relevant_files.return_value = None
        
        # Call the function with print_prompt=True
        prepare_task(
            task=mock_task_instance,
            install=False,
            print_prompt=True,
            task_type=TaskType.AGENT
        )
        
        # Verify prompt was retrieved
        mock_task_instance.get_prompt.assert_called_once_with(TaskType.AGENT)
    
    @patch("pathlib.Path.resolve")
    @patch("builtins.open", new_callable=MagicMock)
    @patch("liveswebench.harness.prepare.get_relevant_files_for_task")
    @patch("liveswebench.harness.prepare.get_removal_patch_for_task")
    @patch("liveswebench.harness.prepare.execute_commands")
    @patch("liveswebench.harness.prepare.get_partial_gold_patch")
    @patch("liveswebench.harness.prepare.get_repo")
    def test_prepare_task_edit_type(self, mock_get_repo, mock_partial_gold, 
                                  mock_execute_commands, mock_removal_patch, mock_relevant_files,
                                  mock_open, mock_resolve, mock_repo, mock_task_instance):
        """Test preparation for edit task type"""
        mock_get_repo.return_value = mock_repo
        mock_partial_gold.return_value = "mock partial gold patch"
        mock_removal_patch.return_value = None
        mock_relevant_files.return_value = ["path/to/edit_file.py"]
        
        # Mock the edit patch to include new file creation
        mock_task_instance.get_ground_truth_patch.return_value = "--- /dev/null\n+++ b/path/to/edit_file.py"
        
        # Call the function for edit task type
        prepare_task(
            task=mock_task_instance,
            install=False,
            print_prompt=False,
            task_type=TaskType.EDIT
        )
        
        # Verify relevant files were retrieved
        mock_relevant_files.assert_called_with(mock_task_instance, TaskType.EDIT)
        
        # Verify file was created
        mock_open.assert_called_once()
    
    @patch("pathlib.Path.resolve")
    @patch("builtins.open", new_callable=MagicMock)
    @patch("liveswebench.harness.prepare.get_relevant_files_for_task")
    @patch("liveswebench.harness.prepare.get_removal_patch_for_task")
    @patch("liveswebench.harness.prepare.execute_commands")
    @patch("liveswebench.harness.prepare.get_partial_gold_patch")
    @patch("liveswebench.harness.prepare.get_repo")
    def test_prepare_task_autocomplete_type(self, mock_get_repo, mock_partial_gold, 
                                         mock_execute_commands, mock_removal_patch, mock_relevant_files,
                                         mock_open, mock_resolve, mock_repo, mock_task_instance):
        """Test preparation for autocomplete task type"""
        mock_get_repo.return_value = mock_repo
        mock_partial_gold.return_value = "mock partial gold patch"
        mock_removal_patch.return_value = "mock removal patch"
        mock_relevant_files.return_value = ["path/to/autocomplete_file.py"]
        
        # Call the function for autocomplete task type
        prepare_task(
            task=mock_task_instance,
            install=False,
            print_prompt=False,
            task_type=TaskType.AUTOCOMPLETE
        )
        
        # Verify removal patch was retrieved and applied
        mock_removal_patch.assert_called_once_with(mock_task_instance, TaskType.AUTOCOMPLETE)
        mock_repo.apply_patch.assert_has_calls([
            call("mock partial gold patch", '--ignore-whitespace'),
            call("mock removal patch", '--ignore-whitespace')
        ])
    
    @patch("pathlib.Path.resolve")
    @patch("builtins.open", new_callable=MagicMock)
    @patch("liveswebench.harness.prepare.get_relevant_files_for_task")
    @patch("liveswebench.harness.prepare.get_removal_patch_for_task")
    @patch("liveswebench.harness.prepare.execute_commands")
    @patch("liveswebench.harness.prepare.get_partial_gold_patch")
    @patch("liveswebench.harness.prepare.get_repo")
    def test_prepare_task_existing_branch(self, mock_get_repo, mock_partial_gold, 
                                       mock_execute_commands, mock_removal_patch, mock_relevant_files,
                                       mock_open, mock_resolve, mock_repo, mock_task_instance):
        """Test preparation when branch already exists"""
        mock_get_repo.return_value = mock_repo
        mock_partial_gold.return_value = "mock partial gold patch"
        mock_removal_patch.return_value = None
        mock_relevant_files.return_value = None
        
        # Setup branch to already exist
        branch_name = f"task_{mock_task_instance.task_num}"
        mock_repo.git_repo.heads = [branch_name]
        
        # Call the function
        prepare_task(
            task=mock_task_instance,
            install=False,
            print_prompt=False,
            task_type=TaskType.AGENT
        )
        
        # Verify branch was checked out directly
        mock_repo.git_checkout.assert_called_once_with(branch_name)
        
    @patch("pathlib.Path.resolve")
    @patch("builtins.open", new_callable=MagicMock)
    @patch("liveswebench.harness.prepare.get_relevant_files_for_task")
    @patch("liveswebench.harness.prepare.get_removal_patch_for_task")
    @patch("liveswebench.harness.prepare.execute_commands")
    @patch("liveswebench.harness.prepare.get_partial_gold_patch")
    @patch("liveswebench.harness.prepare.get_repo")
    def test_prepare_task_tool_name(self, mock_get_repo, mock_partial_gold, 
                                  mock_execute_commands, mock_removal_patch, mock_relevant_files,
                                  mock_open, mock_resolve, mock_repo, mock_task_instance,
                                  monkeypatch):
        """Test preparation with tool_name specified"""
        mock_get_repo.return_value = mock_repo
        mock_partial_gold.return_value = "mock partial gold patch"
        mock_removal_patch.return_value = None
        mock_relevant_files.return_value = ["path/to/file.py"]
        
        # Mock subprocess.Popen for vscli commands
        mock_popen = MagicMock()
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = ("", "")
        mock_popen.return_value = mock_process
        
        # Patch subprocess.Popen
        monkeypatch.setattr("subprocess.Popen", mock_popen)
        
        # Call the function with tool_name
        prepare_task(
            task=mock_task_instance,
            install=False,
            print_prompt=False,
            task_type=TaskType.AGENT,
            tool_name="cursor"
        )
        
        # Verify vscli was called to open the tool
        # First call should open the repo
        assert mock_popen.call_count >= 1
        
        # Second call should open the relevant file
        assert mock_popen.call_count >= 2 