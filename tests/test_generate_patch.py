import pytest
from unittest.mock import patch, MagicMock, mock_open, call
from pathlib import Path

from liveswebench.harness.generate_patch import generate_patch
from liveswebench.util.tasks import TaskInstance, TaskType


@pytest.fixture
def mock_repo():
    """Create a mock repository with necessary methods"""
    repo = MagicMock()
    repo.git_add = MagicMock()
    repo.git_diff = MagicMock(return_value="mock diff content")
    repo.clean_ignore = MagicMock()
    return repo


@pytest.fixture
def mock_task_instance():
    """Create a mock task instance"""
    task = MagicMock(spec=TaskInstance)
    task.repo_name = "mock_repo"
    task.task_num = 123
    task.task_data_path = Path("/fake/path/task/data")
    return task


@patch("liveswebench.harness.generate_patch.get_repo")
@patch("liveswebench.harness.generate_patch.get_patch_path")
@patch("liveswebench.harness.generate_patch.get_partial_gold_patch")
@patch("liveswebench.harness.generate_patch.check_and_revert_patch")
@patch("builtins.open", new_callable=mock_open)
@patch("pathlib.Path.exists")
@patch("builtins.input")
@patch("pathlib.Path.mkdir")
class TestGeneratePatchAgent:
    """Test generate_patch function for AGENT task type"""

    def test_generate_patch_agent(self, mock_mkdir, mock_input, mock_exists, mock_file,
                                mock_revert_patch, mock_get_partial_gold, 
                                mock_get_patch_path, mock_get_repo,
                                mock_repo, mock_task_instance):
        """Test generate_patch for agent task type"""
        mock_get_repo.return_value = mock_repo
        mock_get_patch_path.return_value = Path("/fake/path/agent_patch.patch")
        mock_get_partial_gold.return_value = "mock gold patch content"
        mock_exists.return_value = False  # Patch file doesn't exist
        mock_mkdir.return_value = None  # Mock the mkdir call to avoid filesystem errors
        
        # Call the function
        generate_patch(task=mock_task_instance, tool_name="cursor", task_type=TaskType.AGENT)
        
        # Verify repo was cleaned
        mock_repo.clean_ignore.assert_called_once()
        
        # For agent tasks, partial gold patch is reverted even though it's the same as gold patch
        mock_get_partial_gold.assert_called_once_with(mock_task_instance, TaskType.AGENT)
        mock_revert_patch.assert_called_once_with("mock gold patch content", repo=mock_repo)
        
        # Verify git commands were called correctly
        mock_repo.git_add.assert_called_once_with(".")
        mock_repo.git_diff.assert_called_once_with("HEAD")
        
        # Verify patch file was written
        mock_file.assert_called_once_with(Path("/fake/path/agent_patch.patch"), "w", encoding="utf-8")
        mock_file().write.assert_called_once_with("mock diff content")


@patch("liveswebench.harness.generate_patch.get_repo")
@patch("liveswebench.harness.generate_patch.get_patch_path")
@patch("liveswebench.harness.generate_patch.get_partial_gold_patch")
@patch("liveswebench.harness.generate_patch.check_and_revert_patch")
@patch("builtins.open", new_callable=mock_open)
@patch("pathlib.Path.exists")
@patch("builtins.input")
@patch("pathlib.Path.mkdir")
class TestGeneratePatchEdit:
    """Test generate_patch function for EDIT task type"""

    def test_generate_patch_edit(self, mock_mkdir, mock_input, mock_exists, mock_file,
                               mock_revert_patch, mock_get_partial_gold, 
                               mock_get_patch_path, mock_get_repo,
                               mock_repo, mock_task_instance):
        """Test generate_patch for edit task type"""
        mock_get_repo.return_value = mock_repo
        mock_get_patch_path.return_value = Path("/fake/path/edit_patch.patch")
        mock_get_partial_gold.return_value = "mock partial gold patch content"
        mock_exists.return_value = False  # Patch file doesn't exist
        mock_mkdir.return_value = None  # Mock the mkdir call to avoid filesystem errors
        
        # Call the function
        generate_patch(task=mock_task_instance, tool_name="cursor", task_type=TaskType.EDIT)
        
        # Verify repo was cleaned
        mock_repo.clean_ignore.assert_called_once()
        
        # Verify partial gold patch was reverted
        mock_get_partial_gold.assert_called_once_with(mock_task_instance, TaskType.EDIT)
        mock_revert_patch.assert_called_once_with("mock partial gold patch content", repo=mock_repo)
        
        # Verify git commands were called correctly
        mock_repo.git_add.assert_called_once_with(".")
        mock_repo.git_diff.assert_called_once_with("HEAD")
        
        # Verify patch file was written
        mock_file.assert_called_once_with(Path("/fake/path/edit_patch.patch"), "w", encoding="utf-8")
        mock_file().write.assert_called_once_with("mock diff content")


@patch("liveswebench.harness.generate_patch.get_repo")
@patch("liveswebench.harness.generate_patch.get_patch_path")
@patch("liveswebench.harness.generate_patch.get_partial_gold_patch")
@patch("liveswebench.harness.generate_patch.check_and_revert_patch")
@patch("builtins.open", new_callable=mock_open)
@patch("pathlib.Path.exists")
@patch("builtins.input")
@patch("pathlib.Path.mkdir")
class TestGeneratePatchAutocomplete:
    """Test generate_patch function for AUTOCOMPLETE task type"""

    def test_generate_patch_autocomplete(self, mock_mkdir, mock_input, mock_exists, mock_file,
                                       mock_revert_patch, mock_get_partial_gold, 
                                       mock_get_patch_path, mock_get_repo,
                                       mock_repo, mock_task_instance):
        """Test generate_patch for autocomplete task type"""
        mock_get_repo.return_value = mock_repo
        mock_get_patch_path.return_value = Path("/fake/path/autocomplete_patch.patch")
        mock_get_partial_gold.return_value = "mock partial gold patch content"
        mock_exists.return_value = False  # Patch file doesn't exist
        mock_mkdir.return_value = None  # Mock the mkdir call to avoid filesystem errors
        
        # Call the function
        generate_patch(task=mock_task_instance, tool_name="cursor", task_type=TaskType.AUTOCOMPLETE)
        
        # Verify repo was cleaned
        mock_repo.clean_ignore.assert_called_once()
        
        # Verify partial gold patch was reverted
        mock_get_partial_gold.assert_called_once_with(mock_task_instance, TaskType.AUTOCOMPLETE)
        mock_revert_patch.assert_called_once_with("mock partial gold patch content", repo=mock_repo)
        
        # Verify git commands were called correctly
        mock_repo.git_add.assert_called_once_with(".")
        mock_repo.git_diff.assert_called_once_with("HEAD")
        
        # Verify patch file was written
        mock_file.assert_called_once_with(Path("/fake/path/autocomplete_patch.patch"), "w", encoding="utf-8")
        mock_file().write.assert_called_once_with("mock diff content")

    def test_generate_patch_overwrite_existing(self, mock_mkdir, mock_input, mock_exists, mock_file,
                                             mock_revert_patch, mock_get_partial_gold, 
                                             mock_get_patch_path, mock_get_repo,
                                             mock_repo, mock_task_instance):
        """Test generate_patch when patch file already exists and user confirms overwrite"""
        mock_get_repo.return_value = mock_repo
        mock_get_patch_path.return_value = Path("/fake/path/autocomplete_patch.patch")
        mock_get_partial_gold.return_value = "mock partial gold patch content"
        mock_exists.return_value = True  # Patch file exists
        mock_input.return_value = 'y'  # User confirms overwrite
        mock_mkdir.return_value = None  # Mock the mkdir call to avoid filesystem errors
        
        # Call the function
        generate_patch(task=mock_task_instance, tool_name="cursor", task_type=TaskType.AUTOCOMPLETE)
        
        # Verify user was prompted and overwrite proceeded
        mock_input.assert_called_once()
        mock_file.assert_called_once_with(Path("/fake/path/autocomplete_patch.patch"), "w", encoding="utf-8")
        mock_file().write.assert_called_once_with("mock diff content") 