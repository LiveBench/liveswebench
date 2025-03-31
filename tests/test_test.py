import pytest
from unittest.mock import patch, MagicMock, mock_open, call, create_autospec
from pathlib import Path
import os

from liveswebench.harness.test import patch_and_test
# Do not import test_tasks directly to avoid name conflicts
# We'll reference it by full path: liveswebench.harness.test.test_tasks
from liveswebench.util.tasks import TaskInstance, TaskType


# Skip the test_tasks test 
@pytest.mark.skip(reason="Requires filesystem access - this is a wrapper for a real function, not a test")
def test_tasks(*args, **kwargs):
    # This is just a wrapper to catch the automatic pytest collection
    # We're not actually testing anything here
    pass


@pytest.fixture
def mock_repo():
    """Create a mock repository with necessary methods"""
    repo = MagicMock()
    repo.git_repo = MagicMock()
    repo.git_repo.active_branch = MagicMock()
    repo.git_repo.active_branch.name = "task_123"
    repo.repo_path = Path("/fake/repo/path")
    repo.name = "mock_repo"
    return repo


@pytest.fixture
def mock_task_instances():
    """Create mock task instances for testing"""
    task1 = MagicMock(spec=TaskInstance)
    task1.repo_name = "mock_repo"
    task1.task_num = 123
    task1.test_patch = "mock test patch content"
    task1.task_data_path = Path("/fake/task/data/path/123")
    
    task2 = MagicMock(spec=TaskInstance)
    task2.repo_name = "mock_repo"
    task2.task_num = 456
    task2.test_patch = "@gpu_test\nmock test patch content"
    task2.task_data_path = Path("/fake/task/data/path/456")
    
    return [task1, task2]


@patch("liveswebench.harness.test.get_repo")
@patch("liveswebench.harness.test.check_and_revert_patch")
@patch("liveswebench.harness.test.run_tests")
@patch("os.remove")
@patch("pathlib.Path.exists")
@patch("pathlib.Path.mkdir")
class TestPatchAndTest:
    """Test patch_and_test function"""

    def test_patch_and_test_basic(self, mock_mkdir, mock_exists, mock_remove, mock_run_tests, 
                               mock_revert_patch, mock_get_repo, mock_repo):
        """Test basic patch_and_test functionality"""
        mock_get_repo.return_value = mock_repo
        mock_exists.return_value = True  # All paths exist
        mock_mkdir.return_value = None  # Mock the mkdir call to avoid filesystem errors
        patch_file = Path("/fake/patches/cursor_patch.patch")
        
        # Call the function
        task = MagicMock(spec=TaskInstance)
        task.repo_name = "mock_repo"
        task.task_num = 123
        task.test_patch = "mock test patch content"
        task.task_data_path = Path("/fake/task/data/path/123")
        
        patch_and_test(
            task=task,
            tool_name="cursor",
            patch_file=patch_file,
            task_type=TaskType.AGENT
        )
        
        # Verify test patch was reverted and applied
        mock_revert_patch.assert_called_once_with("mock test patch content", repo=mock_repo)
        mock_repo.apply_patch.assert_called_with(Path("/fake/patches/cursor_patch.patch").resolve())
        
        # Verify run_tests was called
        mock_run_tests.assert_called_once()

    def test_patch_and_test_output_file(self, mock_mkdir, mock_exists, mock_remove, mock_run_tests, 
                                     mock_revert_patch, mock_get_repo, mock_repo):
        """Test patch_and_test with existing output file"""
        mock_get_repo.return_value = mock_repo
        mock_exists.return_value = True
        mock_mkdir.return_value = None  # Mock the mkdir call to avoid filesystem errors
        patch_file = Path("/fake/patches/cursor_patch.patch")
        
        # Call the function
        task = MagicMock(spec=TaskInstance)
        task.repo_name = "mock_repo"
        task.task_num = 123
        task.test_patch = "mock test patch content"
        task.task_data_path = Path("/fake/task/data/path/123")
        
        patch_and_test(
            task=task,
            tool_name="cursor",
            patch_file=patch_file,
            task_type=TaskType.AGENT
        )
        
        # Verify existing output file was removed
        mock_remove.assert_called_once()
        
        # Verify run_tests was called
        mock_run_tests.assert_called_once()


# Fixture for test_tasks
@pytest.fixture
def tasks():
    """Create a list of task instances for the test_tasks_function test"""
    task = MagicMock(spec=TaskInstance)
    task.repo_name = "mock_repo"
    task.task_num = 123
    task.test_patch = "mock test patch content"
    task.task_data_path = Path("/fake/task/data/path/123")
    return [task]


# Rename the standalone test to avoid name collision and mark it as skip to avoid filesystem access
@pytest.mark.skip(reason="Requires filesystem access")
def test_direct_test_tasks(tasks):
    """Test the actual test_tasks function directly (skipped to avoid filesystem access)"""
    from liveswebench.harness.test import test_tasks
    test_tasks(tasks=tasks)


# Create a specialized test for test_tasks that avoids filesystem access
def test_tasks_function(tasks):
    """Test the test_tasks function with mocked dependencies"""
    # Import the actual function
    from liveswebench.harness.test import test_tasks as original_test_tasks
    
    # Mock Path and os.listdir to avoid filesystem errors
    with patch("os.path.isdir", return_value=True):  # Mock isdir to return True
        with patch("liveswebench.util.tasks.os.listdir") as mock_listdir:
            with patch("liveswebench.util.tasks.glob.glob") as mock_glob:
                with patch("liveswebench.util.tasks.find_task_logs", return_value=[]):  # Mock find_task_logs
                    with patch("liveswebench.harness.test.prepare_task") as mock_prepare:
                        with patch("liveswebench.harness.test.patch_and_test") as mock_patch_and_test:
                            # Setup os.listdir to return a list of directories in the task data path
                            mock_listdir.return_value = ["cursor"]
                            
                            # Setup mock_glob to return a list with one patch matching the specific glob pattern
                            # This needs to match the pattern in find_task_patches: f"*_{task_type}_patch_*.patch"
                            mock_glob.return_value = ["/fake/task/data/path/123/cursor/cursor_agent_patch_20240401.patch"]
                            
                            # Call the function
                            original_test_tasks(
                                tasks=tasks,
                                tool_name="cursor",
                                skip_gpu_tests=False,
                                retest=False,
                                task_type=TaskType.AGENT,  # Use specific task type to avoid needing all types
                                test_all_patches=False
                            )
                            
                            # Verify prepare_task was called
                            mock_prepare.assert_called_once()
                            
                            # Verify patch_and_test was called
                            mock_patch_and_test.assert_called_once()


class TestTestTasksClass:
    """Test test_tasks function with different scenarios"""
    
    @patch("liveswebench.util.tasks.os.listdir")
    def test_test_tasks_agent(self, mock_listdir, mock_task_instances):
        """Test test_tasks for agent task type"""
        # Import inside the function to avoid name conflict
        from liveswebench.harness.test import test_tasks as harness_test_tasks
        
        # Mock the dependencies but keep pathlib.Path as is
        with patch("os.path.isdir", return_value=True):  # Mock isdir to return True
            with patch("liveswebench.util.tasks.glob.glob") as mock_glob:
                with patch("liveswebench.util.tasks.find_task_logs", return_value=[]):  # Mock find_task_logs
                    with patch("liveswebench.harness.test.prepare_task") as mock_prepare:
                        with patch("liveswebench.harness.test.patch_and_test") as mock_patch_and_test:
                            # Setup mocks
                            mock_listdir.return_value = ["cursor"]
                            # Need to match the pattern in find_task_patches: f"*_{task_type}_patch_*.patch"
                            mock_glob.return_value = [
                                "/fake/task/data/path/123/cursor/cursor_agent_patch_20240401.patch"
                            ]
                            
                            # Call the function
                            harness_test_tasks(
                                tasks=mock_task_instances[:1],  # Use only the first task
                                tool_name="cursor",
                                task_type=TaskType.AGENT,
                                retest=False,
                                test_all_patches=False
                            )
                            
                            # Verify prepare_task was called correctly
                            mock_prepare.assert_called_once_with(
                                task=mock_task_instances[0],
                                install=True,
                                task_type=TaskType.AGENT,
                                test=True
                            )
                            
                            # Verify patch_and_test was called correctly with the right tool_name
                            mock_patch_and_test.assert_called_once()
                            call_args = mock_patch_and_test.call_args[1]
                            assert call_args["tool_name"] == "cursor"
                            assert call_args["task_type"] == TaskType.AGENT
    
    @patch("liveswebench.util.tasks.os.listdir")
    def test_test_tasks_edit(self, mock_listdir, mock_task_instances):
        """Test test_tasks for edit task type"""
        # Import inside the function to avoid name conflict
        from liveswebench.harness.test import test_tasks as harness_test_tasks
        
        # Mock the dependencies but keep pathlib.Path as is
        with patch("os.path.isdir", return_value=True):  # Mock isdir to return True
            with patch("liveswebench.util.tasks.glob.glob") as mock_glob:
                with patch("liveswebench.util.tasks.find_task_logs", return_value=[]):  # Mock find_task_logs
                    with patch("liveswebench.harness.test.prepare_task") as mock_prepare:
                        with patch("liveswebench.harness.test.patch_and_test") as mock_patch_and_test:
                            # Setup mocks
                            mock_listdir.return_value = ["cursor"]
                            # Need to match the pattern in find_task_patches: f"*_{task_type}_patch_*.patch"
                            mock_glob.return_value = [
                                "/fake/task/data/path/123/cursor/cursor_edit_patch_20240401.patch"
                            ]
                            
                            # Call the function
                            harness_test_tasks(
                                tasks=mock_task_instances[:1],  # Use only the first task
                                tool_name="cursor",
                                task_type=TaskType.EDIT,
                                retest=True,
                                test_all_patches=False
                            )
                            
                            # Verify prepare_task was called correctly
                            mock_prepare.assert_called_once_with(
                                task=mock_task_instances[0],
                                install=True,
                                task_type=TaskType.EDIT,
                                test=True
                            )
                            
                            # Verify patch_and_test was called with correct tool_name
                            mock_patch_and_test.assert_called_once()
                            call_args = mock_patch_and_test.call_args[1]
                            assert call_args["tool_name"] == "cursor"
                            assert call_args["task_type"] == TaskType.EDIT
    
    @patch("liveswebench.util.tasks.os.listdir")
    def test_test_tasks_autocomplete(self, mock_listdir, mock_task_instances):
        """Test test_tasks for autocomplete task type"""
        # Import inside the function to avoid name conflict
        from liveswebench.harness.test import test_tasks as harness_test_tasks
        
        # Mock the dependencies but keep pathlib.Path as is
        with patch("os.path.isdir", return_value=True):  # Mock isdir to return True
            with patch("liveswebench.util.tasks.glob.glob") as mock_glob:
                with patch("liveswebench.util.tasks.find_task_logs", return_value=[]):  # Mock find_task_logs
                    with patch("liveswebench.harness.test.prepare_task") as mock_prepare:
                        with patch("liveswebench.harness.test.patch_and_test") as mock_patch_and_test:
                            # Setup mocks
                            mock_listdir.return_value = ["cursor"]
                            # Need to match the pattern in find_task_patches: f"*_{task_type}_patch_*.patch"
                            mock_glob.return_value = [
                                "/fake/task/data/path/123/cursor/cursor_autocomplete_patch_20240401.patch"
                            ]
                            
                            # Call the function
                            harness_test_tasks(
                                tasks=mock_task_instances[:1],  # Use only the first task
                                tool_name="cursor",
                                task_type=TaskType.AUTOCOMPLETE,
                                retest=False,
                                test_all_patches=True  # Test all patches
                            )
                            
                            # Verify prepare_task was called correctly
                            mock_prepare.assert_called_once_with(
                                task=mock_task_instances[0],
                                install=True,
                                task_type=TaskType.AUTOCOMPLETE,
                                test=True
                            )
                            
                            # Verify patch_and_test was called with correct tool_name
                            mock_patch_and_test.assert_called_once()
                            call_args = mock_patch_and_test.call_args[1]
                            assert call_args["tool_name"] == "cursor"
                            assert call_args["task_type"] == TaskType.AUTOCOMPLETE
    
    @patch("liveswebench.util.tasks.os.listdir")
    def test_test_tasks_skip_gpu(self, mock_listdir, mock_task_instances):
        """Test test_tasks skips GPU tasks when skip_gpu_tests=True"""
        # Import inside the function to avoid name conflict
        from liveswebench.harness.test import test_tasks as harness_test_tasks
        
        # Mock the dependencies but keep pathlib.Path as is
        with patch("os.path.isdir", return_value=True):  # Mock isdir to return True
            with patch("liveswebench.util.tasks.glob.glob") as mock_glob:
                with patch("liveswebench.util.tasks.find_task_logs", return_value=[]):  # Mock find_task_logs
                    with patch("liveswebench.harness.test.prepare_task") as mock_prepare:
                        with patch("liveswebench.harness.test.patch_and_test") as mock_patch_and_test:
                            # Setup mocks to return patches for task 1 only (not for task 2 with GPU)
                            mock_listdir.return_value = ["cursor"]
                            # Return real patches for each task type for the non-GPU task
                            # Need to match the pattern in find_task_patches: f"*_{task_type}_patch_*.patch"
                            mock_glob.side_effect = [
                                # Agent patches for task 1
                                ["/fake/task/data/path/123/cursor/cursor_agent_patch_20240401.patch"],
                                # Edit patches for task 1
                                ["/fake/task/data/path/123/cursor/cursor_edit_patch_20240401.patch"],
                                # Autocomplete patches for task 1
                                ["/fake/task/data/path/123/cursor/cursor_autocomplete_patch_20240401.patch"]
                            ]
                            
                            # Call the function with skip_gpu_tests=True and all task types
                            harness_test_tasks(
                                tasks=mock_task_instances,  # Include both tasks (second one has GPU marker)
                                tool_name="cursor",
                                task_type="all",
                                skip_gpu_tests=True,
                                retest=False,
                                test_all_patches=False
                            )
                            
                            # Verify only the first task was prepared and tested (not the GPU task)
                            assert mock_prepare.call_count == 3  # Called for each task type for task 1
                            assert mock_patch_and_test.call_count == 3  # Called for each task type for task 1 