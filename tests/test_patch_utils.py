import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from liveswebench.util.tasks import TaskInstance, TaskType, get_partial_gold_patch
from liveswebench.harness.util import construct_partial_patch


@pytest.fixture
def mock_task_instance():
    """Create a mock task instance with realistic patch content"""
    task = MagicMock(spec=TaskInstance)
    task.repo_name = "mock_repo"
    task.task_num = 123
    
    # Sample patch that modifies two files
    task.gold_patch = """diff --git a/file1.py b/file1.py
index abc123..def456 100644
--- a/file1.py
+++ b/file1.py
@@ -10,7 +10,7 @@
 def existing_function():
     return True
 
-def needs_fix():
+def needs_fix_fixed():
     return "fixed"
 
 # Other code
diff --git a/file2.py b/file2.py
index 123abc..456def 100644
--- a/file2.py
+++ b/file2.py
@@ -5,6 +5,9 @@
 
 # Existing code
 
+def new_function():
+    return "new feature"
+
 class TestClass:
     def test_method(self):
         pass"""
    
    # Edit patch modifies only file1.py
    task.edit_patch = """diff --git a/file1.py b/file1.py
index abc123..def456 100644
--- a/file1.py
+++ b/file1.py
@@ -10,7 +10,7 @@
 def existing_function():
     return True
 
-def needs_fix():
+def needs_fix_fixed():
     return "fixed"
 
 # Other code"""
    
    # Autocomplete patch adds a function in file2.py
    task.autocomplete_patch = """diff --git a/file2.py b/file2.py
index 123abc..456def 100644
--- a/file2.py
+++ b/file2.py
@@ -5,6 +5,9 @@
 
 # Existing code
 
+def new_function():
+    return "new feature"
+
 class TestClass:
     def test_method(self):
         pass"""
    
    # Set up get_ground_truth_patch to return the appropriate patch
    task.get_ground_truth_patch = MagicMock()
    task.get_ground_truth_patch.side_effect = lambda task_type: {
        TaskType.AGENT: task.gold_patch,
        TaskType.EDIT: task.edit_patch,
        TaskType.AUTOCOMPLETE: task.autocomplete_patch
    }[task_type]
    
    return task


class TestPartialGoldPatch:
    """Test the generation of partial gold patches for different task types"""

    def test_partial_gold_patch_agent(self, mock_task_instance):
        """For AGENT tasks, the partial gold patch should be the same as the gold patch"""
        partial_gold = get_partial_gold_patch(mock_task_instance, TaskType.AGENT)
        assert partial_gold == mock_task_instance.gold_patch

    def test_partial_gold_patch_edit(self, mock_task_instance):
        """For EDIT tasks, the partial gold patch should be the gold patch with edit changes removed"""
        partial_gold = get_partial_gold_patch(mock_task_instance, TaskType.EDIT)
        
        # The partial gold patch should contain file2.py changes but not file1.py changes
        assert partial_gold is not None
        assert "file2.py" in partial_gold
        assert "new_function" in partial_gold
        assert "def needs_fix_fixed" not in partial_gold
        
        # Manual verification of the expected result
        expected_partial_gold = """diff --git a/file2.py b/file2.py
index 123abc..456def 100644
--- a/file2.py
+++ b/file2.py
@@ -5,6 +5,9 @@
 
 # Existing code
 
+def new_function():
+    return "new feature"
+
 class TestClass:
     def test_method(self):
         pass"""
        
        # Normalize whitespace for comparison
        assert partial_gold.strip() == expected_partial_gold.strip()

    def test_partial_gold_patch_autocomplete(self, mock_task_instance):
        """For AUTOCOMPLETE tasks, the partial gold patch should be the gold patch with autocomplete changes removed"""
        partial_gold = get_partial_gold_patch(mock_task_instance, TaskType.AUTOCOMPLETE)
        
        # The partial gold patch should contain file1.py changes but not file2.py changes
        assert partial_gold is not None
        assert "file1.py" in partial_gold
        assert "def needs_fix_fixed" in partial_gold
        assert "new_function" not in partial_gold
        
        # Manual verification of the expected result
        expected_partial_gold = """diff --git a/file1.py b/file1.py
index abc123..def456 100644
--- a/file1.py
+++ b/file1.py
@@ -10,7 +10,7 @@
 def existing_function():
     return True
 
-def needs_fix():
+def needs_fix_fixed():
     return "fixed"
 
 # Other code"""
        
        # Normalize whitespace for comparison
        assert partial_gold.strip() == expected_partial_gold.strip()


class TestConstructPartialPatch:
    """Test the construct_partial_patch function directly"""

    def test_construct_partial_patch_basic(self):
        """Test basic functionality of construct_partial_patch"""
        original_patch = """diff --git a/file1.py b/file1.py
index abc123..def456 100644
--- a/file1.py
+++ b/file1.py
@@ -10,7 +10,7 @@
 def existing_function():
     return True
 
-def needs_fix():
+def needs_fix_fixed():
     return "fixed"
 
 # Other code
diff --git a/file2.py b/file2.py
index 123abc..456def 100644
--- a/file2.py
+++ b/file2.py
@@ -5,6 +5,9 @@
 
 # Existing code
 
+def new_function():
+    return "new feature"
+
 class TestClass:
     def test_method(self):
         pass"""
        
        exclude_patch = """diff --git a/file1.py b/file1.py
index abc123..def456 100644
--- a/file1.py
+++ b/file1.py
@@ -10,7 +10,7 @@
 def existing_function():
     return True
 
-def needs_fix():
+def needs_fix_fixed():
     return "fixed"
 
 # Other code"""
        
        partial_patch = construct_partial_patch(original_patch, exclude_patch)
        
        # The partial patch should only contain file2.py changes
        assert partial_patch is not None, "partial_patch should not be None"
        assert "file2.py" in partial_patch
        assert "new_function" in partial_patch
        assert "def needs_fix_fixed" not in partial_patch
        assert "file1.py" not in partial_patch

    def test_construct_partial_patch_multiple_hunks(self):
        """Test construct_partial_patch with multiple hunks in a file"""
        original_patch = """diff --git a/file1.py b/file1.py
index abc123..def456 100644
--- a/file1.py
+++ b/file1.py
@@ -10,7 +10,7 @@
 def existing_function():
     return True
 
-def needs_fix():
+def needs_fix_fixed():
     return "fixed"
 
 # Other code
@@ -50,6 +50,10 @@
 
 # More code
 
+def another_new_function():
+    return "another new feature"
+
 # End of file"""
        
        # Exclude only the first hunk
        exclude_patch = """diff --git a/file1.py b/file1.py
index abc123..def456 100644
--- a/file1.py
+++ b/file1.py
@@ -10,7 +10,7 @@
 def existing_function():
     return True
 
-def needs_fix():
+def needs_fix_fixed():
     return "fixed"
 
 # Other code"""
        
        partial_patch = construct_partial_patch(original_patch, exclude_patch)
        
        # The partial patch should only contain the second hunk
        assert partial_patch is not None, "partial_patch should not be None"
        assert "file1.py" in partial_patch
        assert "another_new_function" in partial_patch
        assert "def needs_fix_fixed" not in partial_patch
        
        # Check for hunk headers
        assert "@@ -50,6 +50,10 @@" in partial_patch
        assert "@@ -10,7 +10,7 @@" not in partial_patch

    def test_construct_partial_patch_exact_hunks(self):
        """Test construct_partial_patch excludes only exact matching hunks"""
        original_patch = """diff --git a/file1.py b/file1.py
index abc123..def456 100644
--- a/file1.py
+++ b/file1.py
@@ -10,7 +10,7 @@
 def existing_function():
     return True
 
-def needs_fix():
+def needs_fix_fixed():
     return "fixed"
 
 # Other code
diff --git a/file2.py b/file2.py
index 123abc..456def 100644
--- a/file2.py
+++ b/file2.py
@@ -5,6 +5,9 @@
 
 # Existing code
 
+def new_function():
+    return "new feature"
+
 class TestClass:
     def test_method(self):
         pass"""
        
        # Similar but not identical hunk (different whitespace)
        exclude_patch = """diff --git a/file1.py b/file1.py
index abc123..def456 100644
--- a/file1.py
+++ b/file1.py
@@ -10,7 +10,7 @@
 def existing_function():
     return True
 
-def needs_fix():
+def needs_fix_fixed():  
     return "fixed"
 
 # Other code"""
        
        partial_patch = construct_partial_patch(original_patch, exclude_patch)
        
        # Both file changes should remain in the patch since hunks don't match exactly
        assert partial_patch is not None, "partial_patch should not be None"
        assert "file1.py" in partial_patch
        assert "file2.py" in partial_patch 