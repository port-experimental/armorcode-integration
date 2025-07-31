"""
Unit tests for the enhanced CLI controller.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cli_controller import CLIConfig, CLIController


class TestCLIConfig(unittest.TestCase):
    """Tests for the CLIConfig dataclass."""
    
    def test_cli_config_creation(self):
        """Test that CLIConfig can be created with all parameters."""
        config = CLIConfig(
            steps=["product", "subproduct"],
            after_key=12345,
            finding_filters_path="/path/to/filters.json",
            batch_size=15,
            retry_attempts=5,
            dry_run=True
        )
        
        self.assertEqual(config.steps, ["product", "subproduct"])
        self.assertEqual(config.after_key, 12345)
        self.assertEqual(config.finding_filters_path, "/path/to/filters.json")
        self.assertEqual(config.batch_size, 15)
        self.assertEqual(config.retry_attempts, 5)
        self.assertTrue(config.dry_run)


class TestCLIController(unittest.TestCase):
    """Tests for the CLIController class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.controller = CLIController()
    
    def test_available_steps(self):
        """Test that available steps are correctly defined."""
        expected_steps = ["product", "subproduct", "finding"]
        self.assertEqual(self.controller.AVAILABLE_STEPS, expected_steps)
        self.assertEqual(self.controller.get_available_steps(), expected_steps)
    
    def test_default_values(self):
        """Test that default values are correctly set."""
        self.assertEqual(self.controller.DEFAULT_BATCH_SIZE, 20)
        self.assertEqual(self.controller.DEFAULT_RETRY_ATTEMPTS, 3)
        self.assertEqual(self.controller.MAX_BATCH_SIZE, 20)
    
    def test_parse_arguments_defaults(self):
        """Test parsing arguments with default values."""
        config = self.controller.parse_arguments([])
        
        self.assertEqual(config.steps, ["product", "subproduct", "finding"])
        self.assertIsNone(config.after_key)
        self.assertIsNone(config.finding_filters_path)
        self.assertEqual(config.batch_size, 20)
        self.assertEqual(config.retry_attempts, 3)
        self.assertFalse(config.dry_run)
    
    def test_parse_arguments_custom_steps(self):
        """Test parsing custom steps argument."""
        config = self.controller.parse_arguments(["--steps", "product,finding"])
        
        self.assertEqual(config.steps, ["product", "finding"])
    
    def test_parse_arguments_custom_steps_with_spaces(self):
        """Test parsing custom steps with spaces."""
        config = self.controller.parse_arguments(["--steps", "product, finding, subproduct"])
        
        self.assertEqual(config.steps, ["product", "finding", "subproduct"])
    
    def test_parse_arguments_after_key(self):
        """Test parsing after-key argument."""
        config = self.controller.parse_arguments(["--after-key", "12345"])
        
        self.assertEqual(config.after_key, 12345)
    
    def test_parse_arguments_batch_size(self):
        """Test parsing batch-size argument."""
        config = self.controller.parse_arguments(["--batch-size", "15"])
        
        self.assertEqual(config.batch_size, 15)
    
    def test_parse_arguments_retry_attempts(self):
        """Test parsing retry-attempts argument."""
        config = self.controller.parse_arguments(["--retry-attempts", "5"])
        
        self.assertEqual(config.retry_attempts, 5)
    
    def test_parse_arguments_dry_run(self):
        """Test parsing dry-run flag."""
        config = self.controller.parse_arguments(["--dry-run"])
        
        self.assertTrue(config.dry_run)
    
    def test_parse_arguments_all_options(self):
        """Test parsing all options together."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"test": "filter"}, f)
            temp_file = f.name
        
        try:
            config = self.controller.parse_arguments([
                "--steps", "product,finding",
                "--after-key", "12345",
                "--finding-filters", temp_file,
                "--batch-size", "15",
                "--retry-attempts", "5",
                "--dry-run"
            ])
            
            self.assertEqual(config.steps, ["product", "finding"])
            self.assertEqual(config.after_key, 12345)
            self.assertEqual(config.finding_filters_path, temp_file)
            self.assertEqual(config.batch_size, 15)
            self.assertEqual(config.retry_attempts, 5)
            self.assertTrue(config.dry_run)
        finally:
            os.unlink(temp_file)


class TestCLIValidation(unittest.TestCase):
    """Tests for CLI validation logic."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.controller = CLIController()
    
    def test_validate_valid_configuration(self):
        """Test validation of a valid configuration."""
        config = CLIConfig(
            steps=["product", "finding"],
            after_key=12345,
            finding_filters_path=None,
            batch_size=15,
            retry_attempts=3,
            dry_run=False
        )
        
        # Should not raise any exception
        self.controller.validate_configuration(config)
    
    def test_validate_invalid_steps(self):
        """Test validation fails for invalid step names."""
        config = CLIConfig(
            steps=["invalid_step", "product"],
            after_key=None,
            finding_filters_path=None,
            batch_size=20,
            retry_attempts=3,
            dry_run=False
        )
        
        with self.assertRaises(SystemExit):
            self.controller.validate_configuration(config)
    
    def test_validate_empty_steps(self):
        """Test validation fails for empty steps list."""
        config = CLIConfig(
            steps=[],
            after_key=None,
            finding_filters_path=None,
            batch_size=20,
            retry_attempts=3,
            dry_run=False
        )
        
        with self.assertRaises(SystemExit):
            self.controller.validate_configuration(config)
    
    def test_validate_invalid_batch_size_zero(self):
        """Test validation fails for zero batch size."""
        config = CLIConfig(
            steps=["product"],
            after_key=None,
            finding_filters_path=None,
            batch_size=0,
            retry_attempts=3,
            dry_run=False
        )
        
        with self.assertRaises(SystemExit):
            self.controller.validate_configuration(config)
    
    def test_validate_invalid_batch_size_negative(self):
        """Test validation fails for negative batch size."""
        config = CLIConfig(
            steps=["product"],
            after_key=None,
            finding_filters_path=None,
            batch_size=-5,
            retry_attempts=3,
            dry_run=False
        )
        
        with self.assertRaises(SystemExit):
            self.controller.validate_configuration(config)
    
    def test_validate_invalid_batch_size_too_large(self):
        """Test validation fails for batch size exceeding maximum."""
        config = CLIConfig(
            steps=["product"],
            after_key=None,
            finding_filters_path=None,
            batch_size=25,  # Exceeds MAX_BATCH_SIZE of 20
            retry_attempts=3,
            dry_run=False
        )
        
        with self.assertRaises(SystemExit):
            self.controller.validate_configuration(config)
    
    def test_validate_invalid_retry_attempts_negative(self):
        """Test validation fails for negative retry attempts."""
        config = CLIConfig(
            steps=["product"],
            after_key=None,
            finding_filters_path=None,
            batch_size=20,
            retry_attempts=-1,
            dry_run=False
        )
        
        with self.assertRaises(SystemExit):
            self.controller.validate_configuration(config)
    
    def test_validate_retry_attempts_zero_allowed(self):
        """Test validation allows zero retry attempts (disables retries)."""
        config = CLIConfig(
            steps=["product"],
            after_key=None,
            finding_filters_path=None,
            batch_size=20,
            retry_attempts=0,
            dry_run=False
        )
        
        # Should not raise any exception
        self.controller.validate_configuration(config)
    
    def test_validate_invalid_after_key_negative(self):
        """Test validation fails for negative after key."""
        config = CLIConfig(
            steps=["product"],
            after_key=-1,
            finding_filters_path=None,
            batch_size=20,
            retry_attempts=3,
            dry_run=False
        )
        
        with self.assertRaises(SystemExit):
            self.controller.validate_configuration(config)
    
    def test_validate_after_key_zero_allowed(self):
        """Test validation allows zero after key."""
        config = CLIConfig(
            steps=["product"],
            after_key=0,
            finding_filters_path=None,
            batch_size=20,
            retry_attempts=3,
            dry_run=False
        )
        
        # Should not raise any exception
        self.controller.validate_configuration(config)


class TestFiltersFileValidation(unittest.TestCase):
    """Tests for filters file validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.controller = CLIController()
    
    def test_validate_filters_file_valid_json(self):
        """Test validation of valid JSON filters file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"severity": ["HIGH", "CRITICAL"]}, f)
            temp_file = f.name
        
        try:
            self.assertTrue(self.controller._validate_filters_file(temp_file))
        finally:
            os.unlink(temp_file)
    
    def test_validate_filters_file_invalid_json(self):
        """Test validation fails for invalid JSON."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{ invalid json }")
            temp_file = f.name
        
        try:
            self.assertFalse(self.controller._validate_filters_file(temp_file))
        finally:
            os.unlink(temp_file)
    
    def test_validate_filters_file_nonexistent(self):
        """Test validation fails for non-existent file."""
        self.assertFalse(self.controller._validate_filters_file("/nonexistent/file.json"))
    
    def test_validate_filters_file_directory(self):
        """Test validation fails for directory instead of file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertFalse(self.controller._validate_filters_file(temp_dir))
    
    def test_validate_configuration_with_invalid_filters_file(self):
        """Test configuration validation fails for invalid filters file."""
        config = CLIConfig(
            steps=["finding"],
            after_key=None,
            finding_filters_path="/nonexistent/file.json",
            batch_size=20,
            retry_attempts=3,
            dry_run=False
        )
        
        with self.assertRaises(SystemExit):
            self.controller.validate_configuration(config)
    
    def test_validate_configuration_with_valid_filters_file(self):
        """Test configuration validation passes for valid filters file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"severity": ["HIGH"]}, f)
            temp_file = f.name
        
        try:
            config = CLIConfig(
                steps=["finding"],
                after_key=None,
                finding_filters_path=temp_file,
                batch_size=20,
                retry_attempts=3,
                dry_run=False
            )
            
            # Should not raise any exception
            self.controller.validate_configuration(config)
        finally:
            os.unlink(temp_file)


class TestArgumentParsingIntegration(unittest.TestCase):
    """Integration tests for argument parsing with validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.controller = CLIController()
    
    def test_parse_arguments_with_validation_failure(self):
        """Test that parse_arguments fails when validation fails."""
        with self.assertRaises(SystemExit):
            self.controller.parse_arguments(["--steps", "invalid_step"])
    
    def test_parse_arguments_with_validation_success(self):
        """Test that parse_arguments succeeds when validation passes."""
        config = self.controller.parse_arguments(["--steps", "product,finding"])
        
        self.assertEqual(config.steps, ["product", "finding"])
    
    def test_parse_arguments_batch_size_validation(self):
        """Test batch size validation during argument parsing."""
        with self.assertRaises(SystemExit):
            self.controller.parse_arguments(["--batch-size", "25"])  # Exceeds maximum
    
    def test_parse_arguments_negative_after_key_validation(self):
        """Test negative after key validation during argument parsing."""
        with self.assertRaises(SystemExit):
            self.controller.parse_arguments(["--after-key", "-1"])


if __name__ == "__main__":
    unittest.main()