"""
Unit tests for FilterManager class.

Tests cover:
- JSON filter file parsing
- Filter validation logic
- AfterKey parameter combination
- Error handling for malformed JSON and invalid structures
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, mock_open

from filter_manager import FilterManager, FilterValidationError


class TestFilterManager(unittest.TestCase):
    """Test cases for FilterManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.filter_manager = FilterManager()
    
    def test_parse_filters_file_success(self):
        """Test successful parsing of a valid JSON filter file."""
        test_filters = {
            "severity": ["HIGH", "CRITICAL"],
            "status": "OPEN",
            "subProduct": {"id": 123}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_filters, f)
            temp_file = f.name
        
        try:
            result = self.filter_manager.parse_filters_file(temp_file)
            self.assertEqual(result, test_filters)
        finally:
            Path(temp_file).unlink()
    
    def test_parse_filters_file_not_exists(self):
        """Test parsing non-existent file raises FilterValidationError."""
        with self.assertRaises(FilterValidationError) as context:
            self.filter_manager.parse_filters_file("/nonexistent/file.json")
        
        self.assertIn("does not exist", str(context.exception))
    
    def test_parse_filters_file_invalid_json(self):
        """Test parsing invalid JSON raises FilterValidationError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"invalid": json}')  # Missing quotes around json
            temp_file = f.name
        
        try:
            with self.assertRaises(FilterValidationError) as context:
                self.filter_manager.parse_filters_file(temp_file)
            
            self.assertIn("Invalid JSON", str(context.exception))
        finally:
            Path(temp_file).unlink()
    
    def test_parse_filters_file_not_dict(self):
        """Test parsing JSON that's not a dictionary raises FilterValidationError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(["not", "a", "dict"], f)
            temp_file = f.name
        
        try:
            with self.assertRaises(FilterValidationError) as context:
                self.filter_manager.parse_filters_file(temp_file)
            
            self.assertIn("must contain a JSON object", str(context.exception))
        finally:
            Path(temp_file).unlink()
    
    def test_parse_filters_file_directory(self):
        """Test parsing a directory path raises FilterValidationError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(FilterValidationError) as context:
                self.filter_manager.parse_filters_file(temp_dir)
            
            self.assertIn("not a file", str(context.exception))
    
    def test_validate_filters_success_basic(self):
        """Test successful validation of basic filters."""
        filters = {
            "severity": "HIGH",
            "status": "OPEN",
            "age": 30,
            "verified": True,
            "tags": ["security", "critical"]
        }
        
        result = self.filter_manager.validate_filters(filters)
        self.assertTrue(result)
    
    def test_validate_filters_success_nested(self):
        """Test successful validation of nested filters."""
        filters = {
            "subProduct": {
                "id": 123,
                "name": "test-product"
            },
            "severity": ["HIGH", "CRITICAL"],
            "metadata": {
                "source": "scanner",
                "confidence": 0.95
            }
        }
        
        result = self.filter_manager.validate_filters(filters)
        self.assertTrue(result)
    
    def test_validate_filters_success_empty(self):
        """Test successful validation of empty filters."""
        result = self.filter_manager.validate_filters({})
        self.assertTrue(result)
    
    def test_validate_filters_success_none_values(self):
        """Test successful validation with None values."""
        filters = {
            "severity": "HIGH",
            "description": None,
            "tags": ["test"]
        }
        
        result = self.filter_manager.validate_filters(filters)
        self.assertTrue(result)
    
    def test_validate_filters_not_dict(self):
        """Test validation fails for non-dictionary input."""
        with self.assertRaises(FilterValidationError) as context:
            self.filter_manager.validate_filters("not a dict")
        
        self.assertIn("must be a dictionary", str(context.exception))
    
    def test_validate_filters_invalid_value_type(self):
        """Test validation fails for invalid value types."""
        filters = {
            "severity": "HIGH",
            "invalid_field": set([1, 2, 3])  # Sets are not allowed
        }
        
        with self.assertRaises(FilterValidationError) as context:
            self.filter_manager.validate_filters(filters)
        
        self.assertIn("Invalid value type", str(context.exception))
    
    def test_validate_filters_invalid_array_item(self):
        """Test validation fails for invalid array item types."""
        filters = {
            "tags": ["valid", set([1, 2])]  # Set in array is not allowed
        }
        
        with self.assertRaises(FilterValidationError) as context:
            self.filter_manager.validate_filters(filters)
        
        self.assertIn("Array item", str(context.exception))
    
    def test_validate_filters_unknown_field_warning(self):
        """Test that unknown fields generate warnings but don't fail validation."""
        filters = {
            "unknown_field": "value",
            "severity": "HIGH"
        }
        
        with patch.object(self.filter_manager.logger, 'warning') as mock_warning:
            result = self.filter_manager.validate_filters(filters)
            self.assertTrue(result)
            mock_warning.assert_called_once()
            self.assertIn("Unknown filter field", mock_warning.call_args[0][0])
    
    def test_combine_filters_both_none(self):
        """Test combining when both base_filters and after_key are None."""
        result = self.filter_manager.combine_filters(None, None)
        self.assertEqual(result, {})
    
    def test_combine_filters_only_base(self):
        """Test combining with only base filters."""
        base_filters = {"severity": "HIGH", "status": "OPEN"}
        result = self.filter_manager.combine_filters(base_filters, None)
        self.assertEqual(result, base_filters)
    
    def test_combine_filters_only_after_key(self):
        """Test combining with only after_key."""
        result = self.filter_manager.combine_filters(None, 12345)
        self.assertEqual(result, {"afterKey": 12345})
    
    def test_combine_filters_both_provided(self):
        """Test combining both base filters and after_key."""
        base_filters = {"severity": "HIGH", "status": "OPEN"}
        result = self.filter_manager.combine_filters(base_filters, 12345)
        
        expected = base_filters.copy()
        expected["afterKey"] = 12345
        self.assertEqual(result, expected)
    
    def test_combine_filters_preserves_original(self):
        """Test that combining filters doesn't modify the original base_filters."""
        base_filters = {"severity": "HIGH"}
        original_filters = base_filters.copy()
        
        self.filter_manager.combine_filters(base_filters, 12345)
        self.assertEqual(base_filters, original_filters)
    
    def test_combine_filters_invalid_after_key_type(self):
        """Test combining with invalid after_key type raises FilterValidationError."""
        with self.assertRaises(FilterValidationError) as context:
            self.filter_manager.combine_filters({}, "not_an_int")
        
        self.assertIn("must be an integer", str(context.exception))
    
    def test_combine_filters_negative_after_key(self):
        """Test combining with negative after_key raises FilterValidationError."""
        with self.assertRaises(FilterValidationError) as context:
            self.filter_manager.combine_filters({}, -1)
        
        self.assertIn("must be non-negative", str(context.exception))
    
    def test_load_and_validate_filters_none_path(self):
        """Test load_and_validate_filters returns None for None path."""
        result = self.filter_manager.load_and_validate_filters(None)
        self.assertIsNone(result)
    
    def test_load_and_validate_filters_empty_path(self):
        """Test load_and_validate_filters returns None for empty path."""
        result = self.filter_manager.load_and_validate_filters("")
        self.assertIsNone(result)
    
    def test_load_and_validate_filters_success(self):
        """Test successful load and validation of filters."""
        test_filters = {"severity": "HIGH", "status": "OPEN"}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_filters, f)
            temp_file = f.name
        
        try:
            result = self.filter_manager.load_and_validate_filters(temp_file)
            self.assertEqual(result, test_filters)
        finally:
            Path(temp_file).unlink()
    
    def test_load_and_validate_filters_invalid_file(self):
        """Test load_and_validate_filters raises error for invalid file."""
        with self.assertRaises(FilterValidationError):
            self.filter_manager.load_and_validate_filters("/nonexistent/file.json")
    
    def test_create_api_request_body_no_params(self):
        """Test creating API request body with no parameters."""
        result = self.filter_manager.create_api_request_body()
        self.assertEqual(result, {})
    
    def test_create_api_request_body_only_after_key(self):
        """Test creating API request body with only after_key."""
        result = self.filter_manager.create_api_request_body(after_key=12345)
        self.assertEqual(result, {"afterKey": 12345})
    
    def test_create_api_request_body_only_filters(self):
        """Test creating API request body with only filters file."""
        test_filters = {"severity": "HIGH", "status": "OPEN"}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_filters, f)
            temp_file = f.name
        
        try:
            result = self.filter_manager.create_api_request_body(filters_file_path=temp_file)
            self.assertEqual(result, test_filters)
        finally:
            Path(temp_file).unlink()
    
    def test_create_api_request_body_both_params(self):
        """Test creating API request body with both filters and after_key."""
        test_filters = {"severity": "HIGH", "status": "OPEN"}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_filters, f)
            temp_file = f.name
        
        try:
            result = self.filter_manager.create_api_request_body(
                filters_file_path=temp_file,
                after_key=12345
            )
            
            expected = test_filters.copy()
            expected["afterKey"] = 12345
            self.assertEqual(result, expected)
        finally:
            Path(temp_file).unlink()
    
    def test_get_filter_summary_empty(self):
        """Test filter summary for empty filters."""
        result = self.filter_manager.get_filter_summary({})
        self.assertEqual(result, "No filters applied")
    
    def test_get_filter_summary_basic(self):
        """Test filter summary for basic filters."""
        filters = {
            "severity": "HIGH",
            "status": "OPEN",
            "afterKey": 12345
        }
        
        result = self.filter_manager.get_filter_summary(filters)
        self.assertIn("afterKey: 12345", result)
        self.assertIn("severity: HIGH", result)
        self.assertIn("status: OPEN", result)
    
    def test_get_filter_summary_complex(self):
        """Test filter summary for complex filters."""
        filters = {
            "tags": ["security", "critical"],
            "subProduct": {"id": 123},
            "afterKey": 12345
        }
        
        result = self.filter_manager.get_filter_summary(filters)
        self.assertIn("afterKey: 12345", result)
        self.assertIn("tags: 2 items", result)
        self.assertIn("subProduct: nested object", result)
    
    def test_validate_filter_field_nested_dict_in_array(self):
        """Test validation of nested dictionaries within arrays."""
        filters = {
            "complexField": [
                {"nestedField": "value1"},
                {"nestedField": "value2", "anotherField": 123}
            ]
        }
        
        result = self.filter_manager.validate_filters(filters)
        self.assertTrue(result)
    
    def test_validate_filter_field_invalid_nested_dict_in_array(self):
        """Test validation fails for invalid nested dictionaries in arrays."""
        filters = {
            "complexField": [
                {"validField": "value"},
                {"invalidField": set([1, 2, 3])}  # Invalid type in nested dict
            ]
        }
        
        with self.assertRaises(FilterValidationError) as context:
            self.filter_manager.validate_filters(filters)
        
        self.assertIn("Invalid value type", str(context.exception))


if __name__ == '__main__':
    unittest.main()