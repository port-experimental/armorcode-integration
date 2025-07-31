#!/usr/bin/env python3
"""
Unit tests for data transformers module.

Tests the transformation of ArmorCode API data to Port-compatible formats.
"""

import pytest
from datetime import datetime, timezone
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from armorcode_integration.utils.data_transformers import (
    timestamp_to_rfc3339,
    transform_finding_timestamps,
    batch_transform_findings
)


class TestTimestampToRFC3339:
    """Test timestamp conversion to RFC 3339 format."""
    
    def test_valid_timestamp_conversion(self):
        """Test conversion of valid millisecond timestamps."""
        # Test data from ArmorCode API
        timestamp_ms = 1732564920000  # 2024-11-25T12:35:20.000Z
        result = timestamp_to_rfc3339(timestamp_ms)
        
        assert result == "2024-11-25T12:35:20.000Z"
        
    def test_another_valid_timestamp(self):
        """Test another valid timestamp conversion."""
        timestamp_ms = 1752603329000  # 2025-07-15T09:01:02.000Z (approximately)
        result = timestamp_to_rfc3339(timestamp_ms)
        
        # Verify it's in correct RFC 3339 format
        assert result.endswith("Z")
        assert "T" in result
        assert len(result) == 24  # YYYY-MM-DDTHH:MM:SS.sssZ
        
    def test_string_timestamp_conversion(self):
        """Test conversion of string timestamps."""
        timestamp_str = "1732564920000"
        result = timestamp_to_rfc3339(timestamp_str)
        
        assert result == "2024-11-25T12:35:20.000Z"
        
    def test_none_timestamp(self):
        """Test handling of None timestamp."""
        result = timestamp_to_rfc3339(None)
        assert result is None
        
    def test_invalid_timestamp(self):
        """Test handling of invalid timestamp."""
        result = timestamp_to_rfc3339("invalid")
        assert result is None
        
    def test_empty_string_timestamp(self):
        """Test handling of empty string timestamp."""
        result = timestamp_to_rfc3339("")
        assert result is None
        
    def test_negative_timestamp(self):
        """Test handling of negative timestamp."""
        result = timestamp_to_rfc3339(-1000)
        # Should handle gracefully, either return None or valid date
        # We accept either behavior as long as it doesn't crash
        assert result is None or isinstance(result, str)


class TestTransformFindingTimestamps:
    """Test finding timestamp transformation."""
    
    def test_transform_valid_finding(self):
        """Test transformation of a finding with valid timestamps."""
        finding = {
            "id": 12345,
            "title": "Test Finding",
            "createdAt": 1732564920000,
            "lastUpdated": 1752603329000,
            "severity": "HIGH"
        }
        
        result = transform_finding_timestamps(finding)
        
        # Original finding should be unchanged
        assert finding["createdAt"] == 1732564920000
        assert finding["lastUpdated"] == 1752603329000
        
        # Result should have RFC 3339 timestamps
        assert result["createdAt"] == "2024-11-25T12:35:20.000Z"
        assert result["lastUpdated"].endswith("Z")
        assert "T" in result["lastUpdated"]
        
        # Other fields should be unchanged
        assert result["id"] == 12345
        assert result["title"] == "Test Finding"
        assert result["severity"] == "HIGH"
        
    def test_transform_finding_with_missing_timestamps(self):
        """Test transformation of finding with missing timestamps."""
        finding = {
            "id": 12345,
            "title": "Test Finding",
            "severity": "HIGH"
        }
        
        result = transform_finding_timestamps(finding)
        
        # Should not crash and preserve other fields
        assert result["id"] == 12345
        assert result["title"] == "Test Finding"
        assert result["severity"] == "HIGH"
        
    def test_transform_finding_with_null_timestamps(self):
        """Test transformation of finding with null timestamps."""
        finding = {
            "id": 12345,
            "title": "Test Finding",
            "createdAt": None,
            "lastUpdated": None,
            "severity": "HIGH"
        }
        
        result = transform_finding_timestamps(finding)
        
        # Null timestamps should remain null
        assert result["createdAt"] is None
        assert result["lastUpdated"] is None
        
        # Other fields should be unchanged
        assert result["id"] == 12345
        assert result["title"] == "Test Finding"
        assert result["severity"] == "HIGH"
        
    def test_transform_non_dict_input(self):
        """Test transformation with non-dictionary input."""
        result = transform_finding_timestamps("not a dict")
        assert result == "not a dict"
        
        result = transform_finding_timestamps(None)
        assert result is None


class TestBatchTransformFindings:
    """Test batch transformation of findings."""
    
    def test_batch_transform_valid_findings(self):
        """Test batch transformation of multiple findings."""
        findings = [
            {
                "id": 1,
                "title": "Finding 1",
                "createdAt": 1732564920000,
                "lastUpdated": 1752603329000
            },
            {
                "id": 2,
                "title": "Finding 2", 
                "createdAt": 1732564920000,
                "lastUpdated": 1752603329000
            }
        ]
        
        results = batch_transform_findings(findings)
        
        assert len(results) == 2
        
        # Check first finding
        assert results[0]["id"] == 1
        assert results[0]["createdAt"] == "2024-11-25T12:35:20.000Z"
        assert results[0]["lastUpdated"].endswith("Z")
        
        # Check second finding
        assert results[1]["id"] == 2
        assert results[1]["createdAt"] == "2024-11-25T12:35:20.000Z"
        assert results[1]["lastUpdated"].endswith("Z")
        
    def test_batch_transform_empty_list(self):
        """Test batch transformation of empty list."""
        result = batch_transform_findings([])
        assert result == []
        
    def test_batch_transform_non_list_input(self):
        """Test batch transformation with non-list input."""
        result = batch_transform_findings("not a list")
        assert result == "not a list"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])