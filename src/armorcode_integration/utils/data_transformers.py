#!/usr/bin/env python3
"""
Data Transformers for ArmorCode Integration

This module contains utility functions for transforming data from ArmorCode API
format to Port-compatible formats.
"""

from datetime import datetime, timezone
from typing import Optional, Union


def timestamp_to_rfc3339(timestamp: Optional[Union[int, float, str]]) -> Optional[str]:
    """
    Convert ArmorCode API timestamp to RFC 3339 format for Port compatibility.
    
    ArmorCode API returns timestamps as milliseconds since epoch (e.g., 1732564920000).
    Port expects RFC 3339 format (e.g., "2024-11-25T12:35:20.000Z").
    
    Args:
        timestamp: Timestamp in milliseconds since epoch, or None
        
    Returns:
        RFC 3339 formatted timestamp string, or None if input is None/invalid
        
    Examples:
        >>> timestamp_to_rfc3339(1732564920000)
        '2024-11-25T12:35:20.000Z'
        
        >>> timestamp_to_rfc3339(None)
        None
        
        >>> timestamp_to_rfc3339("1732564920000")
        '2024-11-25T12:35:20.000Z'
    """
    if timestamp is None:
        return None
        
    try:
        # Convert to int if string
        if isinstance(timestamp, str):
            timestamp = int(timestamp)
            
        # Convert milliseconds to seconds
        timestamp_seconds = timestamp / 1000.0
        
        # Create datetime object in UTC
        dt = datetime.fromtimestamp(timestamp_seconds, tz=timezone.utc)
        
        # Format as RFC 3339 with milliseconds
        # Format: YYYY-MM-DDTHH:MM:SS.sssZ
        return dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        
    except (ValueError, TypeError, OverflowError) as e:
        # Log warning but don't fail the entire operation
        import logging
        logging.warning(f"Failed to convert timestamp {timestamp} to RFC 3339: {e}")
        return None


def transform_finding_timestamps(finding_data: dict) -> dict:
    """
    Transform timestamp fields in a finding object to RFC 3339 format.
    
    Args:
        finding_data: Dictionary containing finding data from ArmorCode API
        
    Returns:
        Dictionary with transformed timestamp fields
    """
    if not isinstance(finding_data, dict):
        return finding_data
        
    # Create a copy to avoid modifying the original
    transformed = finding_data.copy()
    
    # Transform timestamp fields
    timestamp_fields = ['createdAt', 'lastUpdated']
    
    for field in timestamp_fields:
        if field in transformed:
            transformed[field] = timestamp_to_rfc3339(transformed[field])
            
    return transformed


def batch_transform_findings(findings: list) -> list:
    """
    Transform timestamp fields for a batch of findings.
    
    Args:
        findings: List of finding dictionaries from ArmorCode API
        
    Returns:
        List of findings with transformed timestamps
    """
    if not isinstance(findings, list):
        return findings
        
    return [transform_finding_timestamps(finding) for finding in findings]