"""
Filter Manager for ArmorCode Finding Filters and AfterKey Support

This module provides functionality for parsing, validating, and combining
ArmorCode finding filters with afterKey parameters for enhanced API querying.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class FilterValidationError(Exception):
    """Custom exception for filter validation errors."""
    pass


class FilterManager:
    """
    Manages ArmorCode finding filters and afterKey parameter combination.
    
    Provides functionality to:
    - Parse JSON filter files
    - Validate FindingFiltersRequestDto format compliance
    - Combine afterKey parameter with existing filters
    - Handle errors gracefully with detailed messages
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def parse_filters_file(self, file_path: str) -> Dict[str, Any]:
        """
        Parse a JSON filter file and return the filters dictionary.
        
        Args:
            file_path: Path to the JSON filter file
            
        Returns:
            Dict[str, Any]: Parsed filters dictionary
            
        Raises:
            FilterValidationError: If file cannot be read or parsed
        """
        try:
            path = Path(file_path)
            
            # Check if file exists
            if not path.exists():
                raise FilterValidationError(f"Filter file does not exist: {file_path}")
            
            # Check if it's a file (not a directory)
            if not path.is_file():
                raise FilterValidationError(f"Filter path is not a file: {file_path}")
            
            # Read and parse JSON
            with open(path, 'r', encoding='utf-8') as f:
                try:
                    filters = json.load(f)
                except json.JSONDecodeError as e:
                    raise FilterValidationError(
                        f"Invalid JSON in filter file {file_path}: {e}"
                    )
            
            # Ensure we have a dictionary
            if not isinstance(filters, dict):
                raise FilterValidationError(
                    f"Filter file must contain a JSON object, got {type(filters).__name__}: {file_path}"
                )
            
            self.logger.debug(f"Successfully parsed filter file: {file_path}")
            return filters
            
        except (OSError, PermissionError) as e:
            raise FilterValidationError(f"Cannot read filter file {file_path}: {e}")
    
    def validate_filters(self, filters: Dict[str, Any]) -> bool:
        """
        Validate filters against FindingFiltersRequestDto format.
        
        Based on ArmorCode API documentation, the FindingFiltersRequestDto can contain:
        - Various filter fields for findings
        - Nested objects and arrays
        - String, number, boolean, and array values
        
        Args:
            filters: Dictionary containing filter parameters
            
        Returns:
            bool: True if filters are valid
            
        Raises:
            FilterValidationError: If filters are invalid with detailed error message
        """
        if not isinstance(filters, dict):
            raise FilterValidationError(
                f"Filters must be a dictionary, got {type(filters).__name__}"
            )
        
        # Define known valid filter fields based on ArmorCode API
        # This is a comprehensive list of potential filter fields
        valid_filter_fields = {
            'severity', 'status', 'source', 'subProduct', 'product', 'cve',
            'title', 'description', 'mitigation', 'findingUrl', 'ticketUrl',
            'createdAt', 'updatedAt', 'age', 'tags', 'assignee', 'priority',
            'category', 'subcategory', 'riskScore', 'exploitability',
            'businessImpact', 'technicalImpact', 'environment', 'asset',
            'component', 'version', 'library', 'framework', 'language',
            'scanType', 'toolVersion', 'confidence', 'falsePositive',
            'suppressed', 'verified', 'remediated', 'accepted', 'deferred'
        }
        
        # Validate each filter field
        for field_name, field_value in filters.items():
            try:
                self._validate_filter_field(field_name, field_value, valid_filter_fields)
            except FilterValidationError as e:
                raise FilterValidationError(f"Invalid filter field '{field_name}': {e}")
        
        self.logger.debug(f"Successfully validated {len(filters)} filter fields")
        return True
    
    def _validate_filter_field(self, field_name: str, field_value: Any, valid_fields: set) -> None:
        """
        Validate a single filter field.
        
        Args:
            field_name: Name of the filter field
            field_value: Value of the filter field
            valid_fields: Set of valid field names
            
        Raises:
            FilterValidationError: If field is invalid
        """
        # Check if field name is valid (allow unknown fields with warning)
        if field_name not in valid_fields:
            self.logger.warning(f"Unknown filter field '{field_name}' - proceeding anyway")
        
        # Validate field value types
        if field_value is None:
            return  # None values are acceptable
        
        # Allow basic JSON types
        if isinstance(field_value, (str, int, float, bool)):
            return
        
        # Allow arrays of basic types
        if isinstance(field_value, list):
            for i, item in enumerate(field_value):
                if not isinstance(item, (str, int, float, bool, dict, type(None))):
                    raise FilterValidationError(
                        f"Array item at index {i} has invalid type {type(item).__name__}"
                    )
                # Recursively validate dict items in arrays
                if isinstance(item, dict):
                    for sub_field, sub_value in item.items():
                        self._validate_filter_field(sub_field, sub_value, valid_fields)
            return
        
        # Allow nested objects
        if isinstance(field_value, dict):
            for sub_field, sub_value in field_value.items():
                self._validate_filter_field(sub_field, sub_value, valid_fields)
            return
        
        # Reject other types
        raise FilterValidationError(
            f"Invalid value type {type(field_value).__name__} for field '{field_name}'"
        )
    
    def combine_filters(self, base_filters: Optional[Dict[str, Any]], after_key: Optional[int]) -> Dict[str, Any]:
        """
        Combine base filters with afterKey parameter.
        
        The afterKey parameter is used for pagination in the ArmorCode API.
        It should be combined with existing filters to create the final request body.
        
        Args:
            base_filters: Base filters dictionary (can be None)
            after_key: AfterKey parameter for pagination (can be None)
            
        Returns:
            Dict[str, Any]: Combined filters dictionary
            
        Raises:
            FilterValidationError: If afterKey is invalid
        """
        # Start with base filters or empty dict
        combined_filters = (base_filters or {}).copy()
        
        # Validate and add afterKey if provided
        if after_key is not None:
            if not isinstance(after_key, int):
                raise FilterValidationError(
                    f"afterKey must be an integer, got {type(after_key).__name__}"
                )
            
            if after_key < 0:
                raise FilterValidationError(
                    f"afterKey must be non-negative, got {after_key}"
                )
            
            # Note: afterKey is typically used as a query parameter, not in the request body
            # However, some APIs might accept it in the body as well
            # We'll add it to the filters dict for flexibility
            combined_filters['afterKey'] = after_key
            
            self.logger.debug(f"Added afterKey {after_key} to filters")
        
        self.logger.debug(f"Combined filters: {len(combined_filters)} fields")
        return combined_filters
    
    def load_and_validate_filters(self, file_path: Optional[str]) -> Optional[Dict[str, Any]]:
        """
        Convenience method to load and validate filters from a file.
        
        Args:
            file_path: Path to filter file (can be None)
            
        Returns:
            Optional[Dict[str, Any]]: Validated filters or None if no file provided
            
        Raises:
            FilterValidationError: If file cannot be loaded or filters are invalid
        """
        if not file_path:
            return None
        
        filters = self.parse_filters_file(file_path)
        self.validate_filters(filters)
        return filters
    
    def create_api_request_body(self, 
                               filters_file_path: Optional[str] = None,
                               after_key: Optional[int] = None) -> Dict[str, Any]:
        """
        Create a complete API request body for ArmorCode findings API.
        
        This method combines file-based filters with afterKey parameter
        to create the final request body for the ArmorCode findings API.
        
        Args:
            filters_file_path: Path to JSON filter file (optional)
            after_key: AfterKey parameter for pagination (optional)
            
        Returns:
            Dict[str, Any]: Complete request body for ArmorCode API
            
        Raises:
            FilterValidationError: If filters are invalid
        """
        # Load filters from file if provided
        base_filters = self.load_and_validate_filters(filters_file_path)
        
        # Combine with afterKey
        request_body = self.combine_filters(base_filters, after_key)
        
        self.logger.info(f"Created API request body with {len(request_body)} parameters")
        return request_body
    
    def get_filter_summary(self, filters: Dict[str, Any]) -> str:
        """
        Generate a human-readable summary of the filters.
        
        Args:
            filters: Filters dictionary
            
        Returns:
            str: Human-readable filter summary
        """
        if not filters:
            return "No filters applied"
        
        summary_parts = []
        
        for key, value in filters.items():
            if key == 'afterKey':
                summary_parts.append(f"afterKey: {value}")
            elif isinstance(value, list):
                summary_parts.append(f"{key}: {len(value)} items")
            elif isinstance(value, dict):
                summary_parts.append(f"{key}: nested object")
            else:
                summary_parts.append(f"{key}: {value}")
        
        return f"Filters applied: {', '.join(summary_parts)}"