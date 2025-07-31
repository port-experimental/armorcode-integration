#!/usr/bin/env python3
"""
Simple integration test for the enhanced findings filtering functionality.
This script tests the FilterManager integration without making actual API calls.
"""

import asyncio
import json
import os
import tempfile
from pathlib import Path

from filter_manager import FilterManager, FilterValidationError


async def test_filter_manager_basic_functionality():
    """Test basic FilterManager functionality."""
    print("Testing FilterManager basic functionality...")
    
    # Create a temporary filter file
    filter_data = {
        "severity": ["HIGH", "CRITICAL"],
        "status": "OPEN",
        "source": ["SAST", "DAST"]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(filter_data, f)
        temp_filter_file = f.name
    
    try:
        # Test FilterManager
        filter_manager = FilterManager()
        
        # Test loading and validating filters
        loaded_filters = filter_manager.load_and_validate_filters(temp_filter_file)
        print(f"✓ Successfully loaded filters: {loaded_filters}")
        
        # Test combining with afterKey
        combined_filters = filter_manager.combine_filters(loaded_filters, 12345)
        print(f"✓ Combined filters with afterKey: {combined_filters}")
        
        # Test filter summary
        summary = filter_manager.get_filter_summary(combined_filters)
        print(f"✓ Filter summary: {summary}")
        
        # Test creating API request body
        request_body = filter_manager.create_api_request_body(
            filters_file_path=temp_filter_file,
            after_key=12345
        )
        print(f"✓ API request body: {request_body}")
        
        print("✓ All FilterManager tests passed!")
        
    except Exception as e:
        print(f"✗ FilterManager test failed: {e}")
        raise
    finally:
        # Cleanup
        os.unlink(temp_filter_file)


async def test_filter_validation_errors():
    """Test FilterManager error handling."""
    print("\nTesting FilterManager error handling...")
    
    filter_manager = FilterManager()
    
    # Test with non-existent file
    try:
        filter_manager.load_and_validate_filters("/tmp/nonexistent_file.json")
        print("✗ Should have raised FilterValidationError for non-existent file")
    except FilterValidationError as e:
        print(f"✓ Correctly caught error for non-existent file: {e}")
    
    # Test with invalid JSON
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write("invalid json {")
        invalid_json_file = f.name
    
    try:
        try:
            filter_manager.load_and_validate_filters(invalid_json_file)
            print("✗ Should have raised FilterValidationError for invalid JSON")
        except FilterValidationError as e:
            print(f"✓ Correctly caught error for invalid JSON: {e}")
    finally:
        os.unlink(invalid_json_file)
    
    # Test with invalid afterKey
    try:
        filter_manager.combine_filters({}, -1)
        print("✗ Should have raised FilterValidationError for negative afterKey")
    except FilterValidationError as e:
        print(f"✓ Correctly caught error for negative afterKey: {e}")
    
    print("✓ All error handling tests passed!")


async def test_cli_integration():
    """Test CLI integration with filtering."""
    print("\nTesting CLI integration...")
    
    from cli_controller import CLIController, CLIConfig
    
    # Create a temporary filter file
    filter_data = {"severity": ["HIGH"], "status": "OPEN"}
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(filter_data, f)
        temp_filter_file = f.name
    
    try:
        cli_controller = CLIController()
        
        # Test parsing arguments with filters
        test_args = [
            "--steps", "finding",
            "--finding-filters", temp_filter_file,
            "--after-key", "12345",
            "--batch-size", "10",
            "--dry-run"
        ]
        
        config = cli_controller.parse_arguments(test_args)
        
        # Verify configuration
        assert config.steps == ["finding"]
        assert config.finding_filters_path == temp_filter_file
        assert config.after_key == 12345
        assert config.batch_size == 10
        assert config.dry_run == True
        
        print("✓ CLI parsing with filters works correctly")
        
        # Test configuration summary
        cli_controller.print_configuration_summary(config)
        
        print("✓ CLI integration tests passed!")
        
    except Exception as e:
        print(f"✗ CLI integration test failed: {e}")
        raise
    finally:
        os.unlink(temp_filter_file)


async def main():
    """Run all integration tests."""
    print("Running enhanced findings filtering integration tests...\n")
    
    try:
        await test_filter_manager_basic_functionality()
        await test_filter_validation_errors()
        await test_cli_integration()
        
        print("\n🎉 All integration tests passed successfully!")
        print("\nThe enhanced findings filtering functionality is working correctly:")
        print("- FilterManager can load and validate JSON filter files")
        print("- Filters can be combined with afterKey parameters")
        print("- Error handling works for invalid files and parameters")
        print("- CLI integration supports the new filtering options")
        
    except Exception as e:
        print(f"\n❌ Integration tests failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)