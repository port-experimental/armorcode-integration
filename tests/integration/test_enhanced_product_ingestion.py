#!/usr/bin/env python3
"""
Integration tests for enhanced product ingestion with bulk operations and retry logic.

Tests the enhanced ingest_products function with mocked APIs to verify:
- Bulk operations using BulkPortManager
- Retry logic integration with RetryManager
- Progress logging and statistics tracking
- Error handling for individual product failures
"""

import asyncio
import json
import logging
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any

# Import the function under test
from main import ingest_products
from bulk_port_manager import BulkPortManager, BulkResult
from retry_manager import RetryManager


class MockArmorCodeClient:
    """Mock ArmorCode client for testing."""
    
    def __init__(self, products: List[Dict[str, Any]] = None, should_fail: bool = False):
        self.products = products or []
        self.should_fail = should_fail
        
    async def get_all_products(self):
        """Mock get_all_products method."""
        if self.should_fail:
            raise Exception("ArmorCode API error")
        return self.products


@pytest.fixture
def sample_products():
    """Sample product data for testing."""
    return [
        {
            "id": 1,
            "name": "Product A",
            "description": "Description A",
            "businessOwnerName": "Owner A",
            "securityOwnerName": "Security A"
        },
        {
            "id": 2,
            "name": "Product B",
            "description": "Description B",
            "businessOwnerName": "Owner B",
            "securityOwnerName": "Security B"
        },
        {
            "id": 3,
            "name": "Product C",
            "description": None,  # Test null handling
            "businessOwnerName": None,
            "securityOwnerName": "Security C"
        }
    ]


@pytest.fixture
def mock_bulk_manager():
    """Mock BulkPortManager for testing."""
    manager = AsyncMock(spec=BulkPortManager)
    manager.__aenter__ = AsyncMock(return_value=manager)
    manager.__aexit__ = AsyncMock(return_value=None)
    return manager


@pytest.fixture
def mock_retry_manager():
    """Mock RetryManager for testing."""
    manager = MagicMock(spec=RetryManager)
    # Make with_retry pass through the function call
    async def mock_with_retry(func, *args, **kwargs):
        return await func(*args, **kwargs)
    manager.with_retry = mock_with_retry
    return manager


@pytest.mark.asyncio
async def test_successful_product_ingestion(sample_products, mock_bulk_manager, mock_retry_manager, caplog):
    """Test successful product ingestion with bulk operations."""
    # Setup
    ac_client = MockArmorCodeClient(sample_products)
    port_token = "test-token"
    
    # Mock successful bulk result
    bulk_result = BulkResult(
        successful_entities=["1", "2", "3"],
        failed_entities=[],
        total_processed=3
    )
    mock_bulk_manager.create_entities_bulk.return_value = bulk_result
    
    # Execute
    with caplog.at_level(logging.INFO):
        await ingest_products(
            ac_client, 
            port_token, 
            dry_run=False,
            bulk_manager=mock_bulk_manager,
            retry_manager=mock_retry_manager,
            batch_size=20
        )
    
    # Verify ArmorCode API was called
    assert "Found 3 products in Armorcode" in caplog.text
    
    # Verify bulk manager was used
    mock_bulk_manager.create_entities_bulk.assert_called_once()
    call_args = mock_bulk_manager.create_entities_bulk.call_args
    assert call_args[0][0] == "armorcodeProduct"  # blueprint_id
    assert len(call_args[0][1]) == 3  # entities
    assert call_args[0][2] == 20  # batch_size
    
    # Verify entities were transformed correctly
    entities = call_args[0][1]
    assert entities[0]["identifier"] == "1"
    assert entities[0]["title"] == "Product A"
    assert entities[0]["properties"]["name"] == "Product A"
    assert entities[0]["properties"]["description"] == "Description A"
    
    # Verify statistics logging
    assert "Total processed: 3" in caplog.text
    assert "Successful: 3" in caplog.text
    assert "Failed: 0" in caplog.text
    assert "Success rate: 100.0%" in caplog.text


@pytest.mark.asyncio
async def test_partial_failure_handling(sample_products, mock_bulk_manager, mock_retry_manager, caplog):
    """Test handling of partial failures in bulk operations."""
    # Setup
    ac_client = MockArmorCodeClient(sample_products)
    port_token = "test-token"
    
    # Mock partial failure result
    bulk_result = BulkResult(
        successful_entities=["1", "3"],
        failed_entities=[("2", "Validation error")],
        total_processed=3
    )
    mock_bulk_manager.create_entities_bulk.return_value = bulk_result
    
    # Execute
    with caplog.at_level(logging.INFO):
        await ingest_products(
            ac_client, 
            port_token, 
            dry_run=False,
            bulk_manager=mock_bulk_manager,
            retry_manager=mock_retry_manager
        )
    
    # Verify statistics logging
    assert "Total processed: 3" in caplog.text
    assert "Successful: 2" in caplog.text
    assert "Failed: 1" in caplog.text
    assert "Success rate: 66.7%" in caplog.text
    
    # Verify failed entities are logged
    assert "Failed product entities:" in caplog.text
    assert "2: Validation error" in caplog.text


@pytest.mark.asyncio
async def test_armorcode_api_failure_with_retry(mock_bulk_manager, caplog):
    """Test retry logic when ArmorCode API fails."""
    # Setup
    ac_client = MockArmorCodeClient(should_fail=True)
    port_token = "test-token"
    
    # Mock retry manager that fails after retries
    retry_manager = MagicMock(spec=RetryManager)
    async def mock_with_retry_fail(func, *args, **kwargs):
        raise Exception("ArmorCode API error after retries")
    retry_manager.with_retry = mock_with_retry_fail
    
    # Execute
    with caplog.at_level(logging.ERROR):
        await ingest_products(
            ac_client, 
            port_token, 
            dry_run=False,
            bulk_manager=mock_bulk_manager,
            retry_manager=retry_manager
        )
    
    # Verify error handling
    assert "Failed to fetch products from ArmorCode after retries" in caplog.text
    
    # Verify bulk manager was not called
    mock_bulk_manager.create_entities_bulk.assert_not_called()


@pytest.mark.asyncio
async def test_transformation_error_handling(mock_bulk_manager, mock_retry_manager, caplog):
    """Test handling of product transformation errors."""
    # Setup - products with invalid data
    invalid_products = [
        {"id": 1, "name": "Valid Product"},
        {"invalid": "missing id field"},  # This will cause transformation error
        {"id": 3, "name": "Another Valid Product"}
    ]
    ac_client = MockArmorCodeClient(invalid_products)
    port_token = "test-token"
    
    # Mock successful bulk result for valid entities
    bulk_result = BulkResult(
        successful_entities=["1", "3"],
        failed_entities=[],
        total_processed=2
    )
    mock_bulk_manager.create_entities_bulk.return_value = bulk_result
    
    # Execute
    with caplog.at_level(logging.WARNING):  # Changed to WARNING to catch the summary message
        await ingest_products(
            ac_client, 
            port_token, 
            dry_run=False,
            bulk_manager=mock_bulk_manager,
            retry_manager=mock_retry_manager
        )
    
    # Verify transformation error was logged
    assert "Failed to transform product" in caplog.text
    assert "Failed to transform 1 products" in caplog.text
    assert "continuing with 2 valid entities" in caplog.text
    
    # Verify only valid entities were sent to bulk manager
    mock_bulk_manager.create_entities_bulk.assert_called_once()
    call_args = mock_bulk_manager.create_entities_bulk.call_args
    entities = call_args[0][1]
    assert len(entities) == 2
    assert entities[0]["identifier"] == "1"
    assert entities[1]["identifier"] == "3"


@pytest.mark.asyncio
async def test_bulk_operation_failure_with_retry(sample_products, mock_bulk_manager, caplog):
    """Test retry logic when bulk operations fail."""
    # Setup
    ac_client = MockArmorCodeClient(sample_products)
    port_token = "test-token"
    
    # Mock retry manager that fails bulk operations after retries
    retry_manager = MagicMock(spec=RetryManager)
    call_count = 0
    async def mock_with_retry(func, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        if func == ac_client.get_all_products:
            return await func(*args, **kwargs)
        else:  # bulk operation
            raise Exception("Bulk operation failed after retries")
    retry_manager.with_retry = mock_with_retry
    
    # Execute
    with caplog.at_level(logging.INFO):  # Changed to INFO to capture the finished message
        await ingest_products(
            ac_client, 
            port_token, 
            dry_run=False,
            bulk_manager=mock_bulk_manager,
            retry_manager=retry_manager
        )
    
    # Verify error handling
    assert "Bulk product ingestion failed after retries" in caplog.text
    
    # Verify the function doesn't crash and continues gracefully
    assert "Product ingestion finished" in caplog.text


@pytest.mark.asyncio
async def test_dry_run_mode(sample_products, mock_bulk_manager, mock_retry_manager, caplog):
    """Test dry run mode behavior."""
    # Setup
    ac_client = MockArmorCodeClient(sample_products)
    port_token = "test-token"
    
    # Execute
    with caplog.at_level(logging.INFO):
        await ingest_products(
            ac_client, 
            port_token, 
            dry_run=True,
            bulk_manager=mock_bulk_manager,
            retry_manager=mock_retry_manager,
            batch_size=10
        )
    
    # Verify dry run behavior
    assert "[DRY RUN] Would process 3 products in batches of 10" in caplog.text
    
    # Verify no actual operations were performed
    mock_bulk_manager.create_entities_bulk.assert_not_called()


@pytest.mark.asyncio
async def test_empty_products_list(mock_bulk_manager, mock_retry_manager, caplog):
    """Test handling of empty products list."""
    # Setup
    ac_client = MockArmorCodeClient([])  # Empty products list
    port_token = "test-token"
    
    # Execute
    with caplog.at_level(logging.INFO):
        await ingest_products(
            ac_client, 
            port_token, 
            dry_run=False,
            bulk_manager=mock_bulk_manager,
            retry_manager=mock_retry_manager
        )
    
    # Verify handling
    assert "No products found, skipping product ingestion" in caplog.text
    
    # Verify no bulk operations were attempted
    mock_bulk_manager.create_entities_bulk.assert_not_called()


@pytest.mark.asyncio
async def test_default_managers_creation(sample_products, caplog):
    """Test that default managers are created when not provided."""
    # Setup
    ac_client = MockArmorCodeClient(sample_products)
    port_token = "test-token"
    
    # Mock the manager classes to track instantiation
    with patch('bulk_port_manager.BulkPortManager') as mock_bulk_class, \
         patch('retry_manager.RetryManager') as mock_retry_class:
        
        # Setup mocks
        mock_bulk_instance = AsyncMock()
        mock_bulk_instance.__aenter__ = AsyncMock(return_value=mock_bulk_instance)
        mock_bulk_instance.__aexit__ = AsyncMock(return_value=None)
        mock_bulk_instance.create_entities_bulk.return_value = BulkResult(
            successful_entities=["1", "2", "3"],
            failed_entities=[],
            total_processed=3
        )
        mock_bulk_class.return_value = mock_bulk_instance
        
        mock_retry_instance = MagicMock()
        async def mock_with_retry(func, *args, **kwargs):
            return await func(*args, **kwargs)
        mock_retry_instance.with_retry = mock_with_retry
        mock_retry_class.return_value = mock_retry_instance
        
        # Execute without providing managers
        await ingest_products(ac_client, port_token, dry_run=False)
        
        # Verify managers were created with correct parameters
        mock_bulk_class.assert_called_once_with(port_token)
        mock_retry_class.assert_called_once_with(max_attempts=3)


@pytest.mark.asyncio
async def test_custom_batch_size(sample_products, mock_bulk_manager, mock_retry_manager):
    """Test custom batch size parameter."""
    # Setup
    ac_client = MockArmorCodeClient(sample_products)
    port_token = "test-token"
    
    bulk_result = BulkResult(
        successful_entities=["1", "2", "3"],
        failed_entities=[],
        total_processed=3
    )
    mock_bulk_manager.create_entities_bulk.return_value = bulk_result
    
    # Execute with custom batch size
    await ingest_products(
        ac_client, 
        port_token, 
        dry_run=False,
        bulk_manager=mock_bulk_manager,
        retry_manager=mock_retry_manager,
        batch_size=5
    )
    
    # Verify custom batch size was passed
    call_args = mock_bulk_manager.create_entities_bulk.call_args
    assert call_args[0][2] == 5  # batch_size parameter


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])