"""
Unit tests for BulkPortManager.

Tests bulk operations, batching logic, and failure handling.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

import pytest
import aiohttp
from aioresponses import aioresponses

from bulk_port_manager import BulkPortManager, BulkResult


class TestBulkResult:
    """Test BulkResult dataclass functionality."""
    
    def test_bulk_result_properties(self):
        """Test BulkResult property calculations."""
        result = BulkResult(
            successful_entities=["entity1", "entity2"],
            failed_entities=[("entity3", "error1"), ("entity4", "error2")],
            total_processed=4
        )
        
        assert result.success_count == 2
        assert result.failure_count == 2
        assert result.success_rate == 50.0
    
    def test_bulk_result_empty(self):
        """Test BulkResult with no entities."""
        result = BulkResult(
            successful_entities=[],
            failed_entities=[],
            total_processed=0
        )
        
        assert result.success_count == 0
        assert result.failure_count == 0
        assert result.success_rate == 0.0
    
    def test_bulk_result_all_successful(self):
        """Test BulkResult with all successful entities."""
        result = BulkResult(
            successful_entities=["entity1", "entity2", "entity3"],
            failed_entities=[],
            total_processed=3
        )
        
        assert result.success_count == 3
        assert result.failure_count == 0
        assert result.success_rate == 100.0


class TestBulkPortManager:
    """Test BulkPortManager functionality."""
    
    @pytest.fixture
    def manager(self):
        """Create a BulkPortManager instance for testing."""
        return BulkPortManager("test-token", "https://api.test.com/v1")
    
    @pytest.fixture
    def sample_entities(self):
        """Sample entities for testing."""
        return [
            {"identifier": "entity1", "title": "Entity 1", "properties": {"name": "Test 1"}},
            {"identifier": "entity2", "title": "Entity 2", "properties": {"name": "Test 2"}},
            {"identifier": "entity3", "title": "Entity 3", "properties": {"name": "Test 3"}},
        ]
    
    def test_init(self, manager):
        """Test BulkPortManager initialization."""
        assert manager.port_token == "test-token"
        assert manager.port_api_url == "https://api.test.com/v1"
        assert manager.session is None
    
    def test_get_headers(self, manager):
        """Test header generation."""
        headers = manager._get_headers()
        expected = {
            "Authorization": "Bearer test-token",
            "Content-Type": "application/json"
        }
        assert headers == expected
    
    def test_create_batches_normal(self, manager, sample_entities):
        """Test normal batch creation."""
        batches = manager._create_batches(sample_entities, batch_size=2)
        
        assert len(batches) == 2
        assert len(batches[0]) == 2
        assert len(batches[1]) == 1
        assert batches[0] == sample_entities[:2]
        assert batches[1] == sample_entities[2:]
    
    def test_create_batches_exact_fit(self, manager, sample_entities):
        """Test batch creation with exact fit."""
        batches = manager._create_batches(sample_entities, batch_size=3)
        
        assert len(batches) == 1
        assert len(batches[0]) == 3
        assert batches[0] == sample_entities
    
    def test_create_batches_oversized(self, manager, sample_entities):
        """Test batch creation with oversized batch_size."""
        with patch.object(manager.logger, 'warning') as mock_warning:
            batches = manager._create_batches(sample_entities, batch_size=25)
            
            mock_warning.assert_called_once_with(
                "Batch size 25 exceeds Port API limit of 20, using 20"
            )
            assert len(batches) == 1
            assert len(batches[0]) == 3
    
    def test_create_batches_empty(self, manager):
        """Test batch creation with empty entities list."""
        batches = manager._create_batches([], batch_size=10)
        assert batches == []
    
    def test_create_failed_result(self, manager, sample_entities):
        """Test creation of failed result."""
        error_msg = "Network error"
        result = manager._create_failed_result(sample_entities, error_msg)
        
        assert result.success_count == 0
        assert result.failure_count == 3
        assert result.total_processed == 3
        assert len(result.failed_entities) == 3
        
        for i, (entity_id, error) in enumerate(result.failed_entities):
            assert entity_id == f"entity{i+1}"
            assert error == error_msg
    
    def test_process_successful_response_with_entities(self, manager, sample_entities):
        """Test processing successful response with entity details."""
        response_data = {
            "entities": [
                {"success": True},
                {"success": False, "error": "Validation failed"},
                {"success": True}
            ]
        }
        
        result = manager._process_successful_response(sample_entities, response_data)
        
        assert result.success_count == 2
        assert result.failure_count == 1
        assert result.total_processed == 3
        assert "entity1" in result.successful_entities
        assert "entity3" in result.successful_entities
        assert ("entity2", "Validation failed") in result.failed_entities
    
    def test_process_successful_response_simple(self, manager, sample_entities):
        """Test processing simple successful response."""
        response_data = {"message": "Success"}
        
        result = manager._process_successful_response(sample_entities, response_data)
        
        assert result.success_count == 3
        assert result.failure_count == 0
        assert result.total_processed == 3
        assert all(f"entity{i+1}" in result.successful_entities for i in range(3))
    
    @pytest.mark.asyncio
    async def test_context_manager(self, manager):
        """Test async context manager functionality."""
        async with manager as mgr:
            assert mgr.session is not None
            assert isinstance(mgr.session, aiohttp.ClientSession)
        
        # Session should be closed after exiting context
        assert mgr.session.closed
    
    @pytest.mark.asyncio
    async def test_submit_batch_success(self, manager, sample_entities):
        """Test successful batch submission."""
        with aioresponses() as m:
            m.post(
                "https://api.test.com/v1/blueprints/test-blueprint/entities/bulk",
                payload={"entities": [{"success": True}, {"success": True}, {"success": True}]},
                status=200
            )
            
            async with manager:
                result = await manager._submit_batch("test-blueprint", sample_entities)
            
            assert result.success_count == 3
            assert result.failure_count == 0
            assert result.total_processed == 3
    
    @pytest.mark.asyncio
    async def test_submit_batch_partial_failure(self, manager, sample_entities):
        """Test batch submission with partial failures."""
        with aioresponses() as m:
            m.post(
                "https://api.test.com/v1/blueprints/test-blueprint/entities/bulk",
                payload={
                    "entities": [
                        {"success": True},
                        {"success": False, "error": "Invalid data"},
                        {"success": True}
                    ]
                },
                status=200
            )
            
            async with manager:
                result = await manager._submit_batch("test-blueprint", sample_entities)
            
            assert result.success_count == 2
            assert result.failure_count == 1
            assert result.total_processed == 3
            assert ("entity2", "Invalid data") in result.failed_entities
    
    @pytest.mark.asyncio
    async def test_submit_batch_http_error(self, manager, sample_entities):
        """Test batch submission with HTTP error."""
        with aioresponses() as m:
            m.post(
                "https://api.test.com/v1/blueprints/test-blueprint/entities/bulk",
                payload={"message": "Authentication failed"},
                status=401
            )
            
            async with manager:
                result = await manager._submit_batch("test-blueprint", sample_entities)
            
            assert result.success_count == 0
            assert result.failure_count == 3
            assert result.total_processed == 3
            assert all("Authentication failed" in error for _, error in result.failed_entities)
    
    @pytest.mark.asyncio
    async def test_submit_batch_network_error(self, manager, sample_entities):
        """Test batch submission with network error."""
        with aioresponses() as m:
            m.post(
                "https://api.test.com/v1/blueprints/test-blueprint/entities/bulk",
                exception=aiohttp.ClientError("Connection failed")
            )
            
            async with manager:
                result = await manager._submit_batch("test-blueprint", sample_entities)
            
            assert result.success_count == 0
            assert result.failure_count == 3
            assert result.total_processed == 3
            assert all("Network error" in error for _, error in result.failed_entities)
    
    @pytest.mark.asyncio
    async def test_submit_batch_without_context_manager(self, manager, sample_entities):
        """Test that submit_batch fails without context manager."""
        with pytest.raises(RuntimeError, match="must be used as async context manager"):
            await manager._submit_batch("test-blueprint", sample_entities)
    
    @pytest.mark.asyncio
    async def test_create_entities_bulk_empty(self, manager):
        """Test bulk creation with empty entities list."""
        async with manager:
            result = await manager.create_entities_bulk("test-blueprint", [])
        
        assert result.success_count == 0
        assert result.failure_count == 0
        assert result.total_processed == 0
    
    @pytest.mark.asyncio
    async def test_create_entities_bulk_single_batch(self, manager, sample_entities):
        """Test bulk creation with single batch."""
        with aioresponses() as m:
            m.post(
                "https://api.test.com/v1/blueprints/test-blueprint/entities/bulk",
                payload={"entities": [{"success": True}, {"success": True}, {"success": True}]},
                status=200
            )
            
            async with manager:
                result = await manager.create_entities_bulk("test-blueprint", sample_entities)
            
            assert result.success_count == 3
            assert result.failure_count == 0
            assert result.total_processed == 3
    
    @pytest.mark.asyncio
    async def test_create_entities_bulk_multiple_batches(self, manager):
        """Test bulk creation with multiple batches."""
        # Create 5 entities to force multiple batches with batch_size=2
        entities = [
            {"identifier": f"entity{i}", "title": f"Entity {i}"}
            for i in range(1, 6)
        ]
        
        with aioresponses() as m:
            # Mock 3 batch requests (2, 2, 1 entities)
            for i in range(3):
                m.post(
                    "https://api.test.com/v1/blueprints/test-blueprint/entities/bulk",
                    payload={"message": "Success"},
                    status=200
                )
            
            async with manager:
                result = await manager.create_entities_bulk("test-blueprint", entities, batch_size=2)
            
            assert result.success_count == 5
            assert result.failure_count == 0
            assert result.total_processed == 5
    
    @pytest.mark.asyncio
    async def test_create_entities_bulk_mixed_results(self, manager):
        """Test bulk creation with mixed success/failure across batches."""
        entities = [
            {"identifier": f"entity{i}", "title": f"Entity {i}"}
            for i in range(1, 5)
        ]
        
        with aioresponses() as m:
            # First batch: partial success
            m.post(
                "https://api.test.com/v1/blueprints/test-blueprint/entities/bulk",
                payload={
                    "entities": [
                        {"success": True},
                        {"success": False, "error": "Validation error"}
                    ]
                },
                status=200
            )
            # Second batch: complete success
            m.post(
                "https://api.test.com/v1/blueprints/test-blueprint/entities/bulk",
                payload={"message": "Success"},
                status=200
            )
            
            async with manager:
                result = await manager.create_entities_bulk("test-blueprint", entities, batch_size=2)
            
            assert result.success_count == 3
            assert result.failure_count == 1
            assert result.total_processed == 4
            assert ("entity2", "Validation error") in result.failed_entities
    
    @pytest.mark.asyncio
    async def test_create_entities_bulk_batch_exception(self, manager):
        """Test bulk creation with batch-level exception."""
        entities = [
            {"identifier": f"entity{i}", "title": f"Entity {i}"}
            for i in range(1, 4)
        ]
        
        with aioresponses() as m:
            # First batch succeeds
            m.post(
                "https://api.test.com/v1/blueprints/test-blueprint/entities/bulk",
                payload={"message": "Success"},
                status=200
            )
            # Second batch raises exception
            m.post(
                "https://api.test.com/v1/blueprints/test-blueprint/entities/bulk",
                exception=Exception("Unexpected error")
            )
            
            async with manager:
                result = await manager.create_entities_bulk("test-blueprint", entities, batch_size=2)
            
            assert result.success_count == 2  # First batch succeeded
            assert result.failure_count == 1  # Second batch failed
            assert result.total_processed == 3
            assert any("Unexpected error" in error for _, error in result.failed_entities)

    @pytest.mark.asyncio
    async def test_create_batch_accumulator(self):
        """Test creating a BatchAccumulator instance."""
        manager = BulkPortManager("test-token")
        
        accumulator = manager.create_batch_accumulator("test-blueprint", batch_size=10)
        
        # Import here to avoid circular import issues in tests
        from batch_accumulator import BatchAccumulator
        
        assert isinstance(accumulator, BatchAccumulator)
        assert accumulator.bulk_manager == manager
        assert accumulator.blueprint_id == "test-blueprint"
        assert accumulator.batch_size == 10


if __name__ == "__main__":
    pytest.main([__file__])