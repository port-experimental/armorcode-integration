"""
Unit tests for BatchAccumulator class.

Tests thread safety, automatic submission, statistics tracking,
and error handling scenarios.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

from batch_accumulator import BatchAccumulator, BatchStats
from bulk_port_manager import BulkPortManager, BulkResult


@pytest.fixture
def mock_bulk_manager():
    """Create a mock BulkPortManager for testing."""
    manager = MagicMock(spec=BulkPortManager)
    manager._submit_batch = AsyncMock()
    return manager


@pytest.fixture
def sample_entity():
    """Create a sample entity for testing."""
    return {
        "identifier": "test-entity-1",
        "title": "Test Entity",
        "properties": {"key": "value"}
    }


@pytest.fixture
def sample_entities():
    """Create multiple sample entities for testing."""
    return [
        {
            "identifier": f"test-entity-{i}",
            "title": f"Test Entity {i}",
            "properties": {"key": f"value-{i}"}
        }
        for i in range(1, 6)
    ]


class TestBatchAccumulator:
    """Test cases for BatchAccumulator class."""

    @pytest.mark.asyncio
    async def test_initialization(self, mock_bulk_manager):
        """Test BatchAccumulator initialization."""
        accumulator = BatchAccumulator(mock_bulk_manager, "test-blueprint", batch_size=10)
        
        assert accumulator.bulk_manager == mock_bulk_manager
        assert accumulator.blueprint_id == "test-blueprint"
        assert accumulator.batch_size == 10
        assert accumulator.current_batch_size == 0
        assert not accumulator.is_closed
        
        stats = accumulator.get_stats()
        assert stats.total_entities == 0
        assert stats.batches_submitted == 0
        assert stats.successful_entities == 0
        assert stats.failed_entities == 0

    @pytest.mark.asyncio
    async def test_batch_size_limit_enforcement(self, mock_bulk_manager):
        """Test that batch size is limited to Port API maximum of 20."""
        accumulator = BatchAccumulator(mock_bulk_manager, "test-blueprint", batch_size=25)
        
        assert accumulator.batch_size == 20

    @pytest.mark.asyncio
    async def test_add_entity_basic(self, mock_bulk_manager, sample_entity):
        """Test basic entity addition without triggering batch submission."""
        accumulator = BatchAccumulator(mock_bulk_manager, "test-blueprint", batch_size=5)
        
        result = await accumulator.add_entity(sample_entity)
        
        assert result is None  # No batch submitted yet
        assert accumulator.current_batch_size == 1
        
        stats = accumulator.get_stats()
        assert stats.total_entities == 1
        assert stats.batches_submitted == 0

    @pytest.mark.asyncio
    async def test_automatic_batch_submission(self, mock_bulk_manager, sample_entities):
        """Test automatic batch submission when batch size limit is reached."""
        # Mock successful batch submission
        mock_bulk_manager._submit_batch.return_value = BulkResult(
            successful_entities=["test-entity-1", "test-entity-2"],
            failed_entities=[],
            total_processed=2
        )
        
        accumulator = BatchAccumulator(mock_bulk_manager, "test-blueprint", batch_size=2)
        
        # Add first entity - should not trigger submission
        result1 = await accumulator.add_entity(sample_entities[0])
        assert result1 is None
        assert accumulator.current_batch_size == 1
        
        # Add second entity - should trigger submission
        result2 = await accumulator.add_entity(sample_entities[1])
        assert result2 is not None
        assert isinstance(result2, BulkResult)
        assert result2.success_count == 2
        assert accumulator.current_batch_size == 0  # Batch cleared after submission
        
        # Verify bulk manager was called
        mock_bulk_manager._submit_batch.assert_called_once_with(
            "test-blueprint", 
            [sample_entities[0], sample_entities[1]]
        )
        
        # Check statistics
        stats = accumulator.get_stats()
        assert stats.total_entities == 2
        assert stats.batches_submitted == 1
        assert stats.successful_entities == 2
        assert stats.failed_entities == 0

    @pytest.mark.asyncio
    async def test_flush_remaining_with_entities(self, mock_bulk_manager, sample_entities):
        """Test flushing remaining entities in partial batch."""
        # Mock successful batch submission
        mock_bulk_manager._submit_batch.return_value = BulkResult(
            successful_entities=["test-entity-1", "test-entity-2", "test-entity-3"],
            failed_entities=[],
            total_processed=3
        )
        
        accumulator = BatchAccumulator(mock_bulk_manager, "test-blueprint", batch_size=5)
        
        # Add 3 entities (less than batch size)
        for entity in sample_entities[:3]:
            result = await accumulator.add_entity(entity)
            assert result is None  # No automatic submission
        
        assert accumulator.current_batch_size == 3
        
        # Flush remaining entities
        result = await accumulator.flush_remaining()
        
        assert result is not None
        assert result.success_count == 3
        assert accumulator.current_batch_size == 0
        assert accumulator.is_closed
        
        # Verify bulk manager was called
        mock_bulk_manager._submit_batch.assert_called_once_with(
            "test-blueprint", 
            sample_entities[:3]
        )

    @pytest.mark.asyncio
    async def test_flush_remaining_empty_batch(self, mock_bulk_manager):
        """Test flushing when there are no remaining entities."""
        accumulator = BatchAccumulator(mock_bulk_manager, "test-blueprint", batch_size=5)
        
        result = await accumulator.flush_remaining()
        
        assert result is None
        assert accumulator.is_closed
        
        # Verify bulk manager was not called
        mock_bulk_manager._submit_batch.assert_not_called()

    @pytest.mark.asyncio
    async def test_add_entity_after_close_raises_error(self, mock_bulk_manager, sample_entity):
        """Test that adding entities after close raises RuntimeError."""
        accumulator = BatchAccumulator(mock_bulk_manager, "test-blueprint", batch_size=5)
        
        # Close the accumulator
        await accumulator.flush_remaining()
        assert accumulator.is_closed
        
        # Try to add entity after close
        with pytest.raises(RuntimeError, match="BatchAccumulator has been closed"):
            await accumulator.add_entity(sample_entity)

    @pytest.mark.asyncio
    async def test_statistics_tracking(self, mock_bulk_manager, sample_entities):
        """Test comprehensive statistics tracking."""
        # Mock mixed success/failure results
        mock_bulk_manager._submit_batch.side_effect = [
            BulkResult(
                successful_entities=["test-entity-1", "test-entity-2"],
                failed_entities=[],
                total_processed=2
            ),
            BulkResult(
                successful_entities=["test-entity-3"],
                failed_entities=[("test-entity-4", "Test error")],
                total_processed=2
            )
        ]
        
        accumulator = BatchAccumulator(mock_bulk_manager, "test-blueprint", batch_size=2)
        
        # Add entities to trigger two batch submissions
        for entity in sample_entities[:4]:
            await accumulator.add_entity(entity)
        
        stats = accumulator.get_stats()
        assert stats.total_entities == 4
        assert stats.batches_submitted == 2
        assert stats.successful_entities == 3
        assert stats.failed_entities == 1
        assert stats.average_batch_size == 2.0
        assert stats.success_rate == 75.0
        assert stats.processing_time > 0

    @pytest.mark.asyncio
    async def test_batch_submission_error_handling(self, mock_bulk_manager, sample_entities):
        """Test error handling during batch submission."""
        # Mock batch submission failure
        mock_bulk_manager._submit_batch.side_effect = Exception("Network error")
        
        accumulator = BatchAccumulator(mock_bulk_manager, "test-blueprint", batch_size=2)
        
        # Add entities to trigger batch submission
        await accumulator.add_entity(sample_entities[0])
        result = await accumulator.add_entity(sample_entities[1])
        
        # Should return failed result
        assert result is not None
        assert result.success_count == 0
        assert result.failure_count == 2
        assert len(result.failed_entities) == 2
        
        # Check that error is properly recorded
        entity_id, error_msg = result.failed_entities[0]
        assert "Network error" in error_msg
        
        # Batch should be cleared even on failure
        assert accumulator.current_batch_size == 0
        
        # Statistics should reflect the failure
        stats = accumulator.get_stats()
        assert stats.total_entities == 2
        assert stats.batches_submitted == 1
        assert stats.successful_entities == 0
        assert stats.failed_entities == 2

    @pytest.mark.asyncio
    async def test_thread_safety_concurrent_access(self, mock_bulk_manager):
        """Test thread safety with concurrent entity additions."""
        # Mock successful batch submissions that return results based on actual batch size
        def mock_submit_batch(blueprint_id, batch):
            return BulkResult(
                successful_entities=[entity["identifier"] for entity in batch],
                failed_entities=[],
                total_processed=len(batch)
            )
        
        mock_bulk_manager._submit_batch.side_effect = mock_submit_batch
        
        accumulator = BatchAccumulator(mock_bulk_manager, "test-blueprint", batch_size=2)
        
        # Create multiple concurrent tasks adding entities
        async def add_entities_task(start_id: int, count: int):
            for i in range(count):
                entity = {
                    "identifier": f"concurrent-entity-{start_id + i}",
                    "title": f"Concurrent Entity {start_id + i}"
                }
                await accumulator.add_entity(entity)
        
        # Run multiple concurrent tasks
        tasks = [
            add_entities_task(1, 5),
            add_entities_task(6, 5),
            add_entities_task(11, 5)
        ]
        
        await asyncio.gather(*tasks)
        
        # Flush any remaining entities
        await accumulator.flush_remaining()
        
        # Verify all entities were processed
        stats = accumulator.get_stats()
        assert stats.total_entities == 15
        assert stats.batches_submitted > 0
        
        # Verify no race conditions occurred (all entities accounted for)
        total_processed = stats.successful_entities + stats.failed_entities
        assert total_processed == 15

    @pytest.mark.asyncio
    async def test_context_manager_usage(self, mock_bulk_manager, sample_entities):
        """Test using BatchAccumulator as async context manager."""
        mock_bulk_manager._submit_batch.return_value = BulkResult(
            successful_entities=["test-entity-1", "test-entity-2"],
            failed_entities=[],
            total_processed=2
        )
        
        async with BatchAccumulator(mock_bulk_manager, "test-blueprint", batch_size=5) as accumulator:
            # Add some entities
            for entity in sample_entities[:2]:
                await accumulator.add_entity(entity)
            
            assert accumulator.current_batch_size == 2
            assert not accumulator.is_closed
        
        # After context exit, should be closed and remaining entities flushed
        assert accumulator.is_closed
        assert accumulator.current_batch_size == 0
        
        # Verify flush was called
        mock_bulk_manager._submit_batch.assert_called_once()

    @pytest.mark.asyncio
    async def test_partial_failure_handling(self, mock_bulk_manager, sample_entities):
        """Test handling of partial failures in batch submissions."""
        # Mock partial failure result
        mock_bulk_manager._submit_batch.return_value = BulkResult(
            successful_entities=["test-entity-1"],
            failed_entities=[("test-entity-2", "Validation error")],
            total_processed=2
        )
        
        accumulator = BatchAccumulator(mock_bulk_manager, "test-blueprint", batch_size=2)
        
        # Add entities to trigger batch submission
        await accumulator.add_entity(sample_entities[0])
        result = await accumulator.add_entity(sample_entities[1])
        
        assert result is not None
        assert result.success_count == 1
        assert result.failure_count == 1
        
        # Check statistics reflect partial failure
        stats = accumulator.get_stats()
        assert stats.successful_entities == 1
        assert stats.failed_entities == 1
        assert stats.success_rate == 50.0

    def test_batch_stats_properties(self):
        """Test BatchStats property calculations."""
        stats = BatchStats(
            total_entities=10,
            batches_submitted=3,
            successful_entities=8,
            failed_entities=2
        )
        
        assert stats.average_batch_size == 10 / 3
        assert stats.success_rate == 80.0
        assert stats.processing_time >= 0
        
        # Test edge cases
        empty_stats = BatchStats()
        assert empty_stats.average_batch_size == 0.0
        assert empty_stats.success_rate == 0.0


if __name__ == "__main__":
    pytest.main([__file__])