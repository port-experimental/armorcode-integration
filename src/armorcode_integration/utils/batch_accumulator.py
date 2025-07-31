"""
Batch Accumulator for thread-safe concurrent entity processing.

This module provides the BatchAccumulator class for accumulating entities
in batches and automatically submitting them when batch size limits are reached,
with full thread safety for concurrent processing scenarios.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..managers.bulk_port_manager import BulkPortManager, BulkResult


@dataclass
class BatchStats:
    """Statistics for batch accumulator operations."""
    total_entities: int = 0
    batches_submitted: int = 0
    successful_entities: int = 0
    failed_entities: int = 0
    start_time: float = field(default_factory=time.time)
    
    @property
    def average_batch_size(self) -> float:
        """Calculate average batch size."""
        if self.batches_submitted == 0:
            return 0.0
        return self.total_entities / self.batches_submitted
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_entities == 0:
            return 0.0
        return (self.successful_entities / self.total_entities) * 100
    
    @property
    def processing_time(self) -> float:
        """Get total processing time in seconds."""
        return time.time() - self.start_time


class BatchAccumulator:
    """
    Thread-safe batch accumulator for concurrent entity processing.
    
    Accumulates entities and automatically submits batches to Port when
    the batch size limit is reached. Provides thread safety for concurrent
    access from multiple async tasks.
    """

    def __init__(self, bulk_manager: BulkPortManager, blueprint_id: str, batch_size: int = 20):
        """
        Initialize the BatchAccumulator.
        
        Args:
            bulk_manager: BulkPortManager instance for submitting batches
            blueprint_id: Port blueprint identifier for entities
            batch_size: Maximum batch size before auto-submission (default: 20)
        """
        self.bulk_manager = bulk_manager
        self.blueprint_id = blueprint_id
        self.batch_size = min(batch_size, 20)  # Enforce Port API limit
        
        # Thread-safe state management
        self._lock = asyncio.Lock()
        self._current_batch: List[Dict[str, Any]] = []
        self._stats = BatchStats()
        self._is_closed = False
        
        self.logger = logging.getLogger(__name__)
        
        if batch_size > 20:
            self.logger.warning(f"Batch size {batch_size} exceeds Port API limit, using 20")

    async def add_entity(self, entity: Dict[str, Any]) -> Optional[BulkResult]:
        """
        Add an entity to the current batch.
        
        If the batch reaches the configured size limit, it will be automatically
        submitted to Port. This method is thread-safe for concurrent access.
        
        Args:
            entity: Entity dictionary to add to the batch
            
        Returns:
            BulkResult if a batch was submitted, None otherwise
            
        Raises:
            RuntimeError: If the accumulator has been closed
        """
        async with self._lock:
            if self._is_closed:
                raise RuntimeError("BatchAccumulator has been closed")
            
            # Add entity to current batch
            self._current_batch.append(entity)
            self._stats.total_entities += 1
            
            self.logger.debug(f"Added entity to batch ({len(self._current_batch)}/{self.batch_size})")
            
            # Check if batch is full and needs submission
            if len(self._current_batch) >= self.batch_size:
                return await self._submit_current_batch()
            
            return None

    async def flush_remaining(self) -> Optional[BulkResult]:
        """
        Submit any remaining entities in the current batch.
        
        This method should be called when all entities have been added
        to ensure any partial batch is submitted. After calling this method,
        the accumulator is closed and no more entities can be added.
        
        Returns:
            BulkResult if there were remaining entities to submit, None otherwise
        """
        async with self._lock:
            self._is_closed = True
            
            if not self._current_batch:
                self.logger.debug("No remaining entities to flush")
                return None
            
            self.logger.info(f"Flushing remaining {len(self._current_batch)} entities")
            return await self._submit_current_batch()

    async def _submit_current_batch(self) -> BulkResult:
        """
        Submit the current batch to Port (internal method, assumes lock is held).
        
        Returns:
            BulkResult from the batch submission
        """
        if not self._current_batch:
            return BulkResult(successful_entities=[], failed_entities=[], total_processed=0)
        
        batch_to_submit = self._current_batch.copy()
        batch_size = len(batch_to_submit)
        
        self.logger.info(f"Submitting batch of {batch_size} entities to blueprint '{self.blueprint_id}'")
        
        try:
            # Submit batch using bulk manager
            result = await self.bulk_manager._submit_batch(self.blueprint_id, batch_to_submit)
            
            # Update statistics
            self._stats.batches_submitted += 1
            self._stats.successful_entities += result.success_count
            self._stats.failed_entities += result.failure_count
            
            # Clear current batch
            self._current_batch.clear()
            
            self.logger.info(f"Batch submitted successfully: {result.success_count} successful, "
                           f"{result.failure_count} failed")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to submit batch: {str(e)}")
            
            # Create failed result for all entities in batch
            failed_entities = []
            for i, entity in enumerate(batch_to_submit):
                entity_id = entity.get("identifier", f"entity_{i}")
                failed_entities.append((entity_id, str(e)))
            
            # Update statistics for failed batch
            self._stats.batches_submitted += 1
            self._stats.failed_entities += len(batch_to_submit)
            
            # Clear current batch even on failure
            self._current_batch.clear()
            
            return BulkResult(
                successful_entities=[],
                failed_entities=failed_entities,
                total_processed=len(batch_to_submit)
            )

    def get_stats(self) -> BatchStats:
        """
        Get current accumulator statistics.
        
        This method is thread-safe and returns a copy of the current statistics.
        
        Returns:
            BatchStats with current accumulator statistics
        """
        # Create a copy to avoid race conditions
        return BatchStats(
            total_entities=self._stats.total_entities,
            batches_submitted=self._stats.batches_submitted,
            successful_entities=self._stats.successful_entities,
            failed_entities=self._stats.failed_entities,
            start_time=self._stats.start_time
        )

    @property
    def is_closed(self) -> bool:
        """Check if the accumulator has been closed."""
        return self._is_closed

    @property
    def current_batch_size(self) -> int:
        """Get the current batch size (thread-safe read)."""
        return len(self._current_batch)

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - automatically flush remaining entities."""
        if not self._is_closed:
            await self.flush_remaining()