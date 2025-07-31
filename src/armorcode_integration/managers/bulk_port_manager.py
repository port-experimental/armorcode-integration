"""
Bulk Port Manager for batch entity operations.

This module provides the BulkPortManager class for efficiently creating entities
in Port using the bulk API endpoint, with support for partial failure handling
and batch processing.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import aiohttp
from .logging_manager import get_logging_manager, OperationType

if TYPE_CHECKING:
    from ..utils.batch_accumulator import BatchAccumulator
    from .retry_manager import RetryManager
    from .retry_manager import RetryManager


@dataclass
class BulkResult:
    """Result of a bulk entity operation."""
    successful_entities: List[str]
    failed_entities: List[Tuple[str, str]]  # (entity_id, error_message)
    total_processed: int

    @property
    def success_count(self) -> int:
        """Number of successfully processed entities."""
        return len(self.successful_entities)

    @property
    def failure_count(self) -> int:
        """Number of failed entities."""
        return len(self.failed_entities)

    @property
    def success_rate(self) -> float:
        """Success rate as a percentage."""
        if self.total_processed == 0:
            return 0.0
        return (self.success_count / self.total_processed) * 100


class BulkPortManager:
    """
    Manager for Port bulk entity operations.
    
    Handles batching entities and submitting them to Port's bulk API endpoint
    with support for partial failure handling and retry logic.
    """

    def __init__(self, port_token: str, port_api_url: str = "https://api.us.getport.io/v1", retry_manager: Optional['RetryManager'] = None):
        """
        Initialize the BulkPortManager.
        
        Args:
            port_token: Port API authentication token
            port_api_url: Base URL for Port API
            retry_manager: RetryManager instance for handling retries
            retry_manager: RetryManager instance for handling retries
        """
        self.port_token = port_token
        self.port_api_url = port_api_url
        self.retry_manager = retry_manager
        self.session: Optional[aiohttp.ClientSession] = None
        self.logging_manager = get_logging_manager()
        self.logger = self.logging_manager.get_logger("BulkPortManager")

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for Port API requests."""
        return {
            "Authorization": f"Bearer {self.port_token}",
            "Content-Type": "application/json"
        }

    def _create_batches(self, entities: List[Dict[str, Any]], batch_size: int = 20) -> List[List[Dict[str, Any]]]:
        """
        Split entities into batches of specified size.
        
        Args:
            entities: List of entities to batch
            batch_size: Maximum size per batch (default: 20, Port API limit)
            
        Returns:
            List of entity batches
        """
        if batch_size > 20:
            self.logger.warning(f"Batch size {batch_size} exceeds Port API limit of 20, using 20")
            batch_size = 20
        
        batches = []
        for i in range(0, len(entities), batch_size):
            batch = entities[i:i + batch_size]
            batches.append(batch)
        
        return batches

    async def _submit_batch(self, blueprint_id: str, batch: List[Dict[str, Any]]) -> BulkResult:
        """
        Submit a single batch to Port's bulk API.
        
        Args:
            blueprint_id: Port blueprint identifier
            batch: List of entities to create
            
        Returns:
            BulkResult with operation results
        """
        if not self.session:
            raise RuntimeError("BulkPortManager must be used as async context manager")

        url = f"{self.port_api_url}/blueprints/{blueprint_id}/entities/bulk?upsert=true&merge=true"
        payload = {"entities": batch}
        
        self.logger.debug(f"Submitting batch of {len(batch)} entities to {blueprint_id}")
        
        try:
            async with self.session.post(url, json=payload, headers=self._get_headers()) as response:
                response_data = await response.json()
                
                if response.status == 200:
                    return self._process_successful_response(batch, response_data)
                else:
                    # Handle error response
                    error_msg = response_data.get("message", f"HTTP {response.status}")
                    self.logger.error(f"Bulk operation failed: {error_msg}")
                    return self._create_failed_result(batch, error_msg)
                    
        except aiohttp.ClientError as e:
            error_msg = f"Network error: {str(e)}"
            self.logger.error(f"Failed to submit batch: {error_msg}")
            return self._create_failed_result(batch, error_msg)
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            self.logger.error(f"Failed to submit batch: {error_msg}")
            return self._create_failed_result(batch, error_msg)

    def _process_successful_response(self, batch: List[Dict[str, Any]], response_data: Dict[str, Any]) -> BulkResult:
        """
        Process a successful bulk API response, handling partial failures.
        
        Args:
            batch: Original batch of entities
            response_data: Response from Port API
            
        Returns:
            BulkResult with success/failure details
        """
        successful_entities = []
        failed_entities = []
        
        # Port bulk API response format may vary, handle different response structures
        if "entities" in response_data:
            # Response contains detailed entity results
            entity_results = response_data["entities"]
            for i, entity in enumerate(batch):
                entity_id = entity.get("identifier", f"entity_{i}")
                if i < len(entity_results):
                    result = entity_results[i]
                    if result.get("success", True):  # Assume success if not specified
                        successful_entities.append(entity_id)
                    else:
                        error_msg = result.get("error", "Unknown error")
                        failed_entities.append((entity_id, error_msg))
                        self.logger.warning(f"Entity {entity_id} failed: {error_msg}")
                else:
                    # Missing result, assume success
                    successful_entities.append(entity_id)
        else:
            # Simple success response, assume all entities succeeded
            for i, entity in enumerate(batch):
                entity_id = entity.get("identifier", f"entity_{i}")
                successful_entities.append(entity_id)
        
        return BulkResult(
            successful_entities=successful_entities,
            failed_entities=failed_entities,
            total_processed=len(batch)
        )

    def _create_failed_result(self, batch: List[Dict[str, Any]], error_message: str) -> BulkResult:
        """
        Create a BulkResult for a completely failed batch.
        
        Args:
            batch: The failed batch
            error_message: Error message
            
        Returns:
            BulkResult with all entities marked as failed
        """
        failed_entities = []
        for i, entity in enumerate(batch):
            entity_id = entity.get("identifier", f"entity_{i}")
            failed_entities.append((entity_id, error_message))
        
        return BulkResult(
            successful_entities=[],
            failed_entities=failed_entities,
            total_processed=len(batch)
        )

    async def create_entities_bulk(self, blueprint_id: str, entities: List[Dict[str, Any]], 
                                 batch_size: int = 20) -> BulkResult:
        """
        Create entities in bulk using Port's bulk API.
        
        Args:
            blueprint_id: Port blueprint identifier
            entities: List of entities to create
            batch_size: Maximum batch size (default: 20)
            
        Returns:
            Aggregated BulkResult for all batches
        """
        if not entities:
            return BulkResult(successful_entities=[], failed_entities=[], total_processed=0)
        
        self.logger.info(f"Creating {len(entities)} entities in blueprint '{blueprint_id}' using bulk API")
        
        # Create batches
        batches = self._create_batches(entities, batch_size)
        self.logger.info(f"Split into {len(batches)} batches of max size {batch_size}")
        
        # Process batches
        all_successful = []
        all_failed = []
        total_processed = 0
        
        for i, batch in enumerate(batches, 1):
            self.logger.debug(f"Processing batch {i}/{len(batches)} ({len(batch)} entities)")
            
            try:
                if self.retry_manager:
                    # Use retry for individual batch submissions
                    result = await self.retry_manager.with_retry(
                        self._submit_batch, blueprint_id, batch
                    )
                else:
                    # Fallback to direct call if no retry manager
                    result = await self._submit_batch(blueprint_id, batch)
                all_successful.extend(result.successful_entities)
                all_failed.extend(result.failed_entities)
                total_processed += result.total_processed
                
                self.logger.info(f"Batch {i}/{len(batches)} completed: "
                               f"{result.success_count} successful, {result.failure_count} failed")
                
            except Exception as e:
                # Handle batch-level failures
                error_msg = f"Batch processing error: {str(e)}"
                self.logger.error(f"Failed to process batch {i}: {error_msg}")
                
                # Mark all entities in this batch as failed
                for entity in batch:
                    entity_id = entity.get("identifier", f"entity_{len(all_failed)}")
                    all_failed.append((entity_id, error_msg))
                total_processed += len(batch)
        
        # Create aggregated result
        final_result = BulkResult(
            successful_entities=all_successful,
            failed_entities=all_failed,
            total_processed=total_processed
        )
        
        self.logger.info(f"Bulk operation completed: {final_result.success_count} successful, "
                        f"{final_result.failure_count} failed ({final_result.success_rate:.1f}% success rate)")
        
        return final_result

    def create_batch_accumulator(self, blueprint_id: str, batch_size: int = 20) -> 'BatchAccumulator':
        """
        Create a BatchAccumulator for this manager.
        
        Args:
            blueprint_id: Port blueprint identifier
            batch_size: Maximum batch size (default: 20)
            
        Returns:
            BatchAccumulator instance configured for this manager
        """
        # Import here to avoid circular imports at runtime
        from ..utils.batch_accumulator import BatchAccumulator
        return BatchAccumulator(self, blueprint_id, batch_size)