"""
Recovery Manager for ArmorCode Integration

This module provides recovery mechanisms and continuation logic for handling
failures in bulk operations, network interruptions, and other recoverable errors.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from ..utils.error_handler import ErrorHandler, ErrorContext, RecoveryResult, RecoveryStrategy
from .logging_manager import LoggingManager, OperationType
from .retry_manager import RetryManager


class OperationState(Enum):
    """States of operation execution."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    RECOVERING = "recovering"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


@dataclass
class CheckpointData:
    """Data stored at a checkpoint for recovery."""
    operation_id: str
    operation_type: OperationType
    timestamp: float
    processed_items: int
    successful_items: int
    failed_items: int
    current_batch: int
    total_batches: int
    state_data: Dict[str, Any] = field(default_factory=dict)
    failed_entities: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert checkpoint to dictionary for serialization."""
        return {
            "operation_id": self.operation_id,
            "operation_type": self.operation_type.value,
            "timestamp": self.timestamp,
            "processed_items": self.processed_items,
            "successful_items": self.successful_items,
            "failed_items": self.failed_items,
            "current_batch": self.current_batch,
            "total_batches": self.total_batches,
            "state_data": self.state_data,
            "failed_entities": self.failed_entities
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CheckpointData':
        """Create checkpoint from dictionary."""
        return cls(
            operation_id=data["operation_id"],
            operation_type=OperationType(data["operation_type"]),
            timestamp=data["timestamp"],
            processed_items=data["processed_items"],
            successful_items=data["successful_items"],
            failed_items=data["failed_items"],
            current_batch=data["current_batch"],
            total_batches=data["total_batches"],
            state_data=data.get("state_data", {}),
            failed_entities=data.get("failed_entities", [])
        )


@dataclass
class RecoveryPlan:
    """Plan for recovering from failures."""
    operation_id: str
    recovery_strategy: RecoveryStrategy
    checkpoint: CheckpointData
    retry_items: List[str] = field(default_factory=list)
    skip_items: List[str] = field(default_factory=list)
    fallback_data: Optional[Dict[str, Any]] = None
    estimated_duration: Optional[float] = None


class CheckpointManager:
    """Manages checkpoints for operation recovery."""
    
    def __init__(self, checkpoint_dir: str = ".checkpoints"):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to store checkpoint files
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def save_checkpoint(self, checkpoint: CheckpointData) -> str:
        """
        Save a checkpoint to disk.
        
        Args:
            checkpoint: Checkpoint data to save
            
        Returns:
            Path to saved checkpoint file
        """
        filename = f"checkpoint_{checkpoint.operation_id}_{int(checkpoint.timestamp)}.json"
        filepath = self.checkpoint_dir / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(checkpoint.to_dict(), f, indent=2, default=str)
            
            self.logger.info(f"Saved checkpoint for operation {checkpoint.operation_id}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
            raise
    
    def load_checkpoint(self, operation_id: str) -> Optional[CheckpointData]:
        """
        Load the latest checkpoint for an operation.
        
        Args:
            operation_id: Operation ID to load checkpoint for
            
        Returns:
            CheckpointData if found, None otherwise
        """
        # Find all checkpoint files for this operation
        pattern = f"checkpoint_{operation_id}_*.json"
        checkpoint_files = list(self.checkpoint_dir.glob(pattern))
        
        if not checkpoint_files:
            return None
        
        # Get the most recent checkpoint
        latest_file = max(checkpoint_files, key=lambda f: f.stat().st_mtime)
        
        try:
            with open(latest_file, 'r') as f:
                data = json.load(f)
            
            checkpoint = CheckpointData.from_dict(data)
            self.logger.info(f"Loaded checkpoint for operation {operation_id}")
            return checkpoint
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint from {latest_file}: {e}")
            return None
    
    def list_checkpoints(self) -> List[str]:
        """List all available checkpoint operation IDs."""
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.json"))
        operation_ids = set()
        
        for file in checkpoint_files:
            # Extract operation ID from filename
            parts = file.stem.split('_')
            if len(parts) >= 3:
                operation_id = '_'.join(parts[1:-1])  # Everything between 'checkpoint' and timestamp
                operation_ids.add(operation_id)
        
        return list(operation_ids)
    
    def cleanup_checkpoints(self, operation_id: str):
        """Clean up checkpoint files for a completed operation."""
        pattern = f"checkpoint_{operation_id}_*.json"
        checkpoint_files = list(self.checkpoint_dir.glob(pattern))
        
        for file in checkpoint_files:
            try:
                file.unlink()
                self.logger.debug(f"Deleted checkpoint file: {file}")
            except Exception as e:
                self.logger.warning(f"Failed to delete checkpoint file {file}: {e}")
        
        if checkpoint_files:
            self.logger.info(f"Cleaned up {len(checkpoint_files)} checkpoint files for operation {operation_id}")


class RecoveryManager:
    """
    Manages recovery operations and continuation logic.
    
    Provides mechanisms for:
    - Creating recovery plans from failures
    - Executing recovery strategies
    - Managing checkpoints and continuation points
    - Handling partial failures in bulk operations
    """
    
    def __init__(self, error_handler: ErrorHandler, retry_manager: RetryManager,
                 logging_manager: LoggingManager, checkpoint_dir: str = ".checkpoints"):
        """
        Initialize recovery manager.
        
        Args:
            error_handler: ErrorHandler for error classification and handling
            retry_manager: RetryManager for retry operations
            logging_manager: LoggingManager for structured logging
            checkpoint_dir: Directory for storing checkpoints
        """
        self.error_handler = error_handler
        self.retry_manager = retry_manager
        self.logging_manager = logging_manager
        self.logger = logging_manager.get_logger("RecoveryManager")
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        
        # Recovery state
        self.active_recoveries: Dict[str, RecoveryPlan] = {}
        self.recovery_history: List[RecoveryPlan] = []
        
    async def create_recovery_plan(self, operation_id: str, error: Exception,
                                 context: ErrorContext, checkpoint: CheckpointData) -> RecoveryPlan:
        """
        Create a recovery plan based on the error and current state.
        
        Args:
            operation_id: ID of the failed operation
            error: The error that occurred
            context: Context information about the error
            checkpoint: Current checkpoint data
            
        Returns:
            RecoveryPlan for handling the failure
        """
        # Handle the error to get recovery strategy
        recovery_result = self.error_handler.handle_error(error, context)
        
        # Create recovery plan
        plan = RecoveryPlan(
            operation_id=operation_id,
            recovery_strategy=recovery_result.strategy_used,
            checkpoint=checkpoint
        )
        
        # Determine specific recovery actions based on strategy
        if recovery_result.strategy_used == RecoveryStrategy.RETRY:
            plan.retry_items = [context.entity_id] if context.entity_id else []
            plan.estimated_duration = 30.0  # Estimate retry duration
            
        elif recovery_result.strategy_used == RecoveryStrategy.SKIP:
            plan.skip_items = [context.entity_id] if context.entity_id else []
            plan.estimated_duration = 1.0  # Skip is fast
            
        elif recovery_result.strategy_used == RecoveryStrategy.CONTINUE:
            # Continue with partial results
            plan.estimated_duration = 5.0
            
        elif recovery_result.strategy_used == RecoveryStrategy.FALLBACK:
            # Use fallback mechanism
            plan.fallback_data = {"use_fallback": True}
            plan.estimated_duration = 60.0
            
        # Save the plan
        self.active_recoveries[operation_id] = plan
        
        self.logger.info(
            f"Created recovery plan for operation {operation_id}",
            operation_id=operation_id,
            strategy=recovery_result.strategy_used.value,
            retry_items=len(plan.retry_items),
            skip_items=len(plan.skip_items),
            estimated_duration=plan.estimated_duration
        )
        
        return plan
    
    async def execute_recovery_plan(self, plan: RecoveryPlan,
                                  recovery_function: Callable) -> bool:
        """
        Execute a recovery plan.
        
        Args:
            plan: Recovery plan to execute
            recovery_function: Function to call for recovery execution
            
        Returns:
            True if recovery was successful, False otherwise
        """
        operation_id = plan.operation_id
        
        self.logger.info(
            f"Executing recovery plan for operation {operation_id}",
            operation_id=operation_id,
            strategy=plan.recovery_strategy.value
        )
        
        try:
            # Execute the recovery based on strategy
            if plan.recovery_strategy == RecoveryStrategy.RETRY:
                success = await self._execute_retry_recovery(plan, recovery_function)
                
            elif plan.recovery_strategy == RecoveryStrategy.SKIP:
                success = await self._execute_skip_recovery(plan, recovery_function)
                
            elif plan.recovery_strategy == RecoveryStrategy.CONTINUE:
                success = await self._execute_continue_recovery(plan, recovery_function)
                
            elif plan.recovery_strategy == RecoveryStrategy.FALLBACK:
                success = await self._execute_fallback_recovery(plan, recovery_function)
                
            elif plan.recovery_strategy == RecoveryStrategy.ABORT:
                success = False
                self.logger.error(f"Recovery plan requires abort for operation {operation_id}")
                
            else:
                success = False
                self.logger.error(f"Unknown recovery strategy: {plan.recovery_strategy}")
            
            # Update recovery history
            self.recovery_history.append(plan)
            
            # Clean up active recovery
            if operation_id in self.active_recoveries:
                del self.active_recoveries[operation_id]
            
            if success:
                self.logger.info(f"Recovery successful for operation {operation_id}")
            else:
                self.logger.error(f"Recovery failed for operation {operation_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(
                f"Recovery execution failed for operation {operation_id}: {e}",
                error=e,
                operation_id=operation_id
            )
            return False
    
    async def _execute_retry_recovery(self, plan: RecoveryPlan,
                                    recovery_function: Callable) -> bool:
        """Execute retry recovery strategy."""
        try:
            # Use retry manager for the recovery function
            await self.retry_manager.with_retry(
                recovery_function,
                plan.checkpoint,
                plan.retry_items
            )
            return True
        except Exception as e:
            self.logger.error(f"Retry recovery failed: {e}")
            return False
    
    async def _execute_skip_recovery(self, plan: RecoveryPlan,
                                   recovery_function: Callable) -> bool:
        """Execute skip recovery strategy."""
        try:
            # Call recovery function with skip instructions
            await recovery_function(plan.checkpoint, skip_items=plan.skip_items)
            return True
        except Exception as e:
            self.logger.error(f"Skip recovery failed: {e}")
            return False
    
    async def _execute_continue_recovery(self, plan: RecoveryPlan,
                                       recovery_function: Callable) -> bool:
        """Execute continue recovery strategy."""
        try:
            # Continue with partial results
            await recovery_function(plan.checkpoint, continue_partial=True)
            return True
        except Exception as e:
            self.logger.error(f"Continue recovery failed: {e}")
            return False
    
    async def _execute_fallback_recovery(self, plan: RecoveryPlan,
                                       recovery_function: Callable) -> bool:
        """Execute fallback recovery strategy."""
        try:
            # Use fallback mechanism
            await recovery_function(plan.checkpoint, fallback_data=plan.fallback_data)
            return True
        except Exception as e:
            self.logger.error(f"Fallback recovery failed: {e}")
            return False
    
    def save_checkpoint(self, operation_id: str, operation_type: OperationType,
                       processed_items: int, successful_items: int, failed_items: int,
                       current_batch: int, total_batches: int,
                       state_data: Optional[Dict[str, Any]] = None,
                       failed_entities: Optional[List[str]] = None) -> str:
        """
        Save a checkpoint for an operation.
        
        Args:
            operation_id: Unique operation identifier
            operation_type: Type of operation
            processed_items: Number of items processed
            successful_items: Number of successful items
            failed_items: Number of failed items
            current_batch: Current batch number
            total_batches: Total number of batches
            state_data: Additional state data
            failed_entities: List of failed entity IDs
            
        Returns:
            Path to saved checkpoint file
        """
        checkpoint = CheckpointData(
            operation_id=operation_id,
            operation_type=operation_type,
            timestamp=time.time(),
            processed_items=processed_items,
            successful_items=successful_items,
            failed_items=failed_items,
            current_batch=current_batch,
            total_batches=total_batches,
            state_data=state_data or {},
            failed_entities=failed_entities or []
        )
        
        return self.checkpoint_manager.save_checkpoint(checkpoint)
    
    def load_checkpoint(self, operation_id: str) -> Optional[CheckpointData]:
        """Load checkpoint for an operation."""
        return self.checkpoint_manager.load_checkpoint(operation_id)
    
    def can_resume_operation(self, operation_id: str) -> bool:
        """Check if an operation can be resumed from checkpoint."""
        checkpoint = self.load_checkpoint(operation_id)
        if not checkpoint:
            return False
        
        # Check if checkpoint is recent enough (within 24 hours)
        age_hours = (time.time() - checkpoint.timestamp) / 3600
        if age_hours > 24:
            self.logger.warning(f"Checkpoint for {operation_id} is too old ({age_hours:.1f} hours)")
            return False
        
        return True
    
    async def resume_operation(self, operation_id: str,
                             resume_function: Callable) -> bool:
        """
        Resume an operation from checkpoint.
        
        Args:
            operation_id: Operation ID to resume
            resume_function: Function to call for resuming the operation
            
        Returns:
            True if resume was successful, False otherwise
        """
        checkpoint = self.load_checkpoint(operation_id)
        if not checkpoint:
            self.logger.error(f"No checkpoint found for operation {operation_id}")
            return False
        
        self.logger.info(
            f"Resuming operation {operation_id} from checkpoint",
            operation_id=operation_id,
            checkpoint_timestamp=checkpoint.timestamp,
            processed_items=checkpoint.processed_items,
            current_batch=checkpoint.current_batch,
            total_batches=checkpoint.total_batches
        )
        
        try:
            # Call the resume function with checkpoint data
            success = await resume_function(checkpoint)
            
            if success:
                # Clean up checkpoint after successful resume
                self.checkpoint_manager.cleanup_checkpoints(operation_id)
                self.logger.info(f"Successfully resumed operation {operation_id}")
            else:
                self.logger.error(f"Failed to resume operation {operation_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error resuming operation {operation_id}: {e}", error=e)
            return False
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get statistics about recovery operations."""
        if not self.recovery_history:
            return {
                "total_recoveries": 0,
                "success_rate": 0.0,
                "strategies_used": {},
                "average_duration": 0.0
            }
        
        total_recoveries = len(self.recovery_history)
        strategies_used = {}
        
        for plan in self.recovery_history:
            strategy = plan.recovery_strategy.value
            strategies_used[strategy] = strategies_used.get(strategy, 0) + 1
        
        return {
            "total_recoveries": total_recoveries,
            "strategies_used": strategies_used,
            "active_recoveries": len(self.active_recoveries),
            "available_checkpoints": len(self.checkpoint_manager.list_checkpoints())
        }
    
    def cleanup_old_checkpoints(self, max_age_hours: float = 168):  # 7 days default
        """Clean up old checkpoint files."""
        checkpoint_files = list(self.checkpoint_manager.checkpoint_dir.glob("checkpoint_*.json"))
        current_time = time.time()
        cleaned_count = 0
        
        for file in checkpoint_files:
            try:
                file_age = (current_time - file.stat().st_mtime) / 3600
                if file_age > max_age_hours:
                    file.unlink()
                    cleaned_count += 1
            except Exception as e:
                self.logger.warning(f"Failed to clean up checkpoint file {file}: {e}")
        
        if cleaned_count > 0:
            self.logger.info(f"Cleaned up {cleaned_count} old checkpoint files")
