"""
Progress Tracker for ArmorCode Integration

This module provides enhanced progress tracking capabilities with
real-time updates, ETA calculations, and detailed statistics reporting.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from threading import Lock

from .logging_manager import LoggingManager, OperationType, StructuredLogger


@dataclass
class ProgressSnapshot:
    """Snapshot of progress at a specific point in time."""
    timestamp: float
    processed_items: int
    successful_items: int
    failed_items: int
    percentage_complete: float
    success_rate: float
    throughput: float  # items per second
    estimated_time_remaining: Optional[float] = None


class ProgressTracker:
    """
    Enhanced progress tracker with real-time monitoring and statistics.
    
    Provides detailed progress tracking with automatic ETA calculation,
    throughput monitoring, and periodic progress reporting.
    """
    
    def __init__(self, operation_id: str, operation_type: OperationType,
                 total_items: int, logger: StructuredLogger,
                 report_interval: float = 5.0):
        """
        Initialize progress tracker.
        
        Args:
            operation_id: Unique operation identifier
            operation_type: Type of operation being tracked
            total_items: Total number of items to process
            logger: Logger for progress reporting
            report_interval: Interval in seconds between progress reports
        """
        self.operation_id = operation_id
        self.operation_type = operation_type
        self.total_items = total_items
        self.logger = logger
        self.report_interval = report_interval
        
        # Progress state
        self.start_time = time.time()
        self.processed_items = 0
        self.successful_items = 0
        self.failed_items = 0
        self.last_report_time = self.start_time
        
        # Thread safety
        self._lock = Lock()
        
        # Progress history for trend analysis
        self._progress_history: List[ProgressSnapshot] = []
        self._max_history_size = 100
        
        # Reporting task
        self._reporting_task: Optional[asyncio.Task] = None
        self._stop_reporting = False
    
    def update(self, processed_delta: int = 1, successful_delta: int = 0, 
               failed_delta: int = 0):
        """
        Update progress counters.
        
        Args:
            processed_delta: Number of newly processed items
            successful_delta: Number of newly successful items
            failed_delta: Number of newly failed items
        """
        with self._lock:
            self.processed_items += processed_delta
            self.successful_items += successful_delta
            self.failed_items += failed_delta
            
            # Ensure consistency
            if self.successful_items + self.failed_items > self.processed_items:
                self.processed_items = self.successful_items + self.failed_items
            
            # Add to history
            self._add_to_history()
    
    def set_totals(self, processed: int, successful: int, failed: int):
        """
        Set absolute totals (useful for batch updates).
        
        Args:
            processed: Total processed items
            successful: Total successful items
            failed: Total failed items
        """
        with self._lock:
            self.processed_items = processed
            self.successful_items = successful
            self.failed_items = failed
            
            # Add to history
            self._add_to_history()
    
    def _add_to_history(self):
        """Add current state to progress history."""
        snapshot = ProgressSnapshot(
            timestamp=time.time(),
            processed_items=self.processed_items,
            successful_items=self.successful_items,
            failed_items=self.failed_items,
            percentage_complete=self.percentage_complete,
            success_rate=self.success_rate,
            throughput=self.current_throughput,
            estimated_time_remaining=self.estimated_time_remaining
        )
        
        self._progress_history.append(snapshot)
        
        # Limit history size
        if len(self._progress_history) > self._max_history_size:
            self._progress_history.pop(0)
    
    @property
    def percentage_complete(self) -> float:
        """Calculate percentage completion."""
        if self.total_items == 0:
            return 100.0
        return min(100.0, (self.processed_items / self.total_items) * 100)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.processed_items == 0:
            return 0.0
        return (self.successful_items / self.processed_items) * 100
    
    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.start_time
    
    @property
    def current_throughput(self) -> float:
        """Calculate current throughput (items per second)."""
        elapsed = self.elapsed_time
        if elapsed == 0:
            return 0.0
        return self.processed_items / elapsed
    
    @property
    def estimated_time_remaining(self) -> Optional[float]:
        """Estimate remaining time in seconds based on current throughput."""
        if self.processed_items == 0 or self.percentage_complete >= 100:
            return None
        
        throughput = self.current_throughput
        if throughput <= 0:
            return None
        
        remaining_items = self.total_items - self.processed_items
        return remaining_items / throughput
    
    @property
    def recent_throughput(self) -> float:
        """Calculate throughput based on recent progress (last 30 seconds)."""
        if len(self._progress_history) < 2:
            return self.current_throughput
        
        current_time = time.time()
        recent_snapshots = [
            s for s in self._progress_history 
            if current_time - s.timestamp <= 30.0
        ]
        
        if len(recent_snapshots) < 2:
            return self.current_throughput
        
        oldest = recent_snapshots[0]
        newest = recent_snapshots[-1]
        
        time_diff = newest.timestamp - oldest.timestamp
        items_diff = newest.processed_items - oldest.processed_items
        
        if time_diff <= 0:
            return self.current_throughput
        
        return items_diff / time_diff
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get comprehensive progress summary."""
        with self._lock:
            eta_seconds = self.estimated_time_remaining
            eta_str = "N/A"
            if eta_seconds:
                if eta_seconds < 60:
                    eta_str = f"{eta_seconds:.1f}s"
                elif eta_seconds < 3600:
                    eta_str = f"{eta_seconds/60:.1f}m"
                else:
                    eta_str = f"{eta_seconds/3600:.1f}h"
            
            return {
                "operation_id": self.operation_id,
                "operation_type": self.operation_type.value,
                "total_items": self.total_items,
                "processed_items": self.processed_items,
                "successful_items": self.successful_items,
                "failed_items": self.failed_items,
                "percentage_complete": round(self.percentage_complete, 2),
                "success_rate": round(self.success_rate, 2),
                "elapsed_time": round(self.elapsed_time, 2),
                "current_throughput": round(self.current_throughput, 2),
                "recent_throughput": round(self.recent_throughput, 2),
                "estimated_time_remaining": eta_str,
                "eta_seconds": eta_seconds
            }
    
    def should_report_progress(self) -> bool:
        """Check if it's time to report progress."""
        return time.time() - self.last_report_time >= self.report_interval
    
    def report_progress(self, force: bool = False):
        """
        Report current progress if interval has elapsed.
        
        Args:
            force: Force reporting regardless of interval
        """
        if not force and not self.should_report_progress():
            return
        
        summary = self.get_progress_summary()
        
        # Remove conflicting keys from summary before passing as kwargs
        summary_metadata = {k: v for k, v in summary.items() if k != 'operation_type'}
        
        self.logger.info(
            f"Progress: {summary['percentage_complete']}% complete "
            f"({summary['processed_items']}/{summary['total_items']} items)",
            operation_type=self.operation_type,
            **summary_metadata
        )
        
        self.last_report_time = time.time()
    
    async def start_periodic_reporting(self):
        """Start periodic progress reporting in background."""
        if self._reporting_task is not None:
            return  # Already started
        
        self._stop_reporting = False
        self._reporting_task = asyncio.create_task(self._periodic_report_loop())
    
    async def stop_periodic_reporting(self):
        """Stop periodic progress reporting."""
        self._stop_reporting = True
        if self._reporting_task:
            try:
                await self._reporting_task
            except asyncio.CancelledError:
                pass
            self._reporting_task = None
    
    async def _periodic_report_loop(self):
        """Background task for periodic progress reporting."""
        try:
            while not self._stop_reporting:
                await asyncio.sleep(self.report_interval)
                if not self._stop_reporting:
                    self.report_progress()
        except asyncio.CancelledError:
            pass
    
    def get_trend_analysis(self) -> Dict[str, Any]:
        """Analyze progress trends from history."""
        if len(self._progress_history) < 2:
            return {"trend": "insufficient_data"}
        
        recent_snapshots = self._progress_history[-10:]  # Last 10 snapshots
        
        # Calculate throughput trend
        throughputs = [s.throughput for s in recent_snapshots]
        if len(throughputs) >= 2:
            throughput_trend = "increasing" if throughputs[-1] > throughputs[0] else "decreasing"
        else:
            throughput_trend = "stable"
        
        # Calculate success rate trend
        success_rates = [s.success_rate for s in recent_snapshots if s.processed_items > 0]
        if len(success_rates) >= 2:
            success_trend = "improving" if success_rates[-1] > success_rates[0] else "declining"
        else:
            success_trend = "stable"
        
        return {
            "throughput_trend": throughput_trend,
            "success_rate_trend": success_trend,
            "average_throughput": sum(throughputs) / len(throughputs) if throughputs else 0,
            "average_success_rate": sum(success_rates) / len(success_rates) if success_rates else 0,
            "data_points": len(self._progress_history)
        }


class BatchProgressTracker:
    """
    Progress tracker specifically designed for batch processing operations.
    
    Tracks progress across multiple batches with detailed batch-level statistics.
    """
    
    def __init__(self, operation_id: str, operation_type: OperationType,
                 total_batches: int, total_items: int, logger: StructuredLogger):
        """
        Initialize batch progress tracker.
        
        Args:
            operation_id: Unique operation identifier
            operation_type: Type of operation being tracked
            total_batches: Total number of batches to process
            total_items: Total number of items across all batches
            logger: Logger for progress reporting
        """
        self.operation_id = operation_id
        self.operation_type = operation_type
        self.total_batches = total_batches
        self.total_items = total_items
        self.logger = logger
        
        # Batch tracking
        self.completed_batches = 0
        self.successful_batches = 0
        self.failed_batches = 0
        
        # Item tracking
        self.processed_items = 0
        self.successful_items = 0
        self.failed_items = 0
        
        # Timing
        self.start_time = time.time()
        self.batch_times: List[float] = []
        
        # Thread safety
        self._lock = Lock()
    
    def complete_batch(self, batch_items: int, successful_items: int, 
                      failed_items: int, batch_duration: float):
        """
        Mark a batch as completed and update statistics.
        
        Args:
            batch_items: Number of items in the batch
            successful_items: Number of successful items in batch
            failed_items: Number of failed items in batch
            batch_duration: Time taken to process the batch
        """
        with self._lock:
            self.completed_batches += 1
            self.processed_items += batch_items
            self.successful_items += successful_items
            self.failed_items += failed_items
            self.batch_times.append(batch_duration)
            
            # Determine if batch was successful (majority of items succeeded)
            if successful_items > failed_items:
                self.successful_batches += 1
            else:
                self.failed_batches += 1
        
        # Report batch completion
        self.logger.info(
            f"Batch {self.completed_batches}/{self.total_batches} completed",
            operation_type=self.operation_type,
            operation_id=self.operation_id,
            batch_items=batch_items,
            batch_successful=successful_items,
            batch_failed=failed_items,
            batch_duration=batch_duration,
            batch_success_rate=(successful_items / batch_items * 100) if batch_items > 0 else 0,
            overall_progress=(self.completed_batches / self.total_batches * 100)
        )
    
    @property
    def batch_completion_percentage(self) -> float:
        """Calculate batch completion percentage."""
        if self.total_batches == 0:
            return 100.0
        return (self.completed_batches / self.total_batches) * 100
    
    @property
    def item_completion_percentage(self) -> float:
        """Calculate item completion percentage."""
        if self.total_items == 0:
            return 100.0
        return (self.processed_items / self.total_items) * 100
    
    @property
    def overall_success_rate(self) -> float:
        """Calculate overall success rate."""
        if self.processed_items == 0:
            return 0.0
        return (self.successful_items / self.processed_items) * 100
    
    @property
    def batch_success_rate(self) -> float:
        """Calculate batch success rate."""
        if self.completed_batches == 0:
            return 0.0
        return (self.successful_batches / self.completed_batches) * 100
    
    @property
    def average_batch_time(self) -> float:
        """Calculate average time per batch."""
        if not self.batch_times:
            return 0.0
        return sum(self.batch_times) / len(self.batch_times)
    
    @property
    def estimated_time_remaining(self) -> Optional[float]:
        """Estimate remaining time based on average batch time."""
        if not self.batch_times or self.completed_batches >= self.total_batches:
            return None
        
        remaining_batches = self.total_batches - self.completed_batches
        return remaining_batches * self.average_batch_time
    
    def get_batch_summary(self) -> Dict[str, Any]:
        """Get comprehensive batch processing summary."""
        with self._lock:
            eta_seconds = self.estimated_time_remaining
            eta_str = "N/A"
            if eta_seconds:
                if eta_seconds < 60:
                    eta_str = f"{eta_seconds:.1f}s"
                elif eta_seconds < 3600:
                    eta_str = f"{eta_seconds/60:.1f}m"
                else:
                    eta_str = f"{eta_seconds/3600:.1f}h"
            
            return {
                "operation_id": self.operation_id,
                "operation_type": self.operation_type.value,
                "total_batches": self.total_batches,
                "completed_batches": self.completed_batches,
                "successful_batches": self.successful_batches,
                "failed_batches": self.failed_batches,
                "batch_completion_percentage": round(self.batch_completion_percentage, 2),
                "batch_success_rate": round(self.batch_success_rate, 2),
                "total_items": self.total_items,
                "processed_items": self.processed_items,
                "successful_items": self.successful_items,
                "failed_items": self.failed_items,
                "item_completion_percentage": round(self.item_completion_percentage, 2),
                "overall_success_rate": round(self.overall_success_rate, 2),
                "average_batch_time": round(self.average_batch_time, 2),
                "estimated_time_remaining": eta_str,
                "eta_seconds": eta_seconds,
                "elapsed_time": round(time.time() - self.start_time, 2)
            }


def create_progress_tracker(operation_type: OperationType, total_items: int,
                          logging_manager: LoggingManager,
                          component_name: str = "ProgressTracker") -> ProgressTracker:
    """
    Factory function to create a progress tracker with proper logging setup.
    
    Args:
        operation_type: Type of operation to track
        total_items: Total number of items to process
        logging_manager: LoggingManager instance
        component_name: Name of component using the tracker
        
    Returns:
        Configured ProgressTracker instance
    """
    logger = logging_manager.get_logger(component_name)
    operation_id = logging_manager.progress_reporter.start_operation(
        operation_type, total_items
    )
    
    return ProgressTracker(
        operation_id=operation_id,
        operation_type=operation_type,
        total_items=total_items,
        logger=logger
    )