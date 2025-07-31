"""
Logging Manager for ArmorCode Integration

This module provides comprehensive logging and progress reporting functionality
with structured logging, correlation IDs, progress tracking, and detailed
error reporting with troubleshooting information.
"""

import asyncio
import json
import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from threading import Lock


class LogLevel(Enum):
    """Log levels for structured logging."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class OperationType(Enum):
    """Types of operations for progress tracking."""
    PRODUCT_INGESTION = "product_ingestion"
    SUBPRODUCT_INGESTION = "subproduct_ingestion"
    FINDING_INGESTION = "finding_ingestion"
    BLUEPRINT_SETUP = "blueprint_setup"
    BATCH_PROCESSING = "batch_processing"
    API_REQUEST = "api_request"


@dataclass
class ProgressInfo:
    """Progress information for long-running operations."""
    operation_id: str
    operation_type: OperationType
    total_items: int
    processed_items: int = 0
    successful_items: int = 0
    failed_items: int = 0
    start_time: float = field(default_factory=time.time)
    last_update_time: float = field(default_factory=time.time)
    estimated_completion_time: Optional[float] = None
    
    @property
    def percentage_complete(self) -> float:
        """Calculate percentage completion."""
        if self.total_items == 0:
            return 100.0
        return (self.processed_items / self.total_items) * 100
    
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
    def estimated_time_remaining(self) -> Optional[float]:
        """Estimate remaining time in seconds."""
        if self.processed_items == 0 or self.percentage_complete >= 100:
            return None
        
        elapsed = self.elapsed_time
        rate = self.processed_items / elapsed
        remaining_items = self.total_items - self.processed_items
        
        if rate > 0:
            return remaining_items / rate
        return None
    
    def update_progress(self, processed: int, successful: int, failed: int):
        """Update progress counters."""
        self.processed_items = processed
        self.successful_items = successful
        self.failed_items = failed
        self.last_update_time = time.time()
        
        # Update ETA
        eta_seconds = self.estimated_time_remaining
        if eta_seconds:
            self.estimated_completion_time = time.time() + eta_seconds


@dataclass
class LogEntry:
    """Structured log entry with correlation ID and metadata."""
    timestamp: str
    level: LogLevel
    message: str
    correlation_id: str
    operation_type: Optional[OperationType] = None
    component: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_details: Optional[Dict[str, Any]] = None
    troubleshooting_info: Optional[Dict[str, str]] = None


@dataclass
class OperationSummary:
    """Summary of completed operation with statistics."""
    operation_id: str
    operation_type: OperationType
    start_time: float
    end_time: float
    total_items: int
    successful_items: int
    failed_items: int
    error_count: int
    warning_count: int
    
    @property
    def duration(self) -> float:
        """Operation duration in seconds."""
        return self.end_time - self.start_time
    
    @property
    def success_rate(self) -> float:
        """Success rate as percentage."""
        if self.total_items == 0:
            return 0.0
        return (self.successful_items / self.total_items) * 100
    
    @property
    def throughput(self) -> float:
        """Items processed per second."""
        if self.duration == 0:
            return 0.0
        return self.total_items / self.duration


class StructuredLogger:
    """
    Structured logger with correlation ID support and JSON formatting.
    
    Provides consistent logging format across all components with
    correlation ID tracking for tracing operations across components.
    """
    
    def __init__(self, name: str, correlation_id: Optional[str] = None):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name (typically component name)
            correlation_id: Optional correlation ID for operation tracking
        """
        self.name = name
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.logger = logging.getLogger(name)
        self._setup_formatter()
    
    def _setup_formatter(self):
        """Setup JSON formatter for structured logging."""
        # Only add handler if none exists to avoid duplicate logs
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = StructuredFormatter()
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def with_correlation_id(self, correlation_id: str) -> 'StructuredLogger':
        """Create a new logger instance with different correlation ID."""
        return StructuredLogger(self.name, correlation_id)
    
    def debug(self, message: str, operation_type: Optional[OperationType] = None, 
              component: Optional[str] = None, **metadata):
        """Log debug message."""
        self._log(LogLevel.DEBUG, message, operation_type, component, metadata)
    
    def info(self, message: str, operation_type: Optional[OperationType] = None,
             component: Optional[str] = None, **metadata):
        """Log info message."""
        self._log(LogLevel.INFO, message, operation_type, component, metadata)
    
    def warning(self, message: str, operation_type: Optional[OperationType] = None,
                component: Optional[str] = None, **metadata):
        """Log warning message."""
        self._log(LogLevel.WARNING, message, operation_type, component, metadata)
    
    def error(self, message: str, error: Optional[Exception] = None,
              operation_type: Optional[OperationType] = None,
              component: Optional[str] = None, 
              troubleshooting_info: Optional[Dict[str, str]] = None,
              **metadata):
        """Log error message with optional exception details and troubleshooting info."""
        error_details = None
        if error:
            error_details = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": self._format_traceback(error)
            }
        
        self._log(LogLevel.ERROR, message, operation_type, component, metadata,
                 error_details=error_details, troubleshooting_info=troubleshooting_info)
    
    def critical(self, message: str, error: Optional[Exception] = None,
                 operation_type: Optional[OperationType] = None,
                 component: Optional[str] = None,
                 troubleshooting_info: Optional[Dict[str, str]] = None,
                 **metadata):
        """Log critical message with optional exception details and troubleshooting info."""
        error_details = None
        if error:
            error_details = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": self._format_traceback(error)
            }
        
        self._log(LogLevel.CRITICAL, message, operation_type, component, metadata,
                 error_details=error_details, troubleshooting_info=troubleshooting_info)
    
    def _log(self, level: LogLevel, message: str, operation_type: Optional[OperationType],
             component: Optional[str], metadata: Dict[str, Any],
             error_details: Optional[Dict[str, Any]] = None,
             troubleshooting_info: Optional[Dict[str, str]] = None):
        """Internal logging method."""
        log_entry = LogEntry(
            timestamp=datetime.utcnow().isoformat() + "Z",
            level=level,
            message=message,
            correlation_id=self.correlation_id,
            operation_type=operation_type,
            component=component or self.name,
            metadata=metadata,
            error_details=error_details,
            troubleshooting_info=troubleshooting_info
        )
        
        # Convert to standard logging level
        std_level = getattr(logging, level.value)
        self.logger.log(std_level, log_entry)
    
    def _format_traceback(self, error: Exception) -> Optional[str]:
        """Format exception traceback."""
        import traceback
        try:
            return traceback.format_exc()
        except Exception:
            return None


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured log entries with cleaner output."""
    
    def format(self, record):
        """Format log record with simplified, readable format."""
        if isinstance(record.msg, LogEntry):
            # Create a clean, readable format
            timestamp = record.msg.timestamp.split('T')[1].split('.')[0] if 'T' in record.msg.timestamp else record.msg.timestamp
            component = record.msg.component if record.msg.component else "main"
            level = record.msg.level.value
            message = record.msg.message
            
            # Base format: timestamp - level - component - message
            formatted_log = f"{timestamp} - {level} - {component} - {message}"
            
            # Add operation type if present
            if record.msg.operation_type:
                formatted_log += f" [op: {record.msg.operation_type.value}]"
            
            # Add key metadata if present and useful
            if record.msg.metadata:
                # Only include important metadata to keep logs clean
                important_keys = ['entities_processed', 'entities_successful', 'entities_failed', 'batch_size', 'total_entities']
                relevant_metadata = {k: v for k, v in record.msg.metadata.items() if k in important_keys and v is not None}
                if relevant_metadata:
                    metadata_str = ", ".join([f"{k}: {v}" for k, v in relevant_metadata.items()])
                    formatted_log += f" [{metadata_str}]"
            
            # Add error details if present
            if record.msg.error_details:
                error_type = record.msg.error_details.get('type', 'Error')
                error_message = record.msg.error_details.get('message', 'Unknown error')
                formatted_log += f" | Error: {error_type} - {error_message}"
            
            # Add troubleshooting info if present
            if record.msg.troubleshooting_info:
                # Show only the most relevant troubleshooting tip
                if 'immediate_action' in record.msg.troubleshooting_info:
                    formatted_log += f" | Action: {record.msg.troubleshooting_info['immediate_action']}"
            
            return formatted_log
        else:
            # Fallback to standard formatting
            return super().format(record)


class ProgressReporter:
    """
    Progress reporter for long-running operations with ETA calculation.
    
    Provides real-time progress updates with percentage completion,
    success rates, and estimated time to completion.
    """
    
    def __init__(self, logger: StructuredLogger):
        """
        Initialize progress reporter.
        
        Args:
            logger: StructuredLogger instance for progress logging
        """
        self.logger = logger
        self._operations: Dict[str, ProgressInfo] = {}
        self._lock = Lock()
    
    def start_operation(self, operation_type: OperationType, total_items: int,
                       operation_id: Optional[str] = None) -> str:
        """
        Start tracking a new operation.
        
        Args:
            operation_type: Type of operation being tracked
            total_items: Total number of items to process
            operation_id: Optional custom operation ID
            
        Returns:
            Operation ID for tracking
        """
        if operation_id is None:
            operation_id = str(uuid.uuid4())
        
        with self._lock:
            progress_info = ProgressInfo(
                operation_id=operation_id,
                operation_type=operation_type,
                total_items=total_items
            )
            self._operations[operation_id] = progress_info
        
        self.logger.info(
            f"Started {operation_type.value} operation",
            operation_type=operation_type,
            operation_id=operation_id,
            total_items=total_items
        )
        
        return operation_id
    
    def update_progress(self, operation_id: str, processed: int, 
                       successful: int, failed: int):
        """
        Update progress for an operation.
        
        Args:
            operation_id: Operation ID to update
            processed: Number of items processed
            successful: Number of successful items
            failed: Number of failed items
        """
        with self._lock:
            if operation_id not in self._operations:
                self.logger.warning(f"Unknown operation ID: {operation_id}")
                return
            
            progress = self._operations[operation_id]
            progress.update_progress(processed, successful, failed)
        
        # Log progress update
        self._log_progress_update(progress)
    
    def complete_operation(self, operation_id: str) -> Optional[OperationSummary]:
        """
        Mark operation as complete and return summary.
        
        Args:
            operation_id: Operation ID to complete
            
        Returns:
            OperationSummary if operation exists, None otherwise
        """
        with self._lock:
            if operation_id not in self._operations:
                self.logger.warning(f"Unknown operation ID: {operation_id}")
                return None
            
            progress = self._operations.pop(operation_id)
        
        # Create summary
        summary = OperationSummary(
            operation_id=operation_id,
            operation_type=progress.operation_type,
            start_time=progress.start_time,
            end_time=time.time(),
            total_items=progress.total_items,
            successful_items=progress.successful_items,
            failed_items=progress.failed_items,
            error_count=progress.failed_items,  # Simplified for now
            warning_count=0  # Would need separate tracking
        )
        
        # Log completion
        self.logger.info(
            f"Completed {progress.operation_type.value} operation",
            operation_type=progress.operation_type,
            operation_id=operation_id,
            duration=summary.duration,
            success_rate=summary.success_rate,
            throughput=summary.throughput,
            total_items=summary.total_items,
            successful_items=summary.successful_items,
            failed_items=summary.failed_items
        )
        
        return summary
    
    def get_progress(self, operation_id: str) -> Optional[ProgressInfo]:
        """Get current progress for an operation."""
        with self._lock:
            return self._operations.get(operation_id)
    
    def _log_progress_update(self, progress: ProgressInfo):
        """Log progress update with detailed information."""
        eta_str = "N/A"
        if progress.estimated_time_remaining:
            eta_seconds = progress.estimated_time_remaining
            eta_str = f"{eta_seconds:.1f}s"
            if eta_seconds > 60:
                eta_str = f"{eta_seconds/60:.1f}m"
        
        self.logger.info(
            f"Progress update: {progress.percentage_complete:.1f}% complete",
            operation_type=progress.operation_type,
            operation_id=progress.operation_id,
            processed_items=progress.processed_items,
            total_items=progress.total_items,
            successful_items=progress.successful_items,
            failed_items=progress.failed_items,
            success_rate=progress.success_rate,
            percentage_complete=progress.percentage_complete,
            elapsed_time=progress.elapsed_time,
            estimated_time_remaining=eta_str
        )


class LoggingManager:
    """
    Central logging manager that coordinates structured logging and progress reporting.
    
    Provides a unified interface for all logging and progress reporting needs
    across the ArmorCode integration pipeline.
    """
    
    def __init__(self, correlation_id: Optional[str] = None):
        """
        Initialize logging manager.
        
        Args:
            correlation_id: Optional correlation ID for the entire operation
        """
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.logger = StructuredLogger("LoggingManager", self.correlation_id)
        self.progress_reporter = ProgressReporter(self.logger)
        self._component_loggers: Dict[str, StructuredLogger] = {}
    
    def get_logger(self, component_name: str) -> StructuredLogger:
        """
        Get a structured logger for a specific component.
        
        Args:
            component_name: Name of the component requesting the logger
            
        Returns:
            StructuredLogger instance with shared correlation ID
        """
        if component_name not in self._component_loggers:
            self._component_loggers[component_name] = StructuredLogger(
                component_name, self.correlation_id
            )
        return self._component_loggers[component_name]
    
    def get_progress_reporter(self) -> ProgressReporter:
        """Get the progress reporter instance."""
        return self.progress_reporter
    
    @contextmanager
    def operation_context(self, operation_type: OperationType, total_items: int,
                         component_name: str):
        """
        Context manager for tracking an operation with automatic completion.
        
        Args:
            operation_type: Type of operation
            total_items: Total items to process
            component_name: Component performing the operation
            
        Yields:
            Tuple of (logger, operation_id) for the operation
        """
        logger = self.get_logger(component_name)
        operation_id = self.progress_reporter.start_operation(operation_type, total_items)
        
        try:
            yield logger, operation_id
        finally:
            summary = self.progress_reporter.complete_operation(operation_id)
            if summary:
                logger.info(
                    f"Operation completed with {summary.success_rate:.1f}% success rate",
                    operation_type=operation_type,
                    duration=summary.duration,
                    throughput=summary.throughput
                )
    
    def log_troubleshooting_info(self, component_name: str, error: Exception,
                               context: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate troubleshooting information for common errors.
        
        Args:
            component_name: Component where error occurred
            error: The exception that occurred
            context: Additional context about the error
            
        Returns:
            Dictionary with troubleshooting suggestions
        """
        troubleshooting = {}
        error_type = type(error).__name__
        error_msg = str(error).lower()
        
        # Common troubleshooting scenarios
        if "connection" in error_msg or "timeout" in error_msg:
            troubleshooting.update({
                "issue": "Network connectivity problem",
                "suggestion": "Check network connection and API endpoint availability",
                "action": "Retry the operation or check firewall settings"
            })
        elif "authentication" in error_msg or "unauthorized" in error_msg:
            troubleshooting.update({
                "issue": "Authentication failure",
                "suggestion": "Verify API credentials are correct and not expired",
                "action": "Check PORT_CLIENT_ID, PORT_CLIENT_SECRET, and ARMORCODE_API_KEY"
            })
        elif "rate limit" in error_msg or "429" in error_msg:
            troubleshooting.update({
                "issue": "API rate limiting",
                "suggestion": "Reduce request frequency or implement backoff",
                "action": "Wait before retrying or reduce batch sizes"
            })
        elif "validation" in error_msg or "invalid" in error_msg:
            troubleshooting.update({
                "issue": "Data validation error",
                "suggestion": "Check data format and required fields",
                "action": "Review entity structure and blueprint requirements"
            })
        else:
            troubleshooting.update({
                "issue": f"Unexpected {error_type}",
                "suggestion": "Review error details and context",
                "action": "Check logs for more information or contact support"
            })
        
        # Add context-specific information
        if context:
            troubleshooting["context"] = json.dumps(context, default=str)
        
        return troubleshooting


# Global logging manager instance
_global_logging_manager: Optional[LoggingManager] = None


def get_logging_manager(correlation_id: Optional[str] = None) -> LoggingManager:
    """
    Get or create the global logging manager instance.
    
    Args:
        correlation_id: Optional correlation ID for new instance
        
    Returns:
        LoggingManager instance
    """
    global _global_logging_manager
    if _global_logging_manager is None or correlation_id is not None:
        _global_logging_manager = LoggingManager(correlation_id)
    return _global_logging_manager


def reset_logging_manager():
    """Reset the global logging manager (useful for testing)."""
    global _global_logging_manager
    _global_logging_manager = None