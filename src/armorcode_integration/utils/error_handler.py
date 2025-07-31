"""
Error Handler for ArmorCode Integration

This module provides comprehensive error handling and recovery mechanisms
for the ArmorCode-Port integration pipeline with graceful degradation,
recovery strategies, and detailed error reporting.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from pathlib import Path
import json

from ..managers.retry_manager import RetryManager, ErrorType
from ..managers.logging_manager import LoggingManager, OperationType, StructuredLogger


class ErrorSeverity(Enum):
    """Severity levels for errors."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """Available recovery strategies."""
    RETRY = "retry"
    SKIP = "skip"
    FALLBACK = "fallback"
    ABORT = "abort"
    CONTINUE = "continue"


class ErrorCategory(Enum):
    """Categories of errors for classification."""
    NETWORK = "network"
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"
    VALIDATION = "validation"
    DATA_PROCESSING = "data_processing"
    BULK_OPERATION = "bulk_operation"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Context information for an error occurrence."""
    operation_type: OperationType
    entity_id: Optional[str] = None
    entity_type: Optional[str] = None
    batch_id: Optional[str] = None
    step_name: Optional[str] = None
    additional_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ErrorRecord:
    """Detailed record of an error occurrence."""
    error_id: str
    timestamp: float
    error: Exception
    error_category: ErrorCategory
    severity: ErrorSeverity
    context: ErrorContext
    recovery_strategy: RecoveryStrategy
    recovery_attempted: bool = False
    recovery_successful: bool = False
    retry_count: int = 0
    message: str = ""
    
    def __post_init__(self):
        if not self.message:
            self.message = str(self.error)


@dataclass
class RecoveryResult:
    """Result of a recovery attempt."""
    success: bool
    strategy_used: RecoveryStrategy
    message: str
    recovered_data: Optional[Any] = None
    continuation_point: Optional[Dict[str, Any]] = None


class ErrorClassifier:
    """Classifies errors into categories and determines severity."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def classify_error(self, error: Exception, context: ErrorContext) -> Tuple[ErrorCategory, ErrorSeverity]:
        """
        Classify an error and determine its severity.
        
        Args:
            error: The exception to classify
            context: Context information about the error
            
        Returns:
            Tuple of (ErrorCategory, ErrorSeverity)
        """
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()
        
        # Network-related errors
        if any(keyword in error_str for keyword in ['connection', 'timeout', 'network', 'dns']):
            return ErrorCategory.NETWORK, ErrorSeverity.MEDIUM
        
        # Authentication errors
        if any(keyword in error_str for keyword in ['unauthorized', 'authentication', 'forbidden', '401', '403']):
            return ErrorCategory.AUTHENTICATION, ErrorSeverity.HIGH
        
        # Rate limiting
        if any(keyword in error_str for keyword in ['rate limit', '429', 'too many requests']):
            return ErrorCategory.RATE_LIMIT, ErrorSeverity.MEDIUM
        
        # Validation errors
        if any(keyword in error_str for keyword in ['validation', 'invalid', 'malformed', 'schema']):
            return ErrorCategory.VALIDATION, ErrorSeverity.LOW
        
        # Data processing errors
        if any(keyword in error_str for keyword in ['json', 'parse', 'decode', 'format']):
            return ErrorCategory.DATA_PROCESSING, ErrorSeverity.MEDIUM
        
        # Bulk operation errors
        if context.operation_type in [OperationType.BATCH_PROCESSING] or context.batch_id:
            return ErrorCategory.BULK_OPERATION, ErrorSeverity.MEDIUM
        
        # Configuration errors
        if any(keyword in error_str for keyword in ['config', 'setting', 'parameter', 'missing']):
            return ErrorCategory.CONFIGURATION, ErrorSeverity.HIGH
        
        # Default classification
        return ErrorCategory.UNKNOWN, ErrorSeverity.MEDIUM
    
    def determine_recovery_strategy(self, category: ErrorCategory, severity: ErrorSeverity, 
                                  retry_count: int = 0) -> RecoveryStrategy:
        """
        Determine the appropriate recovery strategy for an error.
        
        Args:
            category: Error category
            severity: Error severity
            retry_count: Number of previous retry attempts
            
        Returns:
            Recommended recovery strategy
        """
        # Critical errors should abort
        if severity == ErrorSeverity.CRITICAL:
            return RecoveryStrategy.ABORT
        
        # Authentication errors should not be retried
        if category == ErrorCategory.AUTHENTICATION:
            return RecoveryStrategy.ABORT
        
        # Network and rate limit errors can be retried
        if category in [ErrorCategory.NETWORK, ErrorCategory.RATE_LIMIT]:
            if retry_count < 3:
                return RecoveryStrategy.RETRY
            else:
                return RecoveryStrategy.SKIP
        
        # Validation errors should be skipped
        if category == ErrorCategory.VALIDATION:
            return RecoveryStrategy.SKIP
        
        # Data processing errors can be retried once
        if category == ErrorCategory.DATA_PROCESSING:
            if retry_count < 1:
                return RecoveryStrategy.RETRY
            else:
                return RecoveryStrategy.SKIP
        
        # Bulk operation errors should continue with partial results
        if category == ErrorCategory.BULK_OPERATION:
            return RecoveryStrategy.CONTINUE
        
        # Configuration errors should abort
        if category == ErrorCategory.CONFIGURATION:
            return RecoveryStrategy.ABORT
        
        # Default strategy
        return RecoveryStrategy.SKIP


class ErrorHandler:
    """
    Comprehensive error handler with recovery mechanisms.
    
    Provides centralized error handling with classification, recovery strategies,
    and detailed reporting for the ArmorCode integration pipeline.
    """
    
    def __init__(self, logging_manager: LoggingManager, retry_manager: Optional[RetryManager] = None):
        """
        Initialize the error handler.
        
        Args:
            logging_manager: LoggingManager for structured logging
            retry_manager: Optional RetryManager for retry operations
        """
        self.logging_manager = logging_manager
        self.retry_manager = retry_manager or RetryManager()
        self.logger = logging_manager.get_logger("ErrorHandler")
        self.classifier = ErrorClassifier()
        
        # Error tracking
        self.error_records: List[ErrorRecord] = []
        self.error_counts: Dict[ErrorCategory, int] = {}
        self.recovery_stats: Dict[RecoveryStrategy, int] = {}
        
        # Recovery state
        self.continuation_points: Dict[str, Dict[str, Any]] = {}
        self.failed_entities: Set[str] = set()
        
    def handle_error(self, error: Exception, context: ErrorContext, 
                    allow_recovery: bool = True) -> RecoveryResult:
        """
        Handle an error with classification and recovery.
        
        Args:
            error: The exception that occurred
            context: Context information about the error
            allow_recovery: Whether to attempt recovery
            
        Returns:
            RecoveryResult with recovery outcome
        """
        # Generate unique error ID
        error_id = f"err_{int(time.time() * 1000)}_{len(self.error_records)}"
        
        # Classify the error
        category, severity = self.classifier.classify_error(error, context)
        
        # Determine recovery strategy
        existing_errors = [r for r in self.error_records 
                          if r.context.entity_id == context.entity_id and r.error_category == category]
        retry_count = len(existing_errors)
        
        recovery_strategy = self.classifier.determine_recovery_strategy(category, severity, retry_count)
        
        # Create error record
        error_record = ErrorRecord(
            error_id=error_id,
            timestamp=time.time(),
            error=error,
            error_category=category,
            severity=severity,
            context=context,
            recovery_strategy=recovery_strategy,
            retry_count=retry_count
        )
        
        # Track the error
        self.error_records.append(error_record)
        self.error_counts[category] = self.error_counts.get(category, 0) + 1
        
        # Log the error
        self._log_error(error_record)
        
        # Track failed entity
        if context.entity_id:
            self.failed_entities.add(context.entity_id)
        
        # Attempt recovery if allowed
        recovery_result = RecoveryResult(
            success=False,
            strategy_used=recovery_strategy,
            message=f"No recovery attempted for {category.value} error"
        )
        
        if allow_recovery:
            recovery_result = self._attempt_recovery(error_record)
            error_record.recovery_attempted = True
            error_record.recovery_successful = recovery_result.success
        
        return recovery_result
    
    def _log_error(self, error_record: ErrorRecord):
        """Log an error record with appropriate severity."""
        context_info = {
            "error_id": error_record.error_id,
            "error_category": error_record.error_category.value,
            "severity": error_record.severity.value,
            "recovery_strategy": error_record.recovery_strategy.value,
            "retry_count": error_record.retry_count,
            "entity_id": error_record.context.entity_id,
            "entity_type": error_record.context.entity_type,
            "batch_id": error_record.context.batch_id,
            "step_name": error_record.context.step_name
        }
        
        if error_record.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(
                f"Critical error in {error_record.context.operation_type.value}: {error_record.message}",
                error=error_record.error,
                operation_type=error_record.context.operation_type,
                **context_info
            )
        elif error_record.severity == ErrorSeverity.HIGH:
            self.logger.error(
                f"High severity error in {error_record.context.operation_type.value}: {error_record.message}",
                error=error_record.error,
                operation_type=error_record.context.operation_type,
                **context_info
            )
        elif error_record.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(
                f"Medium severity error in {error_record.context.operation_type.value}: {error_record.message}",
                operation_type=error_record.context.operation_type,
                **context_info
            )
        else:
            self.logger.info(
                f"Low severity error in {error_record.context.operation_type.value}: {error_record.message}",
                operation_type=error_record.context.operation_type,
                **context_info
            )
    
    def _attempt_recovery(self, error_record: ErrorRecord) -> RecoveryResult:
        """
        Attempt recovery based on the determined strategy.
        
        Args:
            error_record: The error record to recover from
            
        Returns:
            RecoveryResult with recovery outcome
        """
        strategy = error_record.recovery_strategy
        self.recovery_stats[strategy] = self.recovery_stats.get(strategy, 0) + 1
        
        if strategy == RecoveryStrategy.RETRY:
            return self._retry_recovery(error_record)
        elif strategy == RecoveryStrategy.SKIP:
            return self._skip_recovery(error_record)
        elif strategy == RecoveryStrategy.FALLBACK:
            return self._fallback_recovery(error_record)
        elif strategy == RecoveryStrategy.CONTINUE:
            return self._continue_recovery(error_record)
        elif strategy == RecoveryStrategy.ABORT:
            return self._abort_recovery(error_record)
        
        return RecoveryResult(
            success=False,
            strategy_used=strategy,
            message=f"Unknown recovery strategy: {strategy}"
        )
    
    def _retry_recovery(self, error_record: ErrorRecord) -> RecoveryResult:
        """Implement retry recovery strategy."""
        return RecoveryResult(
            success=True,
            strategy_used=RecoveryStrategy.RETRY,
            message="Error marked for retry",
            continuation_point={"retry_count": error_record.retry_count + 1}
        )
    
    def _skip_recovery(self, error_record: ErrorRecord) -> RecoveryResult:
        """Implement skip recovery strategy."""
        return RecoveryResult(
            success=True,
            strategy_used=RecoveryStrategy.SKIP,
            message="Error handled by skipping failed entity"
        )
    
    def _fallback_recovery(self, error_record: ErrorRecord) -> RecoveryResult:
        """Implement fallback recovery strategy."""
        return RecoveryResult(
            success=True,
            strategy_used=RecoveryStrategy.FALLBACK,
            message="Error handled with fallback mechanism"
        )
    
    def _continue_recovery(self, error_record: ErrorRecord) -> RecoveryResult:
        """Implement continue recovery strategy."""
        return RecoveryResult(
            success=True,
            strategy_used=RecoveryStrategy.CONTINUE,
            message="Error handled by continuing with partial results"
        )
    
    def _abort_recovery(self, error_record: ErrorRecord) -> RecoveryResult:
        """Implement abort recovery strategy."""
        return RecoveryResult(
            success=False,
            strategy_used=RecoveryStrategy.ABORT,
            message="Error requires operation abort"
        )
    
    def should_continue_operation(self, operation_type: OperationType) -> bool:
        """
        Determine if an operation should continue based on error history.
        
        Args:
            operation_type: Type of operation to check
            
        Returns:
            True if operation should continue, False otherwise
        """
        # Get recent errors for this operation type
        recent_errors = [
            r for r in self.error_records[-10:]  # Last 10 errors
            if r.context.operation_type == operation_type
        ]
        
        # Check for critical errors
        critical_errors = [r for r in recent_errors if r.severity == ErrorSeverity.CRITICAL]
        if critical_errors:
            return False
        
        # Check for too many high severity errors
        high_severity_errors = [r for r in recent_errors if r.severity == ErrorSeverity.HIGH]
        if len(high_severity_errors) >= 3:
            return False
        
        # Check for authentication errors
        auth_errors = [r for r in recent_errors if r.error_category == ErrorCategory.AUTHENTICATION]
        if auth_errors:
            return False
        
        return True
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get comprehensive error summary and statistics."""
        total_errors = len(self.error_records)
        
        if total_errors == 0:
            return {
                "total_errors": 0,
                "error_rate": 0.0,
                "categories": {},
                "severities": {},
                "recovery_stats": {},
                "failed_entities": 0
            }
        
        # Count by category
        category_counts = {}
        for category in ErrorCategory:
            category_counts[category.value] = self.error_counts.get(category, 0)
        
        # Count by severity
        severity_counts = {}
        for record in self.error_records:
            severity = record.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Recovery statistics
        recovery_stats = {}
        for strategy in RecoveryStrategy:
            recovery_stats[strategy.value] = self.recovery_stats.get(strategy, 0)
        
        return {
            "total_errors": total_errors,
            "categories": category_counts,
            "severities": severity_counts,
            "recovery_stats": recovery_stats,
            "failed_entities": len(self.failed_entities),
            "recent_errors": len([r for r in self.error_records if time.time() - r.timestamp < 300])  # Last 5 minutes
        }
    
    def save_continuation_point(self, operation_id: str, state: Dict[str, Any]):
        """Save a continuation point for resuming operations."""
        self.continuation_points[operation_id] = {
            "timestamp": time.time(),
            "state": state,
            "failed_entities": list(self.failed_entities)
        }
        
        self.logger.info(
            f"Saved continuation point for operation {operation_id}",
            operation_id=operation_id,
            failed_entities_count=len(self.failed_entities)
        )
    
    def load_continuation_point(self, operation_id: str) -> Optional[Dict[str, Any]]:
        """Load a continuation point for resuming operations."""
        if operation_id in self.continuation_points:
            point = self.continuation_points[operation_id]
            self.logger.info(
                f"Loaded continuation point for operation {operation_id}",
                operation_id=operation_id,
                saved_timestamp=point["timestamp"]
            )
            return point
        return None
    
    def clear_continuation_point(self, operation_id: str):
        """Clear a continuation point after successful completion."""
        if operation_id in self.continuation_points:
            del self.continuation_points[operation_id]
            self.logger.info(f"Cleared continuation point for operation {operation_id}")
    
    def export_error_report(self, file_path: str):
        """Export detailed error report to JSON file."""
        report = {
            "timestamp": time.time(),
            "summary": self.get_error_summary(),
            "errors": [
                {
                    "error_id": record.error_id,
                    "timestamp": record.timestamp,
                    "category": record.error_category.value,
                    "severity": record.severity.value,
                    "message": record.message,
                    "context": {
                        "operation_type": record.context.operation_type.value,
                        "entity_id": record.context.entity_id,
                        "entity_type": record.context.entity_type,
                        "batch_id": record.context.batch_id,
                        "step_name": record.context.step_name
                    },
                    "recovery": {
                        "strategy": record.recovery_strategy.value,
                        "attempted": record.recovery_attempted,
                        "successful": record.recovery_successful,
                        "retry_count": record.retry_count
                    }
                }
                for record in self.error_records
            ]
        }
        
        with open(file_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Exported error report to {file_path}")
