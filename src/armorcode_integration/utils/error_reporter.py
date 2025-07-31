"""
Error Reporter for ArmorCode Integration

This module provides detailed error reporting with entity-level failure tracking,
comprehensive statistics, and actionable troubleshooting information.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .error_handler import ErrorHandler, ErrorRecord, ErrorCategory, ErrorSeverity
from ..managers.logging_manager import LoggingManager, OperationType
from ..managers.recovery_manager import RecoveryManager


@dataclass
class EntityFailureReport:
    """Detailed report for a failed entity."""
    entity_id: str
    entity_type: str
    failure_count: int
    first_failure: float
    last_failure: float
    error_categories: List[str]
    error_messages: List[str]
    recovery_attempts: int
    final_status: str  # "failed", "skipped", "recovered"


@dataclass
class OperationReport:
    """Report for a specific operation."""
    operation_id: str
    operation_type: OperationType
    start_time: float
    end_time: Optional[float]
    total_entities: int
    successful_entities: int
    failed_entities: int
    skipped_entities: int
    error_rate: float
    recovery_rate: float
    duration_seconds: Optional[float] = None
    
    def __post_init__(self):
        if self.end_time and not self.duration_seconds:
            self.duration_seconds = self.end_time - self.start_time


@dataclass
class ErrorTrend:
    """Trend analysis for errors over time."""
    time_window: str
    error_count: int
    error_rate: float
    dominant_category: str
    trend_direction: str  # "increasing", "decreasing", "stable"


class ErrorReporter:
    """
    Comprehensive error reporting system.
    
    Provides detailed error analysis, entity-level failure tracking,
    trend analysis, and actionable troubleshooting recommendations.
    """
    
    def __init__(self, error_handler: ErrorHandler, recovery_manager: RecoveryManager,
                 logging_manager: LoggingManager, reports_dir: str = "error_reports"):
        """
        Initialize error reporter.
        
        Args:
            error_handler: ErrorHandler for accessing error data
            recovery_manager: RecoveryManager for recovery statistics
            logging_manager: LoggingManager for structured logging
            reports_dir: Directory to store error reports
        """
        self.error_handler = error_handler
        self.recovery_manager = recovery_manager
        self.logging_manager = logging_manager
        self.logger = logging_manager.get_logger("ErrorReporter")
        
        # Setup reports directory
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(exist_ok=True)
        
        # Tracking data
        self.operation_reports: Dict[str, OperationReport] = {}
        self.entity_failures: Dict[str, EntityFailureReport] = {}
        self.error_trends: List[ErrorTrend] = []
    
    def start_operation_tracking(self, operation_id: str, operation_type: OperationType,
                               total_entities: int):
        """Start tracking an operation for reporting."""
        self.operation_reports[operation_id] = OperationReport(
            operation_id=operation_id,
            operation_type=operation_type,
            start_time=time.time(),
            end_time=None,
            total_entities=total_entities,
            successful_entities=0,
            failed_entities=0,
            skipped_entities=0,
            error_rate=0.0,
            recovery_rate=0.0
        )
        
        self.logger.info(
            f"Started tracking operation {operation_id}",
            operation_id=operation_id,
            operation_type=operation_type.value,
            total_entities=total_entities
        )
    
    def update_operation_progress(self, operation_id: str, successful: int = 0,
                                failed: int = 0, skipped: int = 0):
        """Update progress for an operation."""
        if operation_id not in self.operation_reports:
            self.logger.warning(f"Operation {operation_id} not found for progress update")
            return
        
        report = self.operation_reports[operation_id]
        report.successful_entities += successful
        report.failed_entities += failed
        report.skipped_entities += skipped
        
        # Calculate error rate
        processed = report.successful_entities + report.failed_entities + report.skipped_entities
        if processed > 0:
            report.error_rate = (report.failed_entities + report.skipped_entities) / processed
    
    def complete_operation_tracking(self, operation_id: str):
        """Complete tracking for an operation."""
        if operation_id not in self.operation_reports:
            self.logger.warning(f"Operation {operation_id} not found for completion")
            return
        
        report = self.operation_reports[operation_id]
        report.end_time = time.time()
        report.duration_seconds = report.end_time - report.start_time
        
        # Calculate recovery rate
        recovery_stats = self.recovery_manager.get_recovery_statistics()
        if recovery_stats["total_recoveries"] > 0:
            # This is a simplified calculation - in practice you'd track per-operation
            report.recovery_rate = 0.5  # Placeholder
        
        self.logger.info(
            f"Completed tracking operation {operation_id}",
            operation_id=operation_id,
            duration=report.duration_seconds,
            success_rate=1.0 - report.error_rate,
            error_rate=report.error_rate
        )
    
    def track_entity_failure(self, entity_id: str, entity_type: str, error_record: ErrorRecord):
        """Track a failure for a specific entity."""
        if entity_id not in self.entity_failures:
            self.entity_failures[entity_id] = EntityFailureReport(
                entity_id=entity_id,
                entity_type=entity_type,
                failure_count=0,
                first_failure=error_record.timestamp,
                last_failure=error_record.timestamp,
                error_categories=[],
                error_messages=[],
                recovery_attempts=0,
                final_status="failed"
            )
        
        failure_report = self.entity_failures[entity_id]
        failure_report.failure_count += 1
        failure_report.last_failure = error_record.timestamp
        
        # Track error category if not already present
        category = error_record.error_category.value
        if category not in failure_report.error_categories:
            failure_report.error_categories.append(category)
        
        # Track error message if not already present
        if error_record.message not in failure_report.error_messages:
            failure_report.error_messages.append(error_record.message)
        
        # Track recovery attempts
        if error_record.recovery_attempted:
            failure_report.recovery_attempts += 1
            if error_record.recovery_successful:
                failure_report.final_status = "recovered"
    
    def analyze_error_trends(self, time_window_hours: int = 1) -> List[ErrorTrend]:
        """Analyze error trends over time windows."""
        current_time = time.time()
        window_seconds = time_window_hours * 3600
        
        # Get errors in the time window
        recent_errors = [
            record for record in self.error_handler.error_records
            if current_time - record.timestamp <= window_seconds
        ]
        
        if not recent_errors:
            return []
        
        # Count errors by category
        category_counts = {}
        for record in recent_errors:
            category = record.error_category.value
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Find dominant category
        dominant_category = max(category_counts.items(), key=lambda x: x[1])[0]
        
        # Calculate error rate (simplified)
        error_rate = len(recent_errors) / max(1, time_window_hours)
        
        # Determine trend direction (simplified - would need historical data)
        trend_direction = "stable"  # Placeholder
        
        trend = ErrorTrend(
            time_window=f"{time_window_hours}h",
            error_count=len(recent_errors),
            error_rate=error_rate,
            dominant_category=dominant_category,
            trend_direction=trend_direction
        )
        
        return [trend]
    
    def generate_troubleshooting_recommendations(self) -> List[str]:
        """Generate actionable troubleshooting recommendations."""
        recommendations = []
        error_summary = self.error_handler.get_error_summary()
        
        # Network-related recommendations
        network_errors = error_summary["categories"].get("network", 0)
        if network_errors > 0:
            recommendations.append(
                "Network connectivity issues detected. Check internet connection and firewall settings."
            )
            recommendations.append(
                "Consider increasing retry attempts and timeout values for network operations."
            )
        
        # Authentication recommendations
        auth_errors = error_summary["categories"].get("authentication", 0)
        if auth_errors > 0:
            recommendations.append(
                "Authentication failures detected. Verify API credentials and token validity."
            )
            recommendations.append(
                "Check if API tokens have expired or if permissions have changed."
            )
        
        # Rate limiting recommendations
        rate_limit_errors = error_summary["categories"].get("rate_limit", 0)
        if rate_limit_errors > 0:
            recommendations.append(
                "Rate limiting detected. Reduce batch sizes and increase delays between requests."
            )
            recommendations.append(
                "Consider implementing exponential backoff with jitter for rate-limited operations."
            )
        
        # Validation recommendations
        validation_errors = error_summary["categories"].get("validation", 0)
        if validation_errors > 0:
            recommendations.append(
                "Data validation errors detected. Review entity data formats and required fields."
            )
            recommendations.append(
                "Check API documentation for updated schema requirements."
            )
        
        # High error rate recommendations
        if error_summary["total_errors"] > 100:
            recommendations.append(
                "High error count detected. Consider running in smaller batches to isolate issues."
            )
            recommendations.append(
                "Review system resources and consider scaling up processing capacity."
            )
        
        # Failed entities recommendations
        if error_summary["failed_entities"] > 50:
            recommendations.append(
                "Many entities failed processing. Review entity data quality and format consistency."
            )
            recommendations.append(
                "Consider implementing data validation before processing to catch issues early."
            )
        
        return recommendations
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate a comprehensive error report."""
        current_time = time.time()
        
        # Basic statistics
        error_summary = self.error_handler.get_error_summary()
        recovery_stats = self.recovery_manager.get_recovery_statistics()
        
        # Operation summaries
        operation_summaries = []
        for op_id, report in self.operation_reports.items():
            operation_summaries.append({
                "operation_id": op_id,
                "operation_type": report.operation_type.value,
                "duration_seconds": report.duration_seconds,
                "total_entities": report.total_entities,
                "successful_entities": report.successful_entities,
                "failed_entities": report.failed_entities,
                "skipped_entities": report.skipped_entities,
                "error_rate": report.error_rate,
                "recovery_rate": report.recovery_rate
            })
        
        # Entity failure summaries
        entity_summaries = []
        for entity_id, failure in self.entity_failures.items():
            entity_summaries.append({
                "entity_id": entity_id,
                "entity_type": failure.entity_type,
                "failure_count": failure.failure_count,
                "error_categories": failure.error_categories,
                "recovery_attempts": failure.recovery_attempts,
                "final_status": failure.final_status,
                "first_failure": datetime.fromtimestamp(failure.first_failure).isoformat(),
                "last_failure": datetime.fromtimestamp(failure.last_failure).isoformat()
            })
        
        # Error trends
        trends = self.analyze_error_trends(1)  # 1-hour window
        trend_summaries = []
        for trend in trends:
            trend_summaries.append({
                "time_window": trend.time_window,
                "error_count": trend.error_count,
                "error_rate": trend.error_rate,
                "dominant_category": trend.dominant_category,
                "trend_direction": trend.trend_direction
            })
        
        # Troubleshooting recommendations
        recommendations = self.generate_troubleshooting_recommendations()
        
        return {
            "report_timestamp": datetime.fromtimestamp(current_time).isoformat(),
            "summary": {
                "total_errors": error_summary["total_errors"],
                "error_categories": error_summary["categories"],
                "error_severities": error_summary["severities"],
                "failed_entities": error_summary["failed_entities"],
                "recent_errors": error_summary["recent_errors"]
            },
            "recovery_statistics": recovery_stats,
            "operations": operation_summaries,
            "failed_entities": entity_summaries,
            "error_trends": trend_summaries,
            "troubleshooting_recommendations": recommendations
        }
    
    def export_report(self, report_type: str = "comprehensive") -> str:
        """
        Export error report to file.
        
        Args:
            report_type: Type of report to generate ("comprehensive", "summary", "entities")
            
        Returns:
            Path to exported report file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"error_report_{report_type}_{timestamp}.json"
        filepath = self.reports_dir / filename
        
        if report_type == "comprehensive":
            report_data = self.generate_comprehensive_report()
        elif report_type == "summary":
            report_data = {
                "summary": self.error_handler.get_error_summary(),
                "recovery_stats": self.recovery_manager.get_recovery_statistics(),
                "recommendations": self.generate_troubleshooting_recommendations()
            }
        elif report_type == "entities":
            report_data = {
                "failed_entities": [
                    {
                        "entity_id": entity_id,
                        "entity_type": failure.entity_type,
                        "failure_count": failure.failure_count,
                        "error_categories": failure.error_categories,
                        "final_status": failure.final_status
                    }
                    for entity_id, failure in self.entity_failures.items()
                ]
            }
        else:
            raise ValueError(f"Unknown report type: {report_type}")
        
        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        self.logger.info(f"Exported {report_type} error report to {filepath}")
        return str(filepath)
    
    def get_entity_failure_summary(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed failure summary for a specific entity."""
        if entity_id not in self.entity_failures:
            return None
        
        failure = self.entity_failures[entity_id]
        return {
            "entity_id": failure.entity_id,
            "entity_type": failure.entity_type,
            "failure_count": failure.failure_count,
            "first_failure": datetime.fromtimestamp(failure.first_failure).isoformat(),
            "last_failure": datetime.fromtimestamp(failure.last_failure).isoformat(),
            "error_categories": failure.error_categories,
            "error_messages": failure.error_messages,
            "recovery_attempts": failure.recovery_attempts,
            "final_status": failure.final_status
        }
    
    def get_operation_summary(self, operation_id: str) -> Optional[Dict[str, Any]]:
        """Get summary for a specific operation."""
        if operation_id not in self.operation_reports:
            return None
        
        report = self.operation_reports[operation_id]
        return {
            "operation_id": report.operation_id,
            "operation_type": report.operation_type.value,
            "start_time": datetime.fromtimestamp(report.start_time).isoformat(),
            "end_time": datetime.fromtimestamp(report.end_time).isoformat() if report.end_time else None,
            "duration_seconds": report.duration_seconds,
            "total_entities": report.total_entities,
            "successful_entities": report.successful_entities,
            "failed_entities": report.failed_entities,
            "skipped_entities": report.skipped_entities,
            "error_rate": report.error_rate,
            "recovery_rate": report.recovery_rate,
            "success_rate": 1.0 - report.error_rate
        }
    
    def cleanup_old_reports(self, max_age_days: int = 30):
        """Clean up old report files."""
        cutoff_time = time.time() - (max_age_days * 24 * 3600)
        cleaned_count = 0
        
        for report_file in self.reports_dir.glob("error_report_*.json"):
            try:
                if report_file.stat().st_mtime < cutoff_time:
                    report_file.unlink()
                    cleaned_count += 1
            except Exception as e:
                self.logger.warning(f"Failed to clean up report file {report_file}: {e}")
        
        if cleaned_count > 0:
            self.logger.info(f"Cleaned up {cleaned_count} old report files")