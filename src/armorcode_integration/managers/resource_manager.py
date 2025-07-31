"""
Resource Manager for system resource detection and optimization.

This module provides system resource monitoring and optimization capabilities
for the ArmorCode-Port integration, including CPU detection, memory monitoring,
and dynamic concurrency limit calculation.
"""

import asyncio
import gc
import logging
import os
import platform
import sys
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .logging_manager import get_logging_manager


@dataclass
class SystemResources:
    """System resource information."""
    cpu_count: int
    memory_total_gb: float
    memory_available_gb: float
    memory_usage_percent: float
    platform: str
    python_version: str
    
    @property
    def is_memory_constrained(self) -> bool:
        """Check if system is memory constrained (>80% usage)."""
        return self.memory_usage_percent > 80.0
    
    @property
    def is_low_memory(self) -> bool:
        """Check if system has low memory (<4GB available)."""
        return self.memory_available_gb < 4.0


@dataclass
class ConcurrencyLimits:
    """Calculated concurrency limits based on system resources."""
    max_concurrent_findings: int
    max_concurrent_batches: int
    recommended_batch_size: int
    memory_limit_mb: int
    
    def __post_init__(self):
        """Validate concurrency limits."""
        # Ensure minimum viable limits
        self.max_concurrent_findings = max(1, self.max_concurrent_findings)
        self.max_concurrent_batches = max(1, self.max_concurrent_batches)
        self.recommended_batch_size = max(1, min(20, self.recommended_batch_size))
        self.memory_limit_mb = max(100, self.memory_limit_mb)


class ResourceManager:
    """
    System resource manager for performance optimization.
    
    Provides system resource detection, memory monitoring, and dynamic
    concurrency limit calculation based on available system resources.
    """
    
    def __init__(self):
        """Initialize the ResourceManager."""
        logging_manager = get_logging_manager()
        self.logger = logging_manager.get_logger("ResourceManager")
        
        # Cache system info
        self._system_resources: Optional[SystemResources] = None
        self._last_resource_check = 0
        self._resource_check_interval = 30  # seconds
        
        # Performance tracking
        self._memory_samples = []
        self._max_memory_samples = 100
        
        self.logger.info("ResourceManager initialized")
    
    def get_system_resources(self, force_refresh: bool = False) -> SystemResources:
        """
        Get current system resource information.
        
        Args:
            force_refresh: Force refresh of cached resource info
            
        Returns:
            SystemResources with current system information
        """
        current_time = time.time()
        
        # Use cached info if recent and not forcing refresh
        if (not force_refresh and 
            self._system_resources and 
            current_time - self._last_resource_check < self._resource_check_interval):
            return self._system_resources
        
        # Detect CPU count
        cpu_count = self._get_cpu_count()
        
        # Get memory information
        memory_info = self._get_memory_info()
        
        # Get platform information
        platform_info = f"{platform.system()} {platform.release()}"
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        
        self._system_resources = SystemResources(
            cpu_count=cpu_count,
            memory_total_gb=memory_info[0],
            memory_available_gb=memory_info[1],
            memory_usage_percent=memory_info[2],
            platform=platform_info,
            python_version=python_version
        )
        
        self._last_resource_check = current_time
        
        self.logger.info(
            "System resources detected",
            cpu_count=cpu_count,
            memory_total_gb=round(memory_info[0], 2),
            memory_available_gb=round(memory_info[1], 2),
            memory_usage_percent=round(memory_info[2], 1),
            platform=platform_info,
            python_version=python_version
        )
        
        return self._system_resources
    
    def _get_cpu_count(self) -> int:
        """Get the number of CPU cores available."""
        try:
            # Try os.cpu_count() first (available in Python 3.4+)
            cpu_count = os.cpu_count()
            if cpu_count is not None:
                return cpu_count
        except AttributeError:
            pass
        
        try:
            # Fallback to multiprocessing
            import multiprocessing
            return multiprocessing.cpu_count()
        except (ImportError, NotImplementedError):
            pass
        
        # Final fallback
        self.logger.warning("Could not detect CPU count, defaulting to 2")
        return 2
    
    def _get_memory_info(self) -> Tuple[float, float, float]:
        """
        Get memory information (total, available, usage percentage).
        
        Returns:
            Tuple of (total_gb, available_gb, usage_percent)
        """
        try:
            # Try to use psutil if available
            import psutil
            memory = psutil.virtual_memory()
            total_gb = memory.total / (1024**3)
            available_gb = memory.available / (1024**3)
            usage_percent = memory.percent
            return total_gb, available_gb, usage_percent
        except ImportError:
            pass
        
        try:
            # Fallback for Unix-like systems
            if hasattr(os, 'sysconf') and 'SC_PAGE_SIZE' in os.sysconf_names and 'SC_PHYS_PAGES' in os.sysconf_names:
                page_size = os.sysconf('SC_PAGE_SIZE')
                total_pages = os.sysconf('SC_PHYS_PAGES')
                total_bytes = page_size * total_pages
                total_gb = total_bytes / (1024**3)
                
                # Estimate available memory (rough approximation)
                # This is not accurate but provides a baseline
                available_gb = total_gb * 0.7  # Assume 70% available
                usage_percent = 30.0  # Assume 30% usage
                
                return total_gb, available_gb, usage_percent
        except (AttributeError, OSError):
            pass
        
        # Final fallback - conservative estimates
        self.logger.warning("Could not detect memory info, using conservative estimates")
        return 8.0, 4.0, 50.0  # 8GB total, 4GB available, 50% usage
    
    def calculate_concurrency_limits(self, 
                                   dataset_size: int = 1000,
                                   entity_size_estimate: int = 2048) -> ConcurrencyLimits:
        """
        Calculate optimal concurrency limits based on system resources.
        
        Args:
            dataset_size: Estimated number of entities to process
            entity_size_estimate: Estimated size per entity in bytes
            
        Returns:
            ConcurrencyLimits with optimized values
        """
        resources = self.get_system_resources()
        
        # Base concurrency on CPU count
        base_concurrency = resources.cpu_count
        
        # Adjust for memory constraints
        if resources.is_memory_constrained:
            # Reduce concurrency if memory is constrained
            max_concurrent_findings = max(1, base_concurrency // 2)
            max_concurrent_batches = max(1, base_concurrency // 4)
            memory_limit_mb = int(resources.memory_available_gb * 1024 * 0.3)  # Use 30% of available
        elif resources.is_low_memory:
            # Conservative limits for low memory systems
            max_concurrent_findings = max(2, base_concurrency)
            max_concurrent_batches = max(1, base_concurrency // 2)
            memory_limit_mb = int(resources.memory_available_gb * 1024 * 0.5)  # Use 50% of available
        else:
            # Aggressive limits for well-resourced systems
            max_concurrent_findings = base_concurrency * 2
            max_concurrent_batches = base_concurrency
            memory_limit_mb = int(resources.memory_available_gb * 1024 * 0.7)  # Use 70% of available
        
        # Calculate recommended batch size based on memory and entity size
        estimated_memory_per_entity = entity_size_estimate * 3  # Account for processing overhead
        max_entities_in_memory = (memory_limit_mb * 1024 * 1024) // estimated_memory_per_entity
        
        # Batch size should allow for concurrent processing without memory issues
        recommended_batch_size = min(
            20,  # Port API limit
            max(1, max_entities_in_memory // max_concurrent_findings),
            max(5, dataset_size // 100)  # At least 5, or 1% of dataset
        )
        
        limits = ConcurrencyLimits(
            max_concurrent_findings=max_concurrent_findings,
            max_concurrent_batches=max_concurrent_batches,
            recommended_batch_size=recommended_batch_size,
            memory_limit_mb=memory_limit_mb
        )
        
        self.logger.info(
            "Calculated concurrency limits",
            max_concurrent_findings=limits.max_concurrent_findings,
            max_concurrent_batches=limits.max_concurrent_batches,
            recommended_batch_size=limits.recommended_batch_size,
            memory_limit_mb=limits.memory_limit_mb,
            dataset_size=dataset_size,
            entity_size_estimate=entity_size_estimate,
            cpu_count=resources.cpu_count,
            memory_available_gb=round(resources.memory_available_gb, 2),
            memory_constrained=resources.is_memory_constrained
        )
        
        return limits
    
    def monitor_memory_usage(self) -> Dict[str, float]:
        """
        Monitor current memory usage and return metrics.
        
        Returns:
            Dictionary with memory usage metrics
        """
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            metrics = {
                'rss_mb': memory_info.rss / (1024 * 1024),
                'vms_mb': memory_info.vms / (1024 * 1024),
                'percent': process.memory_percent(),
                'available_system_mb': psutil.virtual_memory().available / (1024 * 1024)
            }
            
            # Track memory samples for trend analysis
            self._memory_samples.append({
                'timestamp': time.time(),
                'rss_mb': metrics['rss_mb'],
                'percent': metrics['percent']
            })
            
            # Keep only recent samples
            if len(self._memory_samples) > self._max_memory_samples:
                self._memory_samples = self._memory_samples[-self._max_memory_samples:]
            
            return metrics
            
        except ImportError:
            # Fallback without psutil
            return {
                'rss_mb': 0.0,
                'vms_mb': 0.0,
                'percent': 0.0,
                'available_system_mb': 0.0
            }
    
    def get_memory_trend(self) -> Dict[str, float]:
        """
        Get memory usage trend analysis.
        
        Returns:
            Dictionary with trend metrics
        """
        if len(self._memory_samples) < 2:
            return {
                'trend': 0.0,
                'peak_mb': 0.0,
                'average_mb': 0.0,
                'samples': len(self._memory_samples)
            }
        
        recent_samples = self._memory_samples[-10:]  # Last 10 samples
        rss_values = [sample['rss_mb'] for sample in recent_samples]
        
        # Calculate trend (simple linear regression slope)
        n = len(rss_values)
        x_values = list(range(n))
        x_mean = sum(x_values) / n
        y_mean = sum(rss_values) / n
        
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, rss_values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)
        
        trend = numerator / denominator if denominator != 0 else 0.0
        
        return {
            'trend': trend,  # MB per sample (positive = increasing)
            'peak_mb': max(rss_values),
            'average_mb': sum(rss_values) / len(rss_values),
            'samples': len(self._memory_samples)
        }
    
    def suggest_gc_collection(self) -> bool:
        """
        Suggest whether garbage collection should be triggered.
        
        Returns:
            True if GC is recommended, False otherwise
        """
        memory_metrics = self.monitor_memory_usage()
        memory_trend = self.get_memory_trend()
        
        # Suggest GC if:
        # 1. Memory usage is high (>70% of system)
        # 2. Memory trend is increasing rapidly (>10MB per sample)
        # 3. Available system memory is low (<1GB)
        
        should_gc = (
            memory_metrics.get('percent', 0) > 70 or
            memory_trend.get('trend', 0) > 10 or
            memory_metrics.get('available_system_mb', float('inf')) < 1024
        )
        
        if should_gc:
            self.logger.info(
                "Garbage collection recommended",
                memory_percent=memory_metrics.get('percent', 0),
                memory_trend=memory_trend.get('trend', 0),
                available_mb=memory_metrics.get('available_system_mb', 0),
                rss_mb=memory_metrics.get('rss_mb', 0)
            )
        
        return should_gc
    
    def trigger_gc_collection(self) -> Dict[str, int]:
        """
        Trigger garbage collection and return statistics.
        
        Returns:
            Dictionary with GC statistics
        """
        self.logger.debug("Triggering garbage collection")
        
        # Get memory before GC
        memory_before = self.monitor_memory_usage()
        
        # Trigger GC
        collected = gc.collect()
        
        # Get memory after GC
        memory_after = self.monitor_memory_usage()
        
        # Calculate freed memory
        freed_mb = memory_before.get('rss_mb', 0) - memory_after.get('rss_mb', 0)
        
        stats = {
            'objects_collected': collected,
            'memory_freed_mb': max(0, freed_mb),
            'memory_before_mb': memory_before.get('rss_mb', 0),
            'memory_after_mb': memory_after.get('rss_mb', 0)
        }
        
        self.logger.info(
            "Garbage collection completed",
            **stats
        )
        
        return stats
    
    def optimize_for_large_dataset(self, dataset_size: int) -> Dict[str, any]:
        """
        Provide optimization recommendations for large datasets.
        
        Args:
            dataset_size: Size of the dataset to process
            
        Returns:
            Dictionary with optimization recommendations
        """
        resources = self.get_system_resources()
        limits = self.calculate_concurrency_limits(dataset_size)
        
        # Determine if dataset is "large" based on system resources
        is_large_dataset = dataset_size > (resources.cpu_count * 1000)
        
        recommendations = {
            'use_streaming': is_large_dataset,
            'enable_gc_hints': dataset_size > 5000,
            'chunk_size': min(1000, max(100, dataset_size // 20)),
            'memory_monitoring_interval': 30 if is_large_dataset else 60,
            'concurrency_limits': limits,
            'estimated_memory_mb': (dataset_size * 2048 * 3) // (1024 * 1024),  # Rough estimate
            'processing_strategy': 'streaming' if is_large_dataset else 'batch'
        }
        
        self.logger.info(
            "Generated optimization recommendations",
            dataset_size=dataset_size,
            is_large_dataset=is_large_dataset,
            **{k: v for k, v in recommendations.items() if k != 'concurrency_limits'}
        )
        
        return recommendations
    
    async def monitor_performance_async(self, 
                                      interval: int = 30,
                                      callback: Optional[callable] = None) -> None:
        """
        Asynchronously monitor system performance.
        
        Args:
            interval: Monitoring interval in seconds
            callback: Optional callback function for performance data
        """
        self.logger.info(f"Starting async performance monitoring (interval: {interval}s)")
        
        while True:
            try:
                memory_metrics = self.monitor_memory_usage()
                memory_trend = self.get_memory_trend()
                
                performance_data = {
                    'timestamp': time.time(),
                    'memory_metrics': memory_metrics,
                    'memory_trend': memory_trend,
                    'gc_recommended': self.suggest_gc_collection()
                }
                
                if callback:
                    await callback(performance_data)
                
                # Auto-trigger GC if recommended
                if performance_data['gc_recommended']:
                    self.trigger_gc_collection()
                
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                self.logger.info("Performance monitoring cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(interval)