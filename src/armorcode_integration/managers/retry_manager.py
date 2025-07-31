#!/usr/bin/env python3
"""
Retry Manager for ArmorCode Integration

This module provides robust retry logic with exponential backoff and error classification
for handling API failures in the ArmorCode-Port integration.

Usage:
    from .retry_manager import RetryManager
    
    retry_manager = RetryManager(max_attempts=3)
    result = await retry_manager.with_retry(some_async_function, arg1, arg2)
"""

import asyncio
import logging
import time
from typing import Any, Callable, Optional, Union, Dict
from dataclasses import dataclass
from enum import Enum
import aiohttp


class ErrorType(Enum):
    """Classification of error types for retry logic."""
    RETRYABLE = "retryable"
    NON_RETRYABLE = "non_retryable"
    RATE_LIMITED = "rate_limited"


@dataclass
class RetryAttempt:
    """Information about a retry attempt."""
    attempt_number: int
    error: Exception
    backoff_delay: float
    timestamp: float


@dataclass
class RetryResult:
    """Result of a retry operation."""
    success: bool
    result: Any = None
    error: Optional[Exception] = None
    attempts: list[RetryAttempt] = None
    total_attempts: int = 0
    total_time: float = 0.0


class RetryManager:
    """
    Manages retry logic with exponential backoff and error classification.
    
    Implements requirements 3.1-3.5:
    - Network errors: Retry up to max_attempts with exponential backoff
    - Rate limiting: Respect Retry-After header
    - Authentication errors: Fail immediately (no retry)
    - Max attempts reached: Log failure and continue
    - Successful retry: Log recovery
    """
    
    def __init__(self, 
                 max_attempts: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 backoff_multiplier: float = 2.0,
                 jitter: bool = True):
        """
        Initialize the RetryManager.
        
        Args:
            max_attempts: Maximum number of retry attempts (default: 3)
            base_delay: Base delay in seconds for exponential backoff (default: 1.0)
            max_delay: Maximum delay in seconds (default: 60.0)
            backoff_multiplier: Multiplier for exponential backoff (default: 2.0)
            jitter: Whether to add random jitter to delays (default: True)
        """
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_multiplier = backoff_multiplier
        self.jitter = jitter
        self.logger = logging.getLogger(__name__)
        
    def classify_error(self, error: Exception) -> ErrorType:
        """
        Classify an error to determine if it should be retried.
        
        Args:
            error: The exception to classify
            
        Returns:
            ErrorType indicating how the error should be handled
        """
        # Handle aiohttp client errors
        if isinstance(error, aiohttp.ClientError):
            if isinstance(error, aiohttp.ClientResponseError):
                status_code = error.status
                
                # Authentication errors - don't retry (requirement 3.3)
                if status_code in (401, 403):
                    return ErrorType.NON_RETRYABLE
                    
                # Rate limiting - special handling (requirement 3.2)
                if status_code == 429:
                    return ErrorType.RATE_LIMITED
                    
                # Server errors - retry
                if 500 <= status_code < 600:
                    return ErrorType.RETRYABLE
                    
                # Client errors (except auth) - don't retry
                if 400 <= status_code < 500:
                    return ErrorType.NON_RETRYABLE
                    
            # Network-level errors - retry (requirement 3.1)
            elif isinstance(error, (aiohttp.ClientConnectionError, 
                                  aiohttp.ClientConnectorError)):
                return ErrorType.RETRYABLE
            # Handle timeout errors separately since ClientTimeout is not an exception
            elif isinstance(error, (aiohttp.ServerTimeoutError, aiohttp.ClientTimeout)):
                return ErrorType.RETRYABLE
                
        # Handle mock errors that have status attribute (for testing)
        elif hasattr(error, 'status'):
            status_code = error.status
            if status_code in (401, 403):
                return ErrorType.NON_RETRYABLE
            elif status_code == 429:
                return ErrorType.RATE_LIMITED
            elif 500 <= status_code < 600:
                return ErrorType.RETRYABLE
            elif 400 <= status_code < 500:
                return ErrorType.NON_RETRYABLE
                
        # Handle requests library errors (for backward compatibility)
        elif hasattr(error, 'response') and hasattr(error.response, 'status_code'):
            status_code = error.response.status_code
            
            # Authentication errors
            if status_code in (401, 403):
                return ErrorType.NON_RETRYABLE
                
            # Rate limiting
            if status_code == 429:
                return ErrorType.RATE_LIMITED
                
            # Server errors
            if 500 <= status_code < 600:
                return ErrorType.RETRYABLE
                
            # Client errors (except auth)
            if 400 <= status_code < 500:
                return ErrorType.NON_RETRYABLE
                
        # Network-level errors from requests
        elif error.__class__.__name__ in ('ConnectionError', 'Timeout', 'ConnectTimeout'):
            return ErrorType.RETRYABLE
            
        # Custom ArmorCode API errors
        elif hasattr(error, 'status_code'):
            status_code = error.status_code
            if status_code in (401, 403):
                return ErrorType.NON_RETRYABLE
            elif status_code == 429:
                return ErrorType.RATE_LIMITED
            elif status_code and 500 <= status_code < 600:
                return ErrorType.RETRYABLE
            elif status_code and 400 <= status_code < 500:
                return ErrorType.NON_RETRYABLE
                
        # Default: don't retry unknown errors
        return ErrorType.NON_RETRYABLE
        
    def calculate_backoff_delay(self, attempt: int) -> float:
        """
        Calculate the delay for exponential backoff.
        
        Args:
            attempt: The attempt number (0-based)
            
        Returns:
            Delay in seconds
        """
        # Calculate exponential backoff
        delay = self.base_delay * (self.backoff_multiplier ** attempt)
        
        # Apply maximum delay limit
        delay = min(delay, self.max_delay)
        
        # Add jitter to prevent thundering herd
        if self.jitter:
            import random
            jitter_factor = random.uniform(0.5, 1.5)
            delay *= jitter_factor
            
        return delay
        
    async def handle_rate_limit(self, error: Exception) -> float:
        """
        Handle rate limiting by extracting Retry-After header.
        
        Args:
            error: The rate limiting error
            
        Returns:
            Delay in seconds to wait before retrying
        """
        retry_after = 60  # Default fallback
        
        # Try to extract Retry-After header from different error types
        if hasattr(error, 'headers') and error.headers:
            retry_after_header = error.headers.get('Retry-After')
            if retry_after_header:
                try:
                    retry_after = int(retry_after_header)
                except (ValueError, TypeError):
                    self.logger.warning(f"Invalid Retry-After header: {retry_after_header}")
                    
        elif hasattr(error, 'response') and hasattr(error.response, 'headers'):
            retry_after_header = error.response.headers.get('Retry-After')
            if retry_after_header:
                try:
                    retry_after = int(retry_after_header)
                except (ValueError, TypeError):
                    self.logger.warning(f"Invalid Retry-After header: {retry_after_header}")
                    
        # Cap the retry delay to our maximum
        retry_after = min(retry_after, self.max_delay)
        
        self.logger.info(f"Rate limited, waiting {retry_after}s before retry")
        return retry_after
        
    async def with_retry(self, 
                        func: Callable,
                        *args,
                        **kwargs) -> Any:
        """
        Execute a function with retry logic.
        
        Args:
            func: The async function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            The result of the function call
            
        Raises:
            The last exception if all retry attempts fail
        """
        attempts = []
        start_time = time.time()
        last_error = None
        
        for attempt in range(self.max_attempts + 1):  # +1 for initial attempt
            try:
                # Log attempt (except for first attempt)
                if attempt > 0:
                    self.logger.info(f"Retry attempt {attempt}/{self.max_attempts}")
                    
                # Execute the function
                result = await func(*args, **kwargs)
                
                # Success! Log recovery if this was a retry (requirement 3.5)
                if attempt > 0:
                    total_time = time.time() - start_time
                    self.logger.info(f"Operation succeeded after {attempt} retries "
                                   f"(total time: {total_time:.2f}s)")
                    
                return result
                
            except Exception as error:
                last_error = error
                error_type = self.classify_error(error)
                
                # Log the error (handle cases where str(error) might fail)
                try:
                    error_str = str(error)
                except Exception:
                    error_str = f"{error.__class__.__name__}: <error string failed>"
                self.logger.warning(f"Attempt {attempt + 1} failed: {error_str}")
                
                # Record this attempt
                attempt_info = RetryAttempt(
                    attempt_number=attempt + 1,
                    error=error,
                    backoff_delay=0.0,
                    timestamp=time.time()
                )
                attempts.append(attempt_info)
                
                # Check if we should retry
                if attempt >= self.max_attempts:
                    # Max attempts reached (requirement 3.4)
                    total_time = time.time() - start_time
                    self.logger.error(f"Operation failed after {self.max_attempts + 1} attempts "
                                    f"(total time: {total_time:.2f}s): {error}")
                    raise error
                    
                if error_type == ErrorType.NON_RETRYABLE:
                    # Don't retry authentication errors, etc. (requirement 3.3)
                    self.logger.error(f"Non-retryable error, failing immediately: {error}")
                    raise error
                    
                # Calculate delay for next attempt
                if error_type == ErrorType.RATE_LIMITED:
                    # Handle rate limiting (requirement 3.2)
                    delay = await self.handle_rate_limit(error)
                else:
                    # Use exponential backoff for retryable errors (requirement 3.1)
                    delay = self.calculate_backoff_delay(attempt)
                    
                attempt_info.backoff_delay = delay
                
                # Wait before retrying
                if delay > 0:
                    self.logger.info(f"Waiting {delay:.2f}s before retry")
                    await asyncio.sleep(delay)
                    
        # This should never be reached due to the loop logic above
        raise last_error or Exception("Unexpected retry loop exit")
        
    def is_retryable_error(self, error: Exception) -> bool:
        """
        Check if an error is retryable.
        
        Args:
            error: The exception to check
            
        Returns:
            True if the error should be retried, False otherwise
        """
        error_type = self.classify_error(error)
        return error_type in (ErrorType.RETRYABLE, ErrorType.RATE_LIMITED)
        
    async def execute_with_stats(self,
                                func: Callable,
                                *args,
                                **kwargs) -> RetryResult:
        """
        Execute a function with retry logic and return detailed statistics.
        
        Args:
            func: The async function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            RetryResult with detailed information about the operation
        """
        attempts = []
        start_time = time.time()
        
        try:
            result = await self.with_retry(func, *args, **kwargs)
            total_time = time.time() - start_time
            
            return RetryResult(
                success=True,
                result=result,
                attempts=attempts,
                total_attempts=len(attempts),
                total_time=total_time
            )
            
        except Exception as error:
            total_time = time.time() - start_time
            
            return RetryResult(
                success=False,
                error=error,
                attempts=attempts,
                total_attempts=len(attempts),
                total_time=total_time
            )


# Convenience function for simple retry operations
async def retry_async(func: Callable, 
                     *args, 
                     max_attempts: int = 3,
                     base_delay: float = 1.0,
                     **kwargs) -> Any:
    """
    Async retry wrapper function with exponential backoff.
    
    This is a convenience function that creates a RetryManager instance
    and executes the function with retry logic.
    
    Args:
        func: The async function to execute
        *args: Positional arguments for the function
        max_attempts: Maximum number of retry attempts (default: 3)
        base_delay: Base delay in seconds for exponential backoff (default: 1.0)
        **kwargs: Keyword arguments for the function
        
    Returns:
        The result of the function call
        
    Raises:
        The last exception if all retry attempts fail
        
    Example:
        result = await retry_async(api_call, "param1", param2="value")
    """
    retry_manager = RetryManager(max_attempts=max_attempts, base_delay=base_delay)
    return await retry_manager.with_retry(func, *args, **kwargs)