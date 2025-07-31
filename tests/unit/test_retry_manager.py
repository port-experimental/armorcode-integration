#!/usr/bin/env python3
"""
Unit tests for RetryManager

Tests the retry logic, error classification, and backoff timing
according to requirements 3.1-3.5.
"""

import asyncio
import pytest
import time
from unittest.mock import Mock, AsyncMock, patch
import aiohttp
from retry_manager import RetryManager, ErrorType, RetryResult


class MockResponse:
    """Mock HTTP response for testing."""
    def __init__(self, status_code: int, headers: dict = None):
        self.status_code = status_code
        self.status = status_code
        self.headers = headers or {}


class MockClientResponseError(Exception):
    """Mock aiohttp ClientResponseError for testing."""
    def __init__(self, status: int, headers: dict = None):
        self.status = status
        self.headers = headers or {}
        super().__init__(f"HTTP {status}")
        
    def __str__(self):
        return f"HTTP {self.status}"


class MockArmorCodeAPIError(Exception):
    """Mock ArmorCode API error for testing."""
    def __init__(self, message: str, status_code: int = None):
        super().__init__(message)
        self.status_code = status_code


@pytest.fixture
def retry_manager():
    """Create a RetryManager instance for testing."""
    return RetryManager(max_attempts=3, base_delay=0.1, jitter=False)


class TestErrorClassification:
    """Test error classification logic."""
    
    def test_classify_authentication_error_aiohttp(self, retry_manager):
        """Test that authentication errors are classified as non-retryable."""
        error_401 = MockClientResponseError(401)
        error_403 = MockClientResponseError(403)
        
        assert retry_manager.classify_error(error_401) == ErrorType.NON_RETRYABLE
        assert retry_manager.classify_error(error_403) == ErrorType.NON_RETRYABLE
        
    def test_classify_rate_limit_error_aiohttp(self, retry_manager):
        """Test that rate limit errors are classified correctly."""
        error_429 = MockClientResponseError(429, {'Retry-After': '60'})
        
        assert retry_manager.classify_error(error_429) == ErrorType.RATE_LIMITED
        
    def test_classify_server_error_aiohttp(self, retry_manager):
        """Test that server errors are classified as retryable."""
        error_500 = MockClientResponseError(500)
        error_502 = MockClientResponseError(502)
        error_503 = MockClientResponseError(503)
        
        assert retry_manager.classify_error(error_500) == ErrorType.RETRYABLE
        assert retry_manager.classify_error(error_502) == ErrorType.RETRYABLE
        assert retry_manager.classify_error(error_503) == ErrorType.RETRYABLE
        
    def test_classify_client_error_aiohttp(self, retry_manager):
        """Test that client errors (except auth) are non-retryable."""
        error_400 = MockClientResponseError(400)
        error_404 = MockClientResponseError(404)
        
        assert retry_manager.classify_error(error_400) == ErrorType.NON_RETRYABLE
        assert retry_manager.classify_error(error_404) == ErrorType.NON_RETRYABLE
        
    def test_classify_network_errors_aiohttp(self, retry_manager):
        """Test that network errors are classified as retryable."""
        connection_error = aiohttp.ClientConnectionError()
        
        # Test with actual timeout exception that aiohttp uses
        timeout_error = aiohttp.ServerTimeoutError("Request timeout")
        
        # Create a mock connector error with proper initialization
        class MockConnectorError(aiohttp.ClientConnectorError):
            def __init__(self):
                self._conn_key = None
                self._os_error = OSError("Connection failed")
                Exception.__init__(self, "Connection failed")
        connector_error = MockConnectorError()
        
        assert retry_manager.classify_error(connection_error) == ErrorType.RETRYABLE
        assert retry_manager.classify_error(timeout_error) == ErrorType.RETRYABLE
        assert retry_manager.classify_error(connector_error) == ErrorType.RETRYABLE
        
    def test_classify_custom_api_error(self, retry_manager):
        """Test classification of custom ArmorCode API errors."""
        auth_error = MockArmorCodeAPIError("Unauthorized", 401)
        rate_limit_error = MockArmorCodeAPIError("Rate limited", 429)
        server_error = MockArmorCodeAPIError("Internal error", 500)
        
        assert retry_manager.classify_error(auth_error) == ErrorType.NON_RETRYABLE
        assert retry_manager.classify_error(rate_limit_error) == ErrorType.RATE_LIMITED
        assert retry_manager.classify_error(server_error) == ErrorType.RETRYABLE
        
    def test_classify_unknown_error(self, retry_manager):
        """Test that unknown errors are classified as non-retryable."""
        unknown_error = ValueError("Some random error")
        
        assert retry_manager.classify_error(unknown_error) == ErrorType.NON_RETRYABLE


class TestBackoffCalculation:
    """Test exponential backoff calculation."""
    
    def test_exponential_backoff_progression(self, retry_manager):
        """Test that backoff delays increase exponentially."""
        delay_0 = retry_manager.calculate_backoff_delay(0)
        delay_1 = retry_manager.calculate_backoff_delay(1)
        delay_2 = retry_manager.calculate_backoff_delay(2)
        
        # With base_delay=0.1 and multiplier=2.0
        assert delay_0 == 0.1  # 0.1 * 2^0
        assert delay_1 == 0.2  # 0.1 * 2^1
        assert delay_2 == 0.4  # 0.1 * 2^2
        
    def test_max_delay_limit(self):
        """Test that delays are capped at max_delay."""
        retry_manager = RetryManager(base_delay=10.0, max_delay=30.0, jitter=False)
        
        delay_5 = retry_manager.calculate_backoff_delay(5)  # Would be 10 * 2^5 = 320
        
        assert delay_5 == 30.0  # Capped at max_delay
        
    def test_jitter_variation(self):
        """Test that jitter adds randomness to delays."""
        retry_manager = RetryManager(base_delay=1.0, jitter=True)
        
        delays = [retry_manager.calculate_backoff_delay(1) for _ in range(10)]
        
        # All delays should be different due to jitter
        assert len(set(delays)) > 1
        # All delays should be within reasonable bounds (0.5x to 1.5x base)
        for delay in delays:
            assert 1.0 <= delay <= 3.0  # 2.0 * 1.5 = 3.0


class TestRateLimitHandling:
    """Test rate limit handling with Retry-After header."""
    
    @pytest.mark.asyncio
    async def test_extract_retry_after_from_aiohttp_error(self, retry_manager):
        """Test extracting Retry-After from aiohttp error."""
        error = MockClientResponseError(429, {'Retry-After': '45'})
        
        delay = await retry_manager.handle_rate_limit(error)
        
        assert delay == 45
        
    @pytest.mark.asyncio
    async def test_extract_retry_after_from_requests_error(self, retry_manager):
        """Test extracting Retry-After from requests-style error."""
        mock_response = MockResponse(429, {'Retry-After': '30'})
        error = Exception("Rate limited")
        error.response = mock_response
        
        delay = await retry_manager.handle_rate_limit(error)
        
        assert delay == 30
        
    @pytest.mark.asyncio
    async def test_invalid_retry_after_header(self, retry_manager):
        """Test handling of invalid Retry-After header."""
        error = MockClientResponseError(429, {'Retry-After': 'invalid'})
        
        delay = await retry_manager.handle_rate_limit(error)
        
        assert delay == 60  # Default fallback
        
    @pytest.mark.asyncio
    async def test_missing_retry_after_header(self, retry_manager):
        """Test handling when Retry-After header is missing."""
        error = MockClientResponseError(429)
        
        delay = await retry_manager.handle_rate_limit(error)
        
        assert delay == 60  # Default fallback
        
    @pytest.mark.asyncio
    async def test_retry_after_capped_at_max_delay(self):
        """Test that Retry-After is capped at max_delay."""
        retry_manager = RetryManager(max_delay=30.0)
        error = MockClientResponseError(429, {'Retry-After': '120'})
        
        delay = await retry_manager.handle_rate_limit(error)
        
        assert delay == 30.0  # Capped at max_delay


class TestRetryLogic:
    """Test the main retry logic."""
    
    @pytest.mark.asyncio
    async def test_successful_first_attempt(self, retry_manager):
        """Test that successful operations don't retry."""
        mock_func = AsyncMock(return_value="success")
        
        result = await retry_manager.with_retry(mock_func, "arg1", kwarg1="value1")
        
        assert result == "success"
        assert mock_func.call_count == 1
        mock_func.assert_called_with("arg1", kwarg1="value1")
        
    @pytest.mark.asyncio
    async def test_retry_on_network_error(self, retry_manager):
        """Test retry behavior on network errors."""
        mock_func = AsyncMock()
        mock_func.side_effect = [
            aiohttp.ClientConnectionError(),
            aiohttp.ClientConnectionError(),
            "success"
        ]
        
        start_time = time.time()
        result = await retry_manager.with_retry(mock_func)
        end_time = time.time()
        
        assert result == "success"
        assert mock_func.call_count == 3
        # Should have waited for backoff delays
        assert end_time - start_time >= 0.3  # 0.1 + 0.2 = 0.3 seconds minimum
        
    @pytest.mark.asyncio
    async def test_no_retry_on_auth_error(self, retry_manager):
        """Test that authentication errors are not retried."""
        mock_func = AsyncMock()
        mock_func.side_effect = MockClientResponseError(401)
        
        with pytest.raises(MockClientResponseError):
            await retry_manager.with_retry(mock_func)
            
        assert mock_func.call_count == 1  # No retries
        
    @pytest.mark.asyncio
    async def test_rate_limit_retry_with_delay(self, retry_manager):
        """Test retry behavior on rate limit errors."""
        mock_func = AsyncMock()
        mock_func.side_effect = [
            MockClientResponseError(429, {'Retry-After': '1'}),
            "success"
        ]
        
        start_time = time.time()
        result = await retry_manager.with_retry(mock_func)
        end_time = time.time()
        
        assert result == "success"
        assert mock_func.call_count == 2
        # Should have waited for Retry-After delay
        assert end_time - start_time >= 1.0
        
    @pytest.mark.asyncio
    async def test_max_attempts_exceeded(self, retry_manager):
        """Test behavior when max attempts are exceeded."""
        mock_func = AsyncMock()
        mock_func.side_effect = aiohttp.ClientConnectionError()
        
        with pytest.raises(aiohttp.ClientConnectionError):
            await retry_manager.with_retry(mock_func)
            
        assert mock_func.call_count == 4  # Initial + 3 retries
        
    @pytest.mark.asyncio
    async def test_mixed_error_types(self, retry_manager):
        """Test handling of mixed error types."""
        mock_func = AsyncMock()
        mock_func.side_effect = [
            aiohttp.ClientConnectionError(),  # Retryable
            MockClientResponseError(500),     # Retryable
            MockClientResponseError(429, {'Retry-After': '0.1'}),  # Rate limited
            "success"
        ]
        
        result = await retry_manager.with_retry(mock_func)
        
        assert result == "success"
        assert mock_func.call_count == 4


class TestRetryStatistics:
    """Test retry statistics and detailed results."""
    
    @pytest.mark.asyncio
    async def test_execute_with_stats_success(self, retry_manager):
        """Test statistics collection for successful operations."""
        mock_func = AsyncMock(return_value="success")
        
        result = await retry_manager.execute_with_stats(mock_func)
        
        assert isinstance(result, RetryResult)
        assert result.success is True
        assert result.result == "success"
        assert result.error is None
        assert result.total_attempts == 0  # No retries needed
        assert result.total_time > 0
        
    @pytest.mark.asyncio
    async def test_execute_with_stats_failure(self, retry_manager):
        """Test statistics collection for failed operations."""
        mock_func = AsyncMock()
        mock_func.side_effect = MockClientResponseError(401)
        
        result = await retry_manager.execute_with_stats(mock_func)
        
        assert isinstance(result, RetryResult)
        assert result.success is False
        assert result.result is None
        assert isinstance(result.error, MockClientResponseError)
        assert result.total_time > 0
        
    @pytest.mark.asyncio
    async def test_is_retryable_error_method(self, retry_manager):
        """Test the is_retryable_error convenience method."""
        retryable_error = aiohttp.ClientConnectionError()
        non_retryable_error = MockClientResponseError(401)
        rate_limit_error = MockClientResponseError(429)
        
        assert retry_manager.is_retryable_error(retryable_error) is True
        assert retry_manager.is_retryable_error(non_retryable_error) is False
        assert retry_manager.is_retryable_error(rate_limit_error) is True


class TestIntegration:
    """Integration tests with realistic scenarios."""
    
    @pytest.mark.asyncio
    async def test_realistic_api_call_scenario(self, retry_manager):
        """Test a realistic API call scenario with various failures."""
        call_count = 0
        
        async def mock_api_call():
            nonlocal call_count
            call_count += 1
            
            if call_count == 1:
                # First call: network error
                raise aiohttp.ClientConnectionError()
            elif call_count == 2:
                # Second call: server error
                raise MockClientResponseError(503)
            elif call_count == 3:
                # Third call: rate limited
                raise MockClientResponseError(429, {'Retry-After': '0.1'})
            else:
                # Fourth call: success
                return {"data": "api_response"}
                
        result = await retry_manager.with_retry(mock_api_call)
        
        assert result == {"data": "api_response"}
        assert call_count == 4
        
    @pytest.mark.asyncio
    async def test_configuration_flexibility(self):
        """Test that RetryManager can be configured for different scenarios."""
        # High-frequency, low-latency scenario
        fast_retry = RetryManager(max_attempts=5, base_delay=0.05, max_delay=1.0)
        
        # Low-frequency, high-latency scenario  
        slow_retry = RetryManager(max_attempts=2, base_delay=2.0, max_delay=30.0)
        
        # Test that configurations work
        assert fast_retry.max_attempts == 5
        assert fast_retry.base_delay == 0.05
        assert slow_retry.max_attempts == 2
        assert slow_retry.base_delay == 2.0


class TestConvenienceFunction:
    """Test the retry_async convenience function."""
    
    @pytest.mark.asyncio
    async def test_retry_async_success(self):
        """Test the retry_async convenience function with success."""
        from retry_manager import retry_async
        
        mock_func = AsyncMock(return_value="success")
        
        result = await retry_async(mock_func, "arg1", kwarg1="value1")
        
        assert result == "success"
        assert mock_func.call_count == 1
        mock_func.assert_called_with("arg1", kwarg1="value1")
        
    @pytest.mark.asyncio
    async def test_retry_async_with_retries(self):
        """Test the retry_async convenience function with retries."""
        from retry_manager import retry_async
        
        mock_func = AsyncMock()
        mock_func.side_effect = [
            aiohttp.ClientConnectionError(),
            "success"
        ]
        
        result = await retry_async(mock_func, max_attempts=2, base_delay=0.1)
        
        assert result == "success"
        assert mock_func.call_count == 2
        
    @pytest.mark.asyncio
    async def test_retry_async_failure(self):
        """Test the retry_async convenience function with failure."""
        from retry_manager import retry_async
        
        mock_func = AsyncMock()
        mock_func.side_effect = MockClientResponseError(401)
        
        with pytest.raises(MockClientResponseError):
            await retry_async(mock_func, max_attempts=2)
            
        assert mock_func.call_count == 1  # No retries for auth error


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])