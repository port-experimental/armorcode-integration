#!/usr/bin/env python3
"""
Direct ArmorCode API Client

This module provides a direct HTTP-based client for the ArmorCode API,
replacing the acsdk dependency with native aiohttp calls.

Usage:
    from armorcode_client import DirectArmorCodeClient
    
    async with DirectArmorCodeClient(api_key) as client:
        findings = await client.get_all_findings()
"""

import asyncio
import logging
import json
from typing import List, Dict, Any, Optional
import aiohttp
from aiohttp import ClientSession, ClientTimeout, ClientError


class ArmorCodeAPIError(Exception):
    """Custom exception for ArmorCode API errors."""
    def __init__(self, message: str, status_code: Optional[int] = None, response_data: Optional[dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data


class DirectArmorCodeClient:
    """
    Direct HTTP client for ArmorCode API.
    
    Provides the same interface as the original acsdk.ArmorCodeClient
    but uses direct HTTP calls for better control and reliability.
    """
    
    def __init__(self, api_key: str, base_url: str = "https://app.armorcode.com"):
        """
        Initialize the ArmorCode client.
        
        Args:
            api_key: ArmorCode API key for authentication
            base_url: Base URL for ArmorCode API (default: https://app.armorcode.com)
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.session: Optional[ClientSession] = None
        self.logger = logging.getLogger(__name__)
        
        # Request configuration
        self.timeout = ClientTimeout(total=300, connect=30)  # 5 minute total, 30s connect
        self.max_retries = 3
        self.retry_backoff = 2.0  # Exponential backoff multiplier
        
    async def __aenter__(self):
        """Async context manager entry."""
        await self._create_session()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self._close_session()
        
    async def _create_session(self):
        """Create aiohttp session with proper headers and timeout."""
        if self.session is None:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'User-Agent': 'DirectArmorCodeClient/1.0'
            }
            
            connector = aiohttp.TCPConnector(
                limit=100,           # Total connection pool size
                limit_per_host=30,   # Per-host connection limit
                ttl_dns_cache=300,   # DNS cache TTL
                use_dns_cache=True,
            )
            
            self.session = ClientSession(
                headers=headers,
                timeout=self.timeout,
                connector=connector,
                raise_for_status=False  # We'll handle status codes manually
            )
            
    async def _close_session(self):
        """Close aiohttp session."""
        if self.session:
            await self.session.close()
            self.session = None
            
    async def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[Any, Any]:
        """
        Make HTTP request with retry logic and error handling.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            **kwargs: Additional arguments for aiohttp request
            
        Returns:
            JSON response data as dictionary
            
        Raises:
            ArmorCodeAPIError: On API errors or request failures
        """
        if not self.session:
            await self._create_session()
            
        url = f"{self.base_url}{endpoint}"
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                self.logger.debug(f"Making {method} request to {url} (attempt {attempt + 1})")
                
                async with self.session.request(method, url, **kwargs) as response:
                    response_text = await response.text()
                    
                    # Log response details for debugging
                    self.logger.debug(f"Response status: {response.status}")
                    self.logger.debug(f"Response headers: {dict(response.headers)}")
                    
                    # Handle different response status codes
                    if response.status == 200:
                        try:
                            data = json.loads(response_text)
                            if data.get('success') is True:
                                return data
                            else:
                                error_msg = data.get('message', 'API returned success=false')
                                raise ArmorCodeAPIError(f"API error: {error_msg}", response.status, data)
                        except json.JSONDecodeError as e:
                            raise ArmorCodeAPIError(f"Invalid JSON response: {e}", response.status)
                            
                    elif response.status == 401:
                        raise ArmorCodeAPIError("Authentication failed - check API key", response.status)
                        
                    elif response.status == 429:
                        # Rate limited - wait and retry
                        retry_after = int(response.headers.get('Retry-After', 60))
                        self.logger.warning(f"Rate limited, waiting {retry_after}s before retry")
                        await asyncio.sleep(retry_after)
                        continue
                        
                    elif response.status >= 500:
                        # Server error - retry
                        error_msg = f"Server error {response.status}: {response_text[:200]}"
                        self.logger.warning(f"{error_msg} (attempt {attempt + 1})")
                        if attempt < self.max_retries:
                            await asyncio.sleep(self.retry_backoff ** attempt)
                            continue
                        raise ArmorCodeAPIError(error_msg, response.status)
                        
                    else:
                        # Client error - don't retry
                        error_msg = f"HTTP {response.status}: {response_text[:200]}"
                        raise ArmorCodeAPIError(error_msg, response.status)
                        
            except ClientError as e:
                last_exception = e
                error_msg = f"Network error: {e}"
                self.logger.warning(f"{error_msg} (attempt {attempt + 1})")
                
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_backoff ** attempt)
                    continue
                else:
                    raise ArmorCodeAPIError(f"Network error after {self.max_retries + 1} attempts: {e}")
                    
        # If we get here, all retries failed
        if last_exception:
            raise ArmorCodeAPIError(f"Request failed after {self.max_retries + 1} attempts") from last_exception
        else:
            raise ArmorCodeAPIError(f"Request failed after {self.max_retries + 1} attempts")
            
    async def get_findings_page(self, 
                               size: int = 100, 
                               after_key: Optional[int] = None,
                               filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get a single page of findings from ArmorCode API.
        
        Args:
            size: Number of findings to retrieve (1-1000, default: 100)
            after_key: Pagination key for next page (optional)
            filters: Additional filters for findings query (optional)
            
        Returns:
            API response containing findings data and pagination info
        """
        # Build query parameters
        params = {'size': min(size, 1000)}  # API max is 1000
        if after_key is not None:
            params['afterKey'] = after_key
            
        # Build request body with filters
        body = filters or {}
        
        self.logger.debug(f"Fetching findings page: size={size}, afterKey={after_key}")
        
        response = await self._make_request(
            'POST',
            '/api/findings',
            params=params,
            json=body
        )
        
        return response
        
    async def get_all_findings(self, 
                              filters: Optional[Dict[str, Any]] = None,
                              page_size: int = 1000) -> List[Dict[str, Any]]:
        """
        Get all findings from ArmorCode API with automatic pagination.
        
        This method provides the same interface as the original acsdk.ArmorCodeClient.get_all_findings()
        but uses direct HTTP calls with proper pagination handling.
        
        Args:
            filters: Optional filters to apply to findings query
            page_size: Number of findings per page (default: 1000, max: 1000)
            
        Returns:
            List of all findings matching the criteria
        """
        all_findings = []
        after_key = None
        page_num = 1
        
        self.logger.info("Starting to fetch all findings with pagination...")
        
        while True:
            try:
                self.logger.debug(f"Fetching page {page_num} (afterKey: {after_key})")
                
                # Get the current page
                response = await self.get_findings_page(
                    size=page_size,
                    after_key=after_key,
                    filters=filters
                )
                
                # Extract findings from response
                data = response.get('data', {})
                findings = data.get('findings', [])
                
                if not findings:
                    self.logger.info(f"No more findings found on page {page_num}")
                    break
                    
                all_findings.extend(findings)
                self.logger.info(f"Page {page_num}: Retrieved {len(findings)} findings "
                               f"(total so far: {len(all_findings)})")
                
                # Check for next page
                after_key = data.get('afterKey')
                if after_key is None:
                    self.logger.info("No more pages available (afterKey is None)")
                    break
                    
                page_num += 1
                
                # Optional: Add small delay between requests to be respectful
                await asyncio.sleep(0.1)
                
            except ArmorCodeAPIError as e:
                self.logger.error(f"API error while fetching page {page_num}: {e}")
                raise
            except Exception as e:
                self.logger.error(f"Unexpected error while fetching page {page_num}: {e}")
                raise ArmorCodeAPIError(f"Failed to fetch findings page {page_num}") from e
                
        self.logger.info(f"Successfully retrieved {len(all_findings)} total findings "
                        f"across {page_num} pages")
        return all_findings
        
    async def test_connection(self) -> bool:
        """
        Test the connection to ArmorCode API.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            self.logger.info("Testing connection to ArmorCode API...")
            response = await self.get_findings_page(size=1)
            self.logger.info("Connection test successful")
            return True
        except ArmorCodeAPIError as e:
            self.logger.error(f"Connection test failed: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Connection test failed with unexpected error: {e}")
            return False
            
# Note: We only implement the findings methods here since acsdk works fine for products/subproducts 