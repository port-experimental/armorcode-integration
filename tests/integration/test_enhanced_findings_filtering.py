"""
Integration tests for enhanced findings ingestion with filtering and afterKey support.

This module tests the integration of FilterManager with the findings ingestion process,
including validation, error handling, and various filter combinations.
"""

import asyncio
import json
import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from filter_manager import FilterManager, FilterValidationError
from armorcode_client import DirectArmorCodeClient
from main import ingest_findings
from step_executor import StepExecutor, ExecutionContext
from cli_controller import CLIConfig


class TestFilterManagerIntegration:
    """Test FilterManager integration with findings ingestion."""
    
    @pytest.fixture
    def sample_findings(self):
        """Sample findings data for testing."""
        return [
            {
                "id": 1,
                "title": "SQL Injection",
                "severity": "HIGH",
                "status": "OPEN",
                "description": "SQL injection vulnerability",
                "mitigation": "Use parameterized queries",
                "source": "SAST",
                "findingUrl": "https://example.com/finding/1",
                "cve": ["CVE-2023-1234"],
                "subProduct": {"id": 100}
            },
            {
                "id": 2,
                "title": "XSS Vulnerability",
                "severity": "MEDIUM",
                "status": "OPEN",
                "description": "Cross-site scripting vulnerability",
                "mitigation": "Sanitize user input",
                "source": "DAST",
                "findingUrl": "https://example.com/finding/2",
                "cve": [],
                "subProduct": {"id": 101}
            }
        ]
    
    @pytest.fixture
    def valid_filter_file(self):
        """Create a temporary valid filter file."""
        filter_data = {
            "severity": ["HIGH", "CRITICAL"],
            "status": "OPEN",
            "source": ["SAST", "DAST"]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(filter_data, f)
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        os.unlink(temp_path)
    
    @pytest.fixture
    def invalid_filter_file(self):
        """Create a temporary invalid filter file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content {")
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        os.unlink(temp_path)
    
    @pytest.fixture
    def nonexistent_filter_file(self):
        """Path to a non-existent filter file."""
        return "/tmp/nonexistent_filter_file.json"
    
    @pytest.mark.asyncio
    async def test_findings_ingestion_with_valid_filters(self, sample_findings, valid_filter_file):
        """Test findings ingestion with valid JSON filters."""
        with patch('armorcode_client.DirectArmorCodeClient') as mock_client_class, \
             patch('bulk_port_manager.BulkPortManager') as mock_bulk_manager_class, \
             patch('retry_manager.RetryManager') as mock_retry_manager_class:
            
            # Setup mock client
            mock_client = AsyncMock()
            mock_client.get_all_findings.return_value = sample_findings
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            # Setup mock bulk manager
            mock_bulk_manager = AsyncMock()
            mock_bulk_manager_class.return_value = mock_bulk_manager
            
            # Setup mock retry manager
            mock_retry_manager = AsyncMock()
            mock_retry_manager.with_retry.return_value = sample_findings
            mock_retry_manager_class.return_value = mock_retry_manager
            
            # Mock environment variable
            with patch.dict(os.environ, {'ARMORCODE_API_KEY': 'test-key'}):
                # Call the function with filter file
                await ingest_findings(
                    ac_client=None,
                    port_token="test-token",
                    dry_run=True,
                    finding_filters_path=valid_filter_file
                )
            
            # Verify that get_all_findings was called with the correct filters
            call_args = mock_retry_manager.with_retry.call_args
            
            # Check that filters were applied
            filters_arg = call_args[1]['filters']
            assert filters_arg is not None
            assert 'severity' in filters_arg
            assert filters_arg['severity'] == ["HIGH", "CRITICAL"]
    
    @pytest.mark.asyncio
    async def test_findings_ingestion_with_after_key(self, sample_findings):
        """Test findings ingestion with afterKey parameter."""
        with patch('armorcode_client.DirectArmorCodeClient') as mock_client_class, \
             patch('bulk_port_manager.BulkPortManager') as mock_bulk_manager_class, \
             patch('retry_manager.RetryManager') as mock_retry_manager_class:
            
            # Setup mock client
            mock_client = AsyncMock()
            mock_client.get_all_findings.return_value = sample_findings
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            # Setup mock bulk manager
            mock_bulk_manager = AsyncMock()
            mock_bulk_manager_class.return_value = mock_bulk_manager
            
            # Setup mock retry manager
            mock_retry_manager = AsyncMock()
            mock_retry_manager.with_retry.return_value = sample_findings
            mock_retry_manager_class.return_value = mock_retry_manager
            
            # Mock environment variable
            with patch.dict(os.environ, {'ARMORCODE_API_KEY': 'test-key'}):
                # Call the function with afterKey
                await ingest_findings(
                    ac_client=None,
                    port_token="test-token",
                    dry_run=True,
                    after_key=12345
                )
            
            # Verify that get_all_findings was called with afterKey
            call_args = mock_retry_manager.with_retry.call_args
            filters_arg = call_args[1]['filters']
            assert filters_arg is not None
            assert 'afterKey' in filters_arg
            assert filters_arg['afterKey'] == 12345
    
    @pytest.mark.asyncio
    async def test_findings_ingestion_with_combined_filters(self, sample_findings, valid_filter_file):
        """Test findings ingestion with both JSON filters and afterKey."""
        with patch('armorcode_client.DirectArmorCodeClient') as mock_client_class, \
             patch('bulk_port_manager.BulkPortManager') as mock_bulk_manager_class, \
             patch('retry_manager.RetryManager') as mock_retry_manager_class:
            
            # Setup mock client
            mock_client = AsyncMock()
            mock_client.get_all_findings.return_value = sample_findings
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            # Setup mock bulk manager
            mock_bulk_manager = AsyncMock()
            mock_bulk_manager_class.return_value = mock_bulk_manager
            
            # Setup mock retry manager
            mock_retry_manager = AsyncMock()
            mock_retry_manager.with_retry.return_value = sample_findings
            mock_retry_manager_class.return_value = mock_retry_manager
            
            # Mock environment variable
            with patch.dict(os.environ, {'ARMORCODE_API_KEY': 'test-key'}):
                # Call the function with both filters and afterKey
                await ingest_findings(
                    ac_client=None,
                    port_token="test-token",
                    dry_run=True,
                    finding_filters_path=valid_filter_file,
                    after_key=12345
                )
            
            # Verify that get_all_findings was called with combined filters
            call_args = mock_retry_manager.with_retry.call_args
            filters_arg = call_args[1]['filters']
            assert filters_arg is not None
            
            # Check that both file filters and afterKey are present
            assert 'severity' in filters_arg
            assert filters_arg['severity'] == ["HIGH", "CRITICAL"]
            assert 'afterKey' in filters_arg
            assert filters_arg['afterKey'] == 12345
    
    @pytest.mark.asyncio
    async def test_findings_ingestion_with_invalid_filter_file(self, invalid_filter_file):
        """Test findings ingestion with invalid JSON filter file."""
        with patch('armorcode_client.DirectArmorCodeClient') as mock_client_class, \
             patch('bulk_port_manager.BulkPortManager') as mock_bulk_manager_class, \
             patch('retry_manager.RetryManager') as mock_retry_manager_class:
            
            # Setup mocks
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            mock_bulk_manager = AsyncMock()
            mock_bulk_manager_class.return_value = mock_bulk_manager
            
            mock_retry_manager = AsyncMock()
            mock_retry_manager_class.return_value = mock_retry_manager
            
            # Mock environment variable
            with patch.dict(os.environ, {'ARMORCODE_API_KEY': 'test-key'}):
                # Call the function with invalid filter file
                await ingest_findings(
                    ac_client=None,
                    port_token="test-token",
                    dry_run=True,
                    finding_filters_path=invalid_filter_file
                )
            
            # Verify that get_all_findings was NOT called due to filter validation error
            mock_retry_manager.with_retry.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_findings_ingestion_with_nonexistent_filter_file(self, nonexistent_filter_file):
        """Test findings ingestion with non-existent filter file."""
        with patch('armorcode_client.DirectArmorCodeClient') as mock_client_class, \
             patch('bulk_port_manager.BulkPortManager') as mock_bulk_manager_class, \
             patch('retry_manager.RetryManager') as mock_retry_manager_class:
            
            # Setup mocks
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            mock_bulk_manager = AsyncMock()
            mock_bulk_manager_class.return_value = mock_bulk_manager
            
            mock_retry_manager = AsyncMock()
            mock_retry_manager_class.return_value = mock_retry_manager
            
            # Mock environment variable
            with patch.dict(os.environ, {'ARMORCODE_API_KEY': 'test-key'}):
                # Call the function with non-existent filter file
                await ingest_findings(
                    ac_client=None,
                    port_token="test-token",
                    dry_run=True,
                    finding_filters_path=nonexistent_filter_file
                )
            
            # Verify that get_all_findings was NOT called due to file not found error
            mock_retry_manager.with_retry.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_findings_ingestion_without_filters(self, sample_findings):
        """Test findings ingestion without any filters (backward compatibility)."""
        with patch('armorcode_client.DirectArmorCodeClient') as mock_client_class, \
             patch('bulk_port_manager.BulkPortManager') as mock_bulk_manager_class, \
             patch('retry_manager.RetryManager') as mock_retry_manager_class:
            
            # Setup mock client
            mock_client = AsyncMock()
            mock_client.get_all_findings.return_value = sample_findings
            mock_client_class.return_value.__aenter__.return_value = mock_client
            
            # Setup mock bulk manager
            mock_bulk_manager = AsyncMock()
            mock_bulk_manager_class.return_value = mock_bulk_manager
            
            # Setup mock retry manager
            mock_retry_manager = AsyncMock()
            mock_retry_manager.with_retry.return_value = sample_findings
            mock_retry_manager_class.return_value = mock_retry_manager
            
            # Mock environment variable
            with patch.dict(os.environ, {'ARMORCODE_API_KEY': 'test-key'}):
                # Call the function without any filters
                await ingest_findings(
                    ac_client=None,
                    port_token="test-token",
                    dry_run=True
                )
            
            # Verify that get_all_findings was called with None filters
            call_args = mock_retry_manager.with_retry.call_args
            filters_arg = call_args[1]['filters']
            assert filters_arg is None


class TestStepExecutorFilteringIntegration:
    """Test StepExecutor integration with filtering functionality."""
    
    @pytest.fixture
    def sample_findings(self):
        """Sample findings data for testing."""
        return [
            {
                "id": 1,
                "title": "SQL Injection",
                "severity": "HIGH",
                "status": "OPEN",
                "description": "SQL injection vulnerability",
                "mitigation": "Use parameterized queries",
                "source": "SAST",
                "findingUrl": "https://example.com/finding/1",
                "cve": ["CVE-2023-1234"],
                "subProduct": {"id": 100}
            }
        ]
    
    @pytest.fixture
    def valid_filter_file(self):
        """Create a temporary valid filter file."""
        filter_data = {
            "severity": ["HIGH", "CRITICAL"],
            "status": "OPEN"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(filter_data, f)
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        os.unlink(temp_path)
    
    @pytest.mark.asyncio
    async def test_step_executor_with_filters(self, sample_findings, valid_filter_file):
        """Test StepExecutor finding step with filters."""
        # Create CLI config with filters
        config = CLIConfig(
            steps=["finding"],
            after_key=None,
            finding_filters_path=valid_filter_file,
            batch_size=20,
            retry_attempts=3,
            dry_run=True
        )
        
        with patch('step_executor.ArmorCodeClient') as mock_ac_client_class, \
             patch('step_executor.DirectArmorCodeClient') as mock_direct_client_class, \
             patch('step_executor.BulkPortManager') as mock_bulk_manager_class:
            
            # Setup mock clients
            mock_ac_client = AsyncMock()
            mock_ac_client_class.return_value = mock_ac_client
            
            mock_direct_client = AsyncMock()
            mock_direct_client.get_all_findings.return_value = sample_findings
            mock_direct_client_class.return_value = mock_direct_client
            
            # Setup mock bulk manager
            mock_bulk_manager = AsyncMock()
            mock_bulk_manager_class.return_value = mock_bulk_manager
            
            # Mock environment variable
            with patch.dict(os.environ, {'ARMORCODE_API_KEY': 'test-key'}):
                # Create execution context
                context = ExecutionContext(
                    config=config,
                    port_token="test-token"
                )
                
                # Create step executor
                executor = StepExecutor()
                
                # Execute finding step
                async with context:
                    results = await executor.execute_steps(context, ["finding"])
                
                # Verify the step completed successfully
                assert "finding" in results
                finding_result = results["finding"]
                assert finding_result.success
                
                # Verify that get_all_findings was called with filters
                mock_direct_client.get_all_findings.assert_called_once()
                call_args = mock_direct_client.get_all_findings.call_args
                filters_arg = call_args[1]['filters']
                assert filters_arg is not None
                assert 'severity' in filters_arg
    
    @pytest.mark.asyncio
    async def test_step_executor_with_after_key(self, sample_findings):
        """Test StepExecutor finding step with afterKey."""
        # Create CLI config with afterKey
        config = CLIConfig(
            steps=["finding"],
            after_key=12345,
            finding_filters_path=None,
            batch_size=20,
            retry_attempts=3,
            dry_run=True
        )
        
        with patch('step_executor.ArmorCodeClient') as mock_ac_client_class, \
             patch('step_executor.DirectArmorCodeClient') as mock_direct_client_class, \
             patch('step_executor.BulkPortManager') as mock_bulk_manager_class:
            
            # Setup mock clients
            mock_ac_client = AsyncMock()
            mock_ac_client_class.return_value = mock_ac_client
            
            mock_direct_client = AsyncMock()
            mock_direct_client.get_all_findings.return_value = sample_findings
            mock_direct_client_class.return_value = mock_direct_client
            
            # Setup mock bulk manager
            mock_bulk_manager = AsyncMock()
            mock_bulk_manager_class.return_value = mock_bulk_manager
            
            # Mock environment variable
            with patch.dict(os.environ, {'ARMORCODE_API_KEY': 'test-key'}):
                # Create execution context
                context = ExecutionContext(
                    config=config,
                    port_token="test-token"
                )
                
                # Create step executor
                executor = StepExecutor()
                
                # Execute finding step
                async with context:
                    results = await executor.execute_steps(context, ["finding"])
                
                # Verify the step completed successfully
                assert "finding" in results
                finding_result = results["finding"]
                assert finding_result.success
                
                # Verify that get_all_findings was called with afterKey
                mock_direct_client.get_all_findings.assert_called_once()
                call_args = mock_direct_client.get_all_findings.call_args
                filters_arg = call_args[1]['filters']
                assert filters_arg is not None
                assert 'afterKey' in filters_arg
                assert filters_arg['afterKey'] == 12345
    
    @pytest.mark.asyncio
    async def test_step_executor_with_invalid_filters(self):
        """Test StepExecutor finding step with invalid filter file."""
        # Create invalid filter file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json {")
            invalid_filter_file = f.name
        
        try:
            # Create CLI config with invalid filter file
            config = CLIConfig(
                steps=["finding"],
                after_key=None,
                finding_filters_path=invalid_filter_file,
                batch_size=20,
                retry_attempts=3,
                dry_run=True
            )
            
            with patch('step_executor.ArmorCodeClient') as mock_ac_client_class, \
                 patch('step_executor.DirectArmorCodeClient') as mock_direct_client_class, \
                 patch('step_executor.BulkPortManager') as mock_bulk_manager_class:
                
                # Setup mock clients
                mock_ac_client = AsyncMock()
                mock_ac_client_class.return_value = mock_ac_client
                
                mock_direct_client = AsyncMock()
                mock_direct_client_class.return_value = mock_direct_client
                
                # Setup mock bulk manager
                mock_bulk_manager = AsyncMock()
                mock_bulk_manager_class.return_value = mock_bulk_manager
                
                # Mock environment variable
                with patch.dict(os.environ, {'ARMORCODE_API_KEY': 'test-key'}):
                    # Create execution context
                    context = ExecutionContext(
                        config=config,
                        port_token="test-token"
                    )
                    
                    # Create step executor
                    executor = StepExecutor()
                    
                    # Execute finding step
                    async with context:
                        results = await executor.execute_steps(context, ["finding"])
                    
                    # Verify the step failed due to filter validation error
                    assert "finding" in results
                    finding_result = results["finding"]
                    assert not finding_result.success
                    assert "Filter validation failed" in finding_result.message
                    
                    # Verify that get_all_findings was NOT called
                    mock_direct_client.get_all_findings.assert_not_called()
        
        finally:
            # Cleanup
            os.unlink(invalid_filter_file)


class TestDirectArmorCodeClientFiltering:
    """Test DirectArmorCodeClient filtering integration."""
    
    @pytest.mark.asyncio
    async def test_direct_client_filters_parameter(self):
        """Test that DirectArmorCodeClient properly passes filters to API."""
        test_filters = {
            "severity": ["HIGH", "CRITICAL"],
            "status": "OPEN",
            "afterKey": 12345
        }
        
        # Mock the _make_request method directly to avoid network calls
        with patch.object(DirectArmorCodeClient, '_make_request') as mock_make_request:
            mock_make_request.return_value = {
                "success": True,
                "data": {
                    "findings": [],
                    "afterKey": None
                }
            }
            
            # Create client and call get_all_findings with filters
            client = DirectArmorCodeClient("test-api-key")
            
            try:
                await client.get_all_findings(filters=test_filters)
                
                # Verify that _make_request was called with correct parameters
                mock_make_request.assert_called_once()
                call_args = mock_make_request.call_args
                
                # Check method and endpoint
                assert call_args[0][0] == 'POST'
                assert call_args[0][1] == '/api/findings'
                
                # Check that filters were passed as JSON body
                assert 'json' in call_args[1]
                json_body = call_args[1]['json']
                assert json_body == test_filters
                
            finally:
                if client.session:
                    await client._close_session()


if __name__ == "__main__":
    pytest.main([__file__])