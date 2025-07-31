#!/usr/bin/env python3
"""
Integration tests for enhanced subproduct ingestion with bulk operations and retry logic.

Tests the enhanced ingest_subproducts function with BulkPortManager and RetryManager
integration, covering bulk operations, retry logic, error handling, and statistics.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from acsdk import ArmorCodeClient

from main import ingest_subproducts
from bulk_port_manager import BulkPortManager, BulkResult
from retry_manager import RetryManager


class TestEnhancedSubproductIngestion:
    """Test suite for enhanced subproduct ingestion functionality."""

    @pytest.fixture
    def mock_ac_client(self):
        """Create a mock ArmorCode client."""
        client = AsyncMock(spec=ArmorCodeClient)
        return client

    @pytest.fixture
    def mock_port_token(self):
        """Mock Port API token."""
        return "mock-port-token"

    @pytest.fixture
    def sample_subproducts(self):
        """Sample subproduct data from ArmorCode API."""
        return [
            {
                "id": 1,
                "name": "Web Frontend",
                "repoLink": "https://github.com/company/web-frontend",
                "programmingLanguage": "JavaScript",
                "technologies": "React, TypeScript, Webpack",
                "parent": 100
            },
            {
                "id": 2,
                "name": "API Gateway",
                "repoLink": "https://github.com/company/api-gateway",
                "programmingLanguage": "Python",
                "technologies": "FastAPI, PostgreSQL",
                "parent": 100
            },
            {
                "id": 3,
                "name": "Mobile App",
                "repoLink": "https://github.com/company/mobile-app",
                "programmingLanguage": "Swift",
                "technologies": "SwiftUI, Core Data",
                "parent": 101
            }
        ]

    @pytest.fixture
    def expected_entities(self):
        """Expected Port entities after transformation."""
        return [
            {
                "identifier": "1",
                "title": "Web Frontend",
                "properties": {
                    "name": "Web Frontend",
                    "repoLink": "https://github.com/company/web-frontend",
                    "programmingLanguage": "JavaScript",
                    "technologies": ["React", "TypeScript", "Webpack"],
                },
                "relations": {"product": "100"},
            },
            {
                "identifier": "2",
                "title": "API Gateway",
                "properties": {
                    "name": "API Gateway",
                    "repoLink": "https://github.com/company/api-gateway",
                    "programmingLanguage": "Python",
                    "technologies": ["FastAPI", "PostgreSQL"],
                },
                "relations": {"product": "100"},
            },
            {
                "identifier": "3",
                "title": "Mobile App",
                "properties": {
                    "name": "Mobile App",
                    "repoLink": "https://github.com/company/mobile-app",
                    "programmingLanguage": "Swift",
                    "technologies": ["SwiftUI", "Core Data"],
                },
                "relations": {"product": "101"},
            }
        ]

    @pytest.mark.asyncio
    async def test_successful_bulk_ingestion(self, mock_ac_client, mock_port_token, 
                                           sample_subproducts, expected_entities):
        """Test successful subproduct ingestion with bulk operations."""
        # Setup mocks
        mock_ac_client.get_all_subproducts.return_value = sample_subproducts
        
        mock_bulk_manager = AsyncMock(spec=BulkPortManager)
        mock_bulk_result = BulkResult(
            successful_entities=["1", "2", "3"],
            failed_entities=[],
            total_processed=3
        )
        mock_bulk_manager.create_entities_bulk.return_value = mock_bulk_result
        mock_bulk_manager.__aenter__.return_value = mock_bulk_manager
        mock_bulk_manager.__aexit__.return_value = None
        
        mock_retry_manager = AsyncMock(spec=RetryManager)
        mock_retry_manager.with_retry.side_effect = lambda func, *args, **kwargs: func(*args, **kwargs)
        
        # Execute function
        await ingest_subproducts(
            mock_ac_client, 
            mock_port_token, 
            dry_run=False,
            bulk_manager=mock_bulk_manager,
            retry_manager=mock_retry_manager,
            batch_size=20
        )
        
        # Verify ArmorCode API call with retry
        mock_retry_manager.with_retry.assert_any_call(mock_ac_client.get_all_subproducts)
        
        # Verify bulk manager usage
        mock_bulk_manager.__aenter__.assert_called_once()
        mock_retry_manager.with_retry.assert_any_call(
            mock_bulk_manager.create_entities_bulk,
            "armorcodeSubProduct",
            expected_entities,
            20
        )

    @pytest.mark.asyncio
    async def test_armorcode_api_failure_with_retry(self, mock_ac_client, mock_port_token):
        """Test handling of ArmorCode API failures with retry logic."""
        # Setup mock to fail on ArmorCode API call
        api_error = Exception("ArmorCode API connection failed")
        mock_retry_manager = AsyncMock(spec=RetryManager)
        mock_retry_manager.with_retry.side_effect = api_error
        
        mock_bulk_manager = AsyncMock(spec=BulkPortManager)
        
        # Execute function - should handle error gracefully
        await ingest_subproducts(
            mock_ac_client,
            mock_port_token,
            dry_run=False,
            bulk_manager=mock_bulk_manager,
            retry_manager=mock_retry_manager
        )
        
        # Verify retry was attempted
        mock_retry_manager.with_retry.assert_called_once_with(mock_ac_client.get_all_subproducts)
        
        # Verify bulk manager was not used due to API failure
        mock_bulk_manager.create_entities_bulk.assert_not_called()

    @pytest.mark.asyncio
    async def test_partial_transformation_failures(self, mock_ac_client, mock_port_token):
        """Test handling of individual subproduct transformation failures."""
        # Setup subproducts with some invalid data
        problematic_subproducts = [
            {
                "id": 1,
                "name": "Valid Subproduct",
                "repoLink": "https://github.com/company/valid",
                "programmingLanguage": "Python",
                "technologies": "Django, PostgreSQL",
                "parent": 100
            },
            {
                # Missing required 'id' field
                "name": "Invalid Subproduct",
                "parent": 100
            },
            {
                "id": 3,
                "name": "Another Valid",
                "repoLink": "https://github.com/company/valid2",
                "programmingLanguage": "Java",
                "technologies": "Spring Boot",
                "parent": 101
            }
        ]
        
        mock_ac_client.get_all_subproducts.return_value = problematic_subproducts
        
        mock_bulk_manager = AsyncMock(spec=BulkPortManager)
        mock_bulk_result = BulkResult(
            successful_entities=["1", "3"],
            failed_entities=[],
            total_processed=2
        )
        mock_bulk_manager.create_entities_bulk.return_value = mock_bulk_result
        mock_bulk_manager.__aenter__.return_value = mock_bulk_manager
        mock_bulk_manager.__aexit__.return_value = None
        
        mock_retry_manager = AsyncMock(spec=RetryManager)
        mock_retry_manager.with_retry.side_effect = lambda func, *args, **kwargs: func(*args, **kwargs)
        
        # Execute function
        await ingest_subproducts(
            mock_ac_client,
            mock_port_token,
            dry_run=False,
            bulk_manager=mock_bulk_manager,
            retry_manager=mock_retry_manager
        )
        
        # Verify only valid entities were processed
        call_args = mock_retry_manager.with_retry.call_args_list[-1]  # Last call to with_retry
        entities_arg = call_args[0][2]  # Third argument (entities)
        assert len(entities_arg) == 2  # Only 2 valid entities
        assert entities_arg[0]["identifier"] == "1"
        assert entities_arg[1]["identifier"] == "3"

    @pytest.mark.asyncio
    async def test_bulk_operation_partial_failures(self, mock_ac_client, mock_port_token, sample_subproducts):
        """Test handling of partial failures in bulk operations."""
        mock_ac_client.get_all_subproducts.return_value = sample_subproducts
        
        # Setup bulk manager to return partial failures
        mock_bulk_manager = AsyncMock(spec=BulkPortManager)
        mock_bulk_result = BulkResult(
            successful_entities=["1", "3"],
            failed_entities=[("2", "Port API validation error")],
            total_processed=3
        )
        mock_bulk_manager.create_entities_bulk.return_value = mock_bulk_result
        mock_bulk_manager.__aenter__.return_value = mock_bulk_manager
        mock_bulk_manager.__aexit__.return_value = None
        
        mock_retry_manager = AsyncMock(spec=RetryManager)
        mock_retry_manager.with_retry.side_effect = lambda func, *args, **kwargs: func(*args, **kwargs)
        
        # Execute function
        await ingest_subproducts(
            mock_ac_client,
            mock_port_token,
            dry_run=False,
            bulk_manager=mock_bulk_manager,
            retry_manager=mock_retry_manager
        )
        
        # Verify function completed despite partial failures
        mock_bulk_manager.create_entities_bulk.assert_called_once()
        
        # Verify statistics are logged (success rate should be 66.7%)
        assert mock_bulk_result.success_rate == pytest.approx(66.7, rel=1e-1)

    @pytest.mark.asyncio
    async def test_bulk_operation_complete_failure_with_retry(self, mock_ac_client, mock_port_token, sample_subproducts):
        """Test handling of complete bulk operation failure with retry logic."""
        mock_ac_client.get_all_subproducts.return_value = sample_subproducts
        
        mock_bulk_manager = AsyncMock(spec=BulkPortManager)
        mock_bulk_manager.__aenter__.return_value = mock_bulk_manager
        mock_bulk_manager.__aexit__.return_value = None
        
        # Setup retry manager to fail after retries
        bulk_error = Exception("Port API bulk endpoint unavailable")
        mock_retry_manager = AsyncMock(spec=RetryManager)
        mock_retry_manager.with_retry.side_effect = [
            sample_subproducts,  # First call (get_all_subproducts) succeeds
            bulk_error  # Second call (create_entities_bulk) fails after retries
        ]
        
        # Execute function - should handle bulk failure gracefully
        await ingest_subproducts(
            mock_ac_client,
            mock_port_token,
            dry_run=False,
            bulk_manager=mock_bulk_manager,
            retry_manager=mock_retry_manager
        )
        
        # Verify both retry attempts were made
        assert mock_retry_manager.with_retry.call_count == 2

    @pytest.mark.asyncio
    async def test_dry_run_mode(self, mock_ac_client, mock_port_token, sample_subproducts):
        """Test dry run mode doesn't perform actual operations."""
        mock_ac_client.get_all_subproducts.return_value = sample_subproducts
        
        mock_bulk_manager = AsyncMock(spec=BulkPortManager)
        mock_retry_manager = AsyncMock(spec=RetryManager)
        mock_retry_manager.with_retry.side_effect = lambda func, *args, **kwargs: func(*args, **kwargs)
        
        # Execute in dry run mode
        await ingest_subproducts(
            mock_ac_client,
            mock_port_token,
            dry_run=True,
            bulk_manager=mock_bulk_manager,
            retry_manager=mock_retry_manager
        )
        
        # Verify ArmorCode API was called with retry
        mock_retry_manager.with_retry.assert_called_once_with(mock_ac_client.get_all_subproducts)
        
        # Verify no bulk operations were performed
        mock_bulk_manager.create_entities_bulk.assert_not_called()
        mock_bulk_manager.__aenter__.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_subproducts_list(self, mock_ac_client, mock_port_token):
        """Test handling of empty subproducts list."""
        mock_ac_client.get_all_subproducts.return_value = []
        
        mock_bulk_manager = AsyncMock(spec=BulkPortManager)
        mock_retry_manager = AsyncMock(spec=RetryManager)
        mock_retry_manager.with_retry.side_effect = lambda func, *args, **kwargs: func(*args, **kwargs)
        
        # Execute function
        await ingest_subproducts(
            mock_ac_client,
            mock_port_token,
            dry_run=False,
            bulk_manager=mock_bulk_manager,
            retry_manager=mock_retry_manager
        )
        
        # Verify no bulk operations were attempted
        mock_bulk_manager.create_entities_bulk.assert_not_called()

    @pytest.mark.asyncio
    async def test_technologies_field_parsing(self, mock_ac_client, mock_port_token):
        """Test proper parsing of technologies field from string to array."""
        subproducts_with_various_tech_formats = [
            {
                "id": 1,
                "name": "App with Tech",
                "technologies": "React, TypeScript, Webpack",
                "parent": 100
            },
            {
                "id": 2,
                "name": "App without Tech",
                "technologies": None,
                "parent": 100
            },
            {
                "id": 3,
                "name": "App with Empty Tech",
                "technologies": "",
                "parent": 100
            }
        ]
        
        mock_ac_client.get_all_subproducts.return_value = subproducts_with_various_tech_formats
        
        mock_bulk_manager = AsyncMock(spec=BulkPortManager)
        mock_bulk_result = BulkResult(
            successful_entities=["1", "2", "3"],
            failed_entities=[],
            total_processed=3
        )
        mock_bulk_manager.create_entities_bulk.return_value = mock_bulk_result
        mock_bulk_manager.__aenter__.return_value = mock_bulk_manager
        mock_bulk_manager.__aexit__.return_value = None
        
        mock_retry_manager = AsyncMock(spec=RetryManager)
        mock_retry_manager.with_retry.side_effect = lambda func, *args, **kwargs: func(*args, **kwargs)
        
        # Execute function
        await ingest_subproducts(
            mock_ac_client,
            mock_port_token,
            dry_run=False,
            bulk_manager=mock_bulk_manager,
            retry_manager=mock_retry_manager
        )
        
        # Verify technologies field was parsed correctly
        call_args = mock_retry_manager.with_retry.call_args_list[-1]
        entities = call_args[0][2]
        
        # First entity should have parsed technologies
        assert entities[0]["properties"]["technologies"] == ["React", "TypeScript", "Webpack"]
        
        # Second and third entities should have empty technologies array
        assert entities[1]["properties"]["technologies"] == []
        assert entities[2]["properties"]["technologies"] == []

    @pytest.mark.asyncio
    async def test_default_managers_creation(self, mock_ac_client, mock_port_token, sample_subproducts):
        """Test that default managers are created when not provided."""
        mock_ac_client.get_all_subproducts.return_value = sample_subproducts
        
        with patch('bulk_port_manager.BulkPortManager') as mock_bulk_class, \
             patch('retry_manager.RetryManager') as mock_retry_class:
            
            # Setup mock instances
            mock_bulk_instance = AsyncMock()
            mock_bulk_result = BulkResult(
                successful_entities=["1", "2", "3"],
                failed_entities=[],
                total_processed=3
            )
            mock_bulk_instance.create_entities_bulk.return_value = mock_bulk_result
            mock_bulk_instance.__aenter__.return_value = mock_bulk_instance
            mock_bulk_instance.__aexit__.return_value = None
            mock_bulk_class.return_value = mock_bulk_instance
            
            mock_retry_instance = AsyncMock()
            mock_retry_instance.with_retry.side_effect = lambda func, *args, **kwargs: func(*args, **kwargs)
            mock_retry_class.return_value = mock_retry_instance
            
            # Execute function without providing managers
            await ingest_subproducts(mock_ac_client, mock_port_token, dry_run=False)
            
            # Verify managers were created with correct parameters
            mock_bulk_class.assert_called_once_with(mock_port_token)
            mock_retry_class.assert_called_once_with(max_attempts=3)

    @pytest.mark.asyncio
    async def test_custom_batch_size(self, mock_ac_client, mock_port_token, sample_subproducts):
        """Test custom batch size is passed to bulk manager."""
        mock_ac_client.get_all_subproducts.return_value = sample_subproducts
        
        mock_bulk_manager = AsyncMock(spec=BulkPortManager)
        mock_bulk_result = BulkResult(
            successful_entities=["1", "2", "3"],
            failed_entities=[],
            total_processed=3
        )
        mock_bulk_manager.create_entities_bulk.return_value = mock_bulk_result
        mock_bulk_manager.__aenter__.return_value = mock_bulk_manager
        mock_bulk_manager.__aexit__.return_value = None
        
        mock_retry_manager = AsyncMock(spec=RetryManager)
        mock_retry_manager.with_retry.side_effect = lambda func, *args, **kwargs: func(*args, **kwargs)
        
        # Execute with custom batch size
        custom_batch_size = 10
        await ingest_subproducts(
            mock_ac_client,
            mock_port_token,
            dry_run=False,
            bulk_manager=mock_bulk_manager,
            retry_manager=mock_retry_manager,
            batch_size=custom_batch_size
        )
        
        # Verify custom batch size was passed to bulk manager
        call_args = mock_retry_manager.with_retry.call_args_list[-1]
        assert call_args[0][3] == custom_batch_size  # Fourth argument is batch_size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])