import pytest
import asyncio
import os
from unittest.mock import AsyncMock, patch, MagicMock
from pathlib import Path

# Test enhanced ingest_findings function with:
# - Batch processing with accumulator pattern
# - Parallel processing with controlled concurrency
# - Proper error handling and retry logic
# - Comprehensive logging and metrics

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from main import ingest_findings
from bulk_port_manager import BulkPortManager, BatchAccumulator, BatchStats, BulkResult


class TestParallelFindingsProcessing:
    """Test parallel findings processing functionality."""
    
    @pytest.fixture
    def sample_findings(self):
        """Sample findings data for testing."""
        return [
            {
                "id": 1,
                "title": "SQL Injection Finding",
                "source": "SAST Tool",
                "description": "SQL injection vulnerability",
                "severity": "HIGH",
                "status": "OPEN",
                "cve": ["CVE-2023-1234"],
                "cwe": ["CWE-89"],
                "findingUrl": "https://example.com/1",
                "product": {"id": "prod1", "name": "Product 1"},
                "subProduct": {"id": "sub1", "name": "SubProduct 1"}
            },
            {
                "id": 2,
                "title": "XSS Finding",
                "source": "DAST Tool", 
                "description": "Cross-site scripting vulnerability",
                "severity": "MEDIUM",
                "status": "OPEN",
                "cve": ["CVE-2023-5678"],
                "cwe": ["CWE-79"],
                "findingUrl": "https://example.com/2",
                "product": {"id": "prod1", "name": "Product 1"},
                "subProduct": {"id": "sub1", "name": "SubProduct 1"}
            },
            {
                "id": 3,
                "title": "Hardcoded Secrets",
                "source": "Secret Scanner",
                "description": "Hardcoded API key found",
                "severity": "CRITICAL",
                "status": "OPEN",
                "cve": [],
                "cwe": ["CWE-798"],
                "findingUrl": "https://example.com/3",
                "product": {"id": "prod2", "name": "Product 2"},
                "subProduct": {"id": "sub2", "name": "SubProduct 2"}
            },
            {
                "id": 4,
                "title": "Weak Encryption",
                "source": "Code Review",
                "description": "Using deprecated encryption algorithm",
                "severity": "LOW",
                "status": "OPEN",
                "cve": [],
                "cwe": ["CWE-327"],
                "findingUrl": "https://example.com/4",
                "product": {"id": "prod2", "name": "Product 2"},
                "subProduct": {"id": "sub2", "name": "SubProduct 2"}
            }
        ]
    
    @pytest.fixture
    def mock_bulk_manager(self):
        """Mock BulkPortManager for testing."""
        manager = AsyncMock(spec=BulkPortManager)
        
        # Mock batch accumulator creation
        finding_accumulator = AsyncMock(spec=BatchAccumulator)
        
        # Mock accumulator stats
        finding_stats = BatchStats(
            total_entities=4,
            batches_submitted=1,
            successful_entities=4,
            failed_entities=0
        )
        
        finding_accumulator.get_stats.return_value = finding_stats
        
        # Mock flush results
        finding_accumulator.flush_remaining.return_value = BulkResult(
            successful_entities=["1", "2", "3", "4"],
            failed_entities=[],
            total_processed=4
        )
        
        # Mock create_batch_accumulator to return the finding accumulator
        def create_accumulator_side_effect(blueprint_id, batch_size):
            if blueprint_id == "armorcodeFinding":
                return finding_accumulator
            else:
                raise ValueError(f"Unexpected blueprint_id: {blueprint_id}")
        
        manager.create_batch_accumulator.side_effect = create_accumulator_side_effect
        manager._finding_accumulator = finding_accumulator
        
        return manager
    
    @pytest.fixture
    def mock_retry_manager(self):
        """Mock RetryManager for testing."""
        retry_manager = AsyncMock()
        
        async def mock_with_retry(func, *args, **kwargs):
            return await func(*args, **kwargs)
        
        retry_manager.with_retry.side_effect = mock_with_retry
        return retry_manager

    @pytest.mark.asyncio
    async def test_parallel_processing_basic(self, sample_findings, mock_bulk_manager, mock_retry_manager):
        """Test basic parallel processing of findings."""
        with patch.dict(os.environ, {'ARMORCODE_API_KEY': 'test-key'}):
            with patch('main.DirectArmorCodeClient') as mock_client_class:
                # Setup mock client
                mock_client = AsyncMock()
                mock_client.get_all_findings.return_value = sample_findings
                mock_client_class.return_value.__aenter__.return_value = mock_client
                
                # Run the function
                await ingest_findings(
                    ac_client=None,  # Will be overridden by DirectArmorCodeClient patch
                    port_token="test-token",
                    dry_run=False,
                    bulk_manager=mock_bulk_manager,
                    retry_manager=mock_retry_manager,
                    batch_size=20,
                    max_concurrent=10
                )
                
                # Verify findings were fetched
                mock_retry_manager.with_retry.assert_called_once()
                
                # Verify batch accumulators were created
                assert mock_bulk_manager.create_batch_accumulator.call_count == 1
                mock_bulk_manager.create_batch_accumulator.assert_any_call("armorcodeFinding", 20)
                
                # Verify entities were added to accumulators
                finding_accumulator = mock_bulk_manager._finding_accumulator
                
                # Should have 4 findings processed
                assert finding_accumulator.add_entity.call_count == 4
                
                # Verify finalize was called
                finding_accumulator.finalize.assert_called_once()

    @pytest.mark.asyncio
    async def test_dry_run_mode(self, sample_findings, mock_bulk_manager, mock_retry_manager):
        """Test dry run mode functionality."""
        with patch.dict(os.environ, {'ARMORCODE_API_KEY': 'test-key'}):
            with patch('main.DirectArmorCodeClient') as mock_client_class:
                # Setup mock client
                mock_client = AsyncMock()
                mock_client.get_all_findings.return_value = sample_findings
                mock_client_class.return_value.__aenter__.return_value = mock_client
                
                # Run in dry run mode
                await ingest_findings(
                    ac_client=None,
                    port_token="test-token", 
                    dry_run=True,
                    bulk_manager=mock_bulk_manager,
                    retry_manager=mock_retry_manager,
                    batch_size=20,
                    max_concurrent=10
                )
                
                # In dry run mode, no entities should be submitted
                finding_accumulator = mock_bulk_manager._finding_accumulator
                finding_accumulator.finalize.assert_not_called()
