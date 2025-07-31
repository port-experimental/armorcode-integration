"""
Integration tests for BulkPortManager demonstrating real-world usage patterns.

These tests show how the BulkPortManager integrates with the existing codebase
and can be used to replace individual entity creation calls.
"""

import asyncio
import json
from unittest.mock import patch
from typing import List, Dict, Any

import pytest
from aioresponses import aioresponses

from bulk_port_manager import BulkPortManager, BulkResult


class TestBulkPortManagerIntegration:
    """Integration tests for BulkPortManager."""
    
    @pytest.fixture
    def port_token(self):
        """Mock Port API token."""
        return "test-port-token"
    
    @pytest.fixture
    def sample_products(self):
        """Sample product entities similar to those created in main.py."""
        return [
            {
                "identifier": "1",
                "title": "Product 1",
                "properties": {
                    "name": "Product 1",
                    "description": "Test product 1",
                    "businessOwner": "John Doe",
                    "securityOwner": "Jane Smith",
                },
            },
            {
                "identifier": "2", 
                "title": "Product 2",
                "properties": {
                    "name": "Product 2",
                    "description": "Test product 2",
                    "businessOwner": "Bob Wilson",
                    "securityOwner": "Alice Brown",
                },
            },
            {
                "identifier": "3",
                "title": "Product 3", 
                "properties": {
                    "name": "Product 3",
                    "description": "Test product 3",
                    "businessOwner": "Charlie Davis",
                    "securityOwner": "Diana Miller",
                },
            }
        ]
    
    @pytest.fixture
    def sample_subproducts(self):
        """Sample subproduct entities similar to those created in main.py."""
        return [
            {
                "identifier": "101",
                "title": "Subproduct 1",
                "properties": {
                    "name": "Subproduct 1",
                    "repoLink": "https://github.com/example/repo1",
                    "programmingLanguage": "Python",
                    "technologies": ["Django", "PostgreSQL"],
                },
                "relations": {"product": "1"},
            },
            {
                "identifier": "102",
                "title": "Subproduct 2", 
                "properties": {
                    "name": "Subproduct 2",
                    "repoLink": "https://github.com/example/repo2",
                    "programmingLanguage": "JavaScript",
                    "technologies": ["React", "Node.js"],
                },
                "relations": {"product": "1"},
            },
            {
                "identifier": "103",
                "title": "Subproduct 3",
                "properties": {
                    "name": "Subproduct 3", 
                    "repoLink": "https://github.com/example/repo3",
                    "programmingLanguage": "Java",
                    "technologies": ["Spring", "MySQL"],
                },
                "relations": {"product": "2"},
            }
        ]
    
    @pytest.mark.asyncio
    async def test_bulk_product_ingestion(self, port_token, sample_products):
        """Test bulk ingestion of products similar to ingest_products function."""
        with aioresponses() as m:
            # Mock successful bulk creation
            m.post(
                "https://api.us.getport.io/v1/blueprints/armorcodeProduct/entities/bulk",
                payload={"message": "Success"},
                status=200
            )
            
            async with BulkPortManager(port_token) as bulk_manager:
                result = await bulk_manager.create_entities_bulk(
                    "armorcodeProduct", 
                    sample_products
                )
            
            assert result.success_count == 3
            assert result.failure_count == 0
            assert result.total_processed == 3
            assert result.success_rate == 100.0
    
    @pytest.mark.asyncio
    async def test_bulk_subproduct_ingestion_with_batching(self, port_token, sample_subproducts):
        """Test bulk ingestion of subproducts with small batch size."""
        with aioresponses() as m:
            # Mock two batch requests (batch_size=2)
            m.post(
                "https://api.us.getport.io/v1/blueprints/armorcodeSubProduct/entities/bulk",
                payload={"message": "Success"},
                status=200
            )
            m.post(
                "https://api.us.getport.io/v1/blueprints/armorcodeSubProduct/entities/bulk", 
                payload={"message": "Success"},
                status=200
            )
            
            async with BulkPortManager(port_token) as bulk_manager:
                result = await bulk_manager.create_entities_bulk(
                    "armorcodeSubProduct",
                    sample_subproducts,
                    batch_size=2
                )
            
            assert result.success_count == 3
            assert result.failure_count == 0
            assert result.total_processed == 3
    
    @pytest.mark.asyncio
    async def test_bulk_ingestion_with_partial_failures(self, port_token, sample_products):
        """Test bulk ingestion handling partial failures gracefully."""
        with aioresponses() as m:
            # Mock partial failure response
            m.post(
                "https://api.us.getport.io/v1/blueprints/armorcodeProduct/entities/bulk",
                payload={
                    "entities": [
                        {"success": True},
                        {"success": False, "error": "Duplicate identifier"},
                        {"success": True}
                    ]
                },
                status=200
            )
            
            async with BulkPortManager(port_token) as bulk_manager:
                result = await bulk_manager.create_entities_bulk(
                    "armorcodeProduct",
                    sample_products
                )
            
            assert result.success_count == 2
            assert result.failure_count == 1
            assert result.total_processed == 3
            assert abs(result.success_rate - 66.67) < 0.1  # 2/3 * 100, approximately
            
            # Check that the failed entity is properly tracked
            failed_entity_ids = [entity_id for entity_id, _ in result.failed_entities]
            assert "2" in failed_entity_ids
    
    @pytest.mark.asyncio
    async def test_bulk_ingestion_error_recovery(self, port_token):
        """Test that bulk ingestion continues processing after batch failures."""
        # Create entities that will be split into 2 batches
        entities = [
            {"identifier": f"entity{i}", "title": f"Entity {i}"}
            for i in range(1, 5)
        ]
        
        with aioresponses() as m:
            # First batch fails completely
            m.post(
                "https://api.us.getport.io/v1/blueprints/testBlueprint/entities/bulk",
                payload={"message": "Server error"},
                status=500
            )
            # Second batch succeeds
            m.post(
                "https://api.us.getport.io/v1/blueprints/testBlueprint/entities/bulk",
                payload={"message": "Success"},
                status=200
            )
            
            async with BulkPortManager(port_token) as bulk_manager:
                result = await bulk_manager.create_entities_bulk(
                    "testBlueprint",
                    entities,
                    batch_size=2
                )
            
            # Should have 2 successful (second batch) and 2 failed (first batch)
            assert result.success_count == 2
            assert result.failure_count == 2
            assert result.total_processed == 4
            assert result.success_rate == 50.0
    
    @pytest.mark.asyncio
    async def test_performance_comparison_simulation(self, port_token):
        """Simulate performance improvement over individual entity creation."""
        # Create a larger dataset to demonstrate batching benefits
        entities = [
            {"identifier": f"entity{i}", "title": f"Entity {i}"}
            for i in range(1, 21)  # 20 entities = 1 batch
        ]
        
        with aioresponses() as m:
            # Mock single bulk request (vs 20 individual requests)
            m.post(
                "https://api.us.getport.io/v1/blueprints/testBlueprint/entities/bulk",
                payload={"message": "Success"},
                status=200
            )
            
            async with BulkPortManager(port_token) as bulk_manager:
                result = await bulk_manager.create_entities_bulk(
                    "testBlueprint",
                    entities
                )
            
            assert result.success_count == 20
            assert result.failure_count == 0
            assert result.total_processed == 20
            
            # In real usage, this would be 1 API call instead of 20
            # representing a significant performance improvement
    
    @pytest.mark.asyncio
    async def test_empty_entity_list_handling(self, port_token):
        """Test that empty entity lists are handled gracefully."""
        async with BulkPortManager(port_token) as bulk_manager:
            result = await bulk_manager.create_entities_bulk(
                "testBlueprint",
                []
            )
        
        assert result.success_count == 0
        assert result.failure_count == 0
        assert result.total_processed == 0
        assert result.success_rate == 0.0
    
    def test_bulk_result_summary_formatting(self):
        """Test that BulkResult provides useful summary information."""
        result = BulkResult(
            successful_entities=["entity1", "entity3", "entity5"],
            failed_entities=[
                ("entity2", "Validation error"),
                ("entity4", "Duplicate identifier")
            ],
            total_processed=5
        )
        
        # Test that the result provides useful metrics
        assert result.success_count == 3
        assert result.failure_count == 2
        assert result.success_rate == 60.0
        
        # Test that failed entities include error details
        failed_ids = [entity_id for entity_id, _ in result.failed_entities]
        failed_errors = [error for _, error in result.failed_entities]
        
        assert "entity2" in failed_ids
        assert "entity4" in failed_ids
        assert "Validation error" in failed_errors
        assert "Duplicate identifier" in failed_errors


if __name__ == "__main__":
    pytest.main([__file__])