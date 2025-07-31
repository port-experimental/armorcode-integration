"""
End-to-End Integration Tests for Enhanced ArmorCode-Port Integration

This module provides comprehensive integration tests for the complete enhanced pipeline execution,
testing the main execution flow with CLIController, StepExecutor, and all enhanced components.
"""

import asyncio
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from cli_controller import CLIController, CLIConfig
from step_executor import StepExecutor, ExecutionContext, StepStatus
from retry_manager import RetryManager
from bulk_port_manager import BulkPortManager, BulkResult


class TestMainIntegration:
    """Test suite for end-to-end integration of the enhanced pipeline."""
    
    @pytest.fixture
    def mock_env_vars(self):
        """Mock environment variables for testing."""
        with patch.dict(os.environ, {
            'PORT_CLIENT_ID': 'test_client_id',
            'PORT_CLIENT_SECRET': 'test_client_secret',
            'ARMORCODE_API_KEY': 'test_api_key'
        }):
            yield
    
    @pytest.fixture
    def sample_config(self):
        """Sample CLI configuration for testing."""
        return CLIConfig(
            steps=['product', 'subproduct'],
            after_key=None,
            finding_filters_path=None,
            batch_size=10,
            retry_attempts=2,
            dry_run=True
        )
    
    @pytest.fixture
    def mock_blueprints_dir(self):
        """Create temporary blueprints directory with sample files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            blueprints_path = Path(temp_dir) / "blueprints"
            blueprints_path.mkdir()
            
            # Create sample blueprint files
            blueprints = {
                "product.json": {
                    "identifier": "armorcodeProduct",
                    "title": "ArmorCode Product",
                    "properties": {}
                },
                "subproduct.json": {
                    "identifier": "armorcodeSubProduct", 
                    "title": "ArmorCode Sub Product",
                    "properties": {}
                },
                                "finding.json": {
                    "identifier": "armorcodeFinding",
                    "title": "ArmorCode Finding",
                    "properties": {}
                }
            }
            
            for filename, content in blueprints.items():
                with open(blueprints_path / filename, 'w') as f:
                    json.dump(content, f)
            
            yield blueprints_path
    
    @pytest.mark.asyncio
    async def test_cli_controller_integration(self):
        """Test CLIController parsing and validation."""
        controller = CLIController()
        
        # Test default configuration
        config = controller.parse_arguments([])
        assert config.steps == controller.AVAILABLE_STEPS
        assert config.batch_size == controller.DEFAULT_BATCH_SIZE
        assert config.retry_attempts == controller.DEFAULT_RETRY_ATTEMPTS
        assert not config.dry_run
        
        # Test custom configuration
        config = controller.parse_arguments([
            '--steps', 'product,finding',
            '--batch-size', '15',
            '--retry-attempts', '5',
            '--dry-run'
        ])
        assert config.steps == ['product', 'finding']
        assert config.batch_size == 15
        assert config.retry_attempts == 5
        assert config.dry_run
    
    @pytest.mark.asyncio
    async def test_step_executor_integration(self, sample_config, mock_env_vars):
        """Test StepExecutor with mocked execution context."""
        step_executor = StepExecutor()
        
        # Mock the execution context
        with patch('step_executor.ExecutionContext') as mock_context_class:
            mock_context = AsyncMock()
            mock_context_class.return_value = mock_context
            mock_context.__aenter__.return_value = mock_context
            mock_context.config = sample_config
            mock_context.port_token = "test_token"
            mock_context.is_step_completed.return_value = False
            mock_context.step_results = {}
            mock_context.completed_steps = set()
            mock_context.mark_step_completed = MagicMock()
            mock_context.get_step_result.return_value = None
            # Set blueprints_path to match the actual implementation
            mock_context.blueprints_path = Path(__file__).parent.parent.parent / "src" / "armorcode_integration" / "blueprints"
            mock_context.port_api_url = "https://api.us.getport.io/v1"
            
            # Mock ArmorCode client responses
            mock_context.armorcode_client.get_all_products.return_value = [
                {"id": 1, "name": "Test Product", "description": "Test Description"}
            ]
            mock_context.armorcode_client.get_all_subproducts.return_value = [
                {"id": 1, "name": "Test SubProduct", "parent": 1, "technologies": "Python,JavaScript"}
            ]
            
            # Mock bulk manager
            mock_bulk_result = BulkResult(
                successful_entities=["1"],
                failed_entities=[],
                total_processed=1
            )
            mock_context.bulk_port_manager.create_entities_bulk.return_value = mock_bulk_result
            
            # Execute steps
            results = await step_executor.execute_steps(mock_context, ['product', 'subproduct'])
            
            # Verify results
            assert len(results) == 2
            assert 'product' in results
            assert 'subproduct' in results
            assert results['product'].success
            assert results['subproduct'].success
            assert results['product'].status == StepStatus.COMPLETED
            assert results['subproduct'].status == StepStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_blueprint_setup_with_retry(self, mock_env_vars):
        """Test blueprint setup with retry logic."""
        from main import setup_blueprints_with_retry
        
        retry_manager = RetryManager(max_attempts=2)
        
        # Mock successful blueprint creation
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value.__aenter__.return_value = mock_session
            
            mock_response = AsyncMock()
            mock_response.status = 201
            mock_session.post.return_value.__aenter__.return_value = mock_response
            
            # Mock blueprint files
            with patch('pathlib.Path.exists', return_value=True), \
                 patch('builtins.open', mock_open_with_json({"identifier": "test"})):
                
                await setup_blueprints_with_retry("test_token", retry_manager, dry_run=False)
                
                # Verify API calls were made
                assert mock_session.post.called
    
    @pytest.mark.asyncio
    async def test_execution_context_lifecycle(self, sample_config, mock_env_vars):
        """Test ExecutionContext async context manager lifecycle."""
        with patch('step_executor.ArmorCodeClient') as mock_ac_client, \
             patch('step_executor.DirectArmorCodeClient') as mock_direct_client, \
             patch('step_executor.BulkPortManager') as mock_bulk_manager:
            
            # Setup mocks
            mock_ac_instance = AsyncMock()
            mock_ac_client.return_value = mock_ac_instance
            
            mock_direct_instance = AsyncMock()
            mock_direct_client.return_value = mock_direct_instance
            
            mock_bulk_instance = AsyncMock()
            mock_bulk_manager.return_value = mock_bulk_instance
            
            # Test context manager
            async with ExecutionContext(config=sample_config, port_token="test_token") as context:
                assert context.config == sample_config
                assert context.port_token == "test_token"
                assert context.armorcode_client is not None
                assert context.direct_armorcode_client is not None
                assert context.bulk_port_manager is not None
            
            # Verify cleanup was called
            mock_ac_instance.__aexit__.assert_called_once()
            mock_direct_instance.__aexit__.assert_called_once()
            mock_bulk_instance.__aexit__.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_main_function_dry_run(self, mock_env_vars, mock_blueprints_dir):
        """Test main function in dry-run mode."""
        from main import main
        
        # Mock CLI arguments for dry run
        test_args = ['--dry-run', '--steps', 'product']
        
        with patch('sys.argv', ['main.py'] + test_args), \
             patch('pathlib.Path.__truediv__', return_value=mock_blueprints_dir), \
             patch('main.get_port_api_token', return_value="test_token"), \
             patch('step_executor.ArmorCodeClient') as mock_ac_client, \
             patch('step_executor.DirectArmorCodeClient') as mock_direct_client, \
             patch('step_executor.BulkPortManager') as mock_bulk_manager:
            
            # Setup mocks
            mock_ac_instance = AsyncMock()
            mock_ac_client.return_value = mock_ac_instance
            mock_ac_instance.get_all_products.return_value = [
                {"id": 1, "name": "Test Product"}
            ]
            
            mock_direct_instance = AsyncMock()
            mock_direct_client.return_value = mock_direct_instance
            
            mock_bulk_instance = AsyncMock()
            mock_bulk_manager.return_value = mock_bulk_instance
            
            # Run main function
            await main()
            
            # Verify dry run behavior - no actual API calls to Port
            mock_bulk_instance.create_entities_bulk.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, sample_config, mock_env_vars):
        """Test error handling and recovery mechanisms."""
        step_executor = StepExecutor()
        
        with patch('step_executor.ExecutionContext') as mock_context_class:
            mock_context = AsyncMock()
            mock_context_class.return_value = mock_context
            mock_context.__aenter__.return_value = mock_context
            mock_context.config = sample_config
            mock_context.is_step_completed.return_value = False
            mock_context.step_results = {}
            mock_context.completed_steps = set()
            
            # Mock ArmorCode client to raise an exception
            mock_context.armorcode_client.get_all_products.side_effect = Exception("API Error")
            mock_context.mark_step_completed = MagicMock()
            mock_context.get_step_result.return_value = None
            
            # Execute steps and expect graceful error handling
            results = await step_executor.execute_steps(mock_context, ['product'])
            
            # Verify error was handled gracefully
            assert 'product' in results
            assert not results['product'].success
            assert results['product'].status == StepStatus.FAILED
            assert "API Error" in results['product'].message
    
    @pytest.mark.asyncio
    async def test_filtering_integration(self, mock_env_vars):
        """Test integration with filtering functionality."""
        # Create temporary filter file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filter_data = {
                "severity": ["HIGH", "CRITICAL"],
                "status": ["OPEN"]
            }
            json.dump(filter_data, f)
            filter_file_path = f.name
        
        try:
            # Test CLI parsing with filter file
            controller = CLIController()
            config = controller.parse_arguments([
                '--finding-filters', filter_file_path,
                '--after-key', '12345'
            ])
            
            assert config.finding_filters_path == filter_file_path
            assert config.after_key == 12345
            
        finally:
            # Clean up temporary file
            os.unlink(filter_file_path)
    
    @pytest.mark.asyncio
    async def test_performance_metrics_collection(self, sample_config, mock_env_vars):
        """Test that performance metrics are properly collected and reported."""
        step_executor = StepExecutor()
        
        with patch('step_executor.ExecutionContext') as mock_context_class:
            mock_context = AsyncMock()
            mock_context_class.return_value = mock_context
            mock_context.__aenter__.return_value = mock_context
            mock_context.config = sample_config
            mock_context.is_step_completed.return_value = False
            mock_context.step_results = {}
            mock_context.completed_steps = set()
            
            # Mock successful execution with timing
            mock_context.armorcode_client.get_all_products.return_value = [
                {"id": i, "name": f"Product {i}"} for i in range(100)
            ]
            
            mock_bulk_result = BulkResult(
                successful_entities=[str(i) for i in range(95)],
                failed_entities=[(str(i), "Test error") for i in range(95, 100)],
                total_processed=100
            )
            mock_context.bulk_port_manager.create_entities_bulk.return_value = mock_bulk_result
            
            # Execute steps
            results = await step_executor.execute_steps(mock_context, ['product'])
            
            # Verify performance metrics are captured
            product_result = results['product']
            assert product_result.entities_processed == 100
            assert product_result.entities_successful == 95
            assert product_result.entities_failed == 5
            assert product_result.success_rate == 95.0
            assert product_result.execution_time > 0
    
    def test_cli_validation_errors(self):
        """Test CLI validation error handling."""
        controller = CLIController()
        
        # Test invalid step names
        with pytest.raises(SystemExit):
            controller.parse_arguments(['--steps', 'invalid_step'])
        
        # Test invalid batch size
        with pytest.raises(SystemExit):
            controller.parse_arguments(['--batch-size', '0'])
        
        # Test invalid retry attempts
        with pytest.raises(SystemExit):
            controller.parse_arguments(['--retry-attempts', '-1'])
        
        # Test invalid after key
        with pytest.raises(SystemExit):
            controller.parse_arguments(['--after-key', '-1'])


def mock_open_with_json(json_data):
    """Helper function to mock file opening with JSON data."""
    from unittest.mock import mock_open
    return mock_open(read_data=json.dumps(json_data))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])