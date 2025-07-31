"""
Unit tests for Step Executor

Tests for ExecutionContext, StepExecutor, and StepResult classes
with comprehensive coverage of step selection, execution context, and result tracking.
"""

import asyncio
import json
import os
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
from dataclasses import dataclass
from typing import Dict, Any, List

from step_executor import (
    ExecutionContext, 
    StepExecutor, 
    StepResult, 
    StepStatus
)
from cli_controller import CLIConfig


@pytest.fixture
def mock_cli_config():
    """Create a mock CLI configuration for testing."""
    return CLIConfig(
        steps=["product", "subproduct"],
        after_key=None,
        finding_filters_path=None,
        batch_size=10,
        retry_attempts=3,
        dry_run=False
    )


@pytest.fixture
def mock_port_token():
    """Mock Port API token."""
    return "mock-port-token-12345"


@pytest.fixture
def mock_execution_context(mock_cli_config, mock_port_token):
    """Create a mock execution context for testing."""
    with patch.dict(os.environ, {"ARMORCODE_API_KEY": "mock-api-key"}):
        context = ExecutionContext(
            config=mock_cli_config,
            port_token=mock_port_token
        )
        return context


class TestStepResult:
    """Test cases for StepResult dataclass."""
    
    def test_step_result_initialization(self):
        """Test StepResult initialization with required fields."""
        result = StepResult(
            step_name="product",
            status=StepStatus.NOT_STARTED,
            success=False
        )
        
        assert result.step_name == "product"
        assert result.status == StepStatus.NOT_STARTED
        assert result.success is False
        assert result.message == ""
        assert result.error is None
        assert result.entities_processed == 0
        assert result.entities_successful == 0
        assert result.entities_failed == 0
        assert result.execution_time == 0.0
        assert result.metadata == {}
    
    def test_step_result_success_rate_zero_processed(self):
        """Test success rate calculation when no entities processed."""
        result = StepResult(
            step_name="test",
            status=StepStatus.COMPLETED,
            success=True,
            entities_processed=0,
            entities_successful=0
        )
        
        assert result.success_rate == 0.0
    
    def test_step_result_success_rate_calculation(self):
        """Test success rate calculation with processed entities."""
        result = StepResult(
            step_name="test",
            status=StepStatus.COMPLETED,
            success=True,
            entities_processed=100,
            entities_successful=85,
            entities_failed=15
        )
        
        assert result.success_rate == 85.0
    
    def test_step_result_with_metadata(self):
        """Test StepResult with custom metadata."""
        metadata = {"custom_field": "value", "count": 42}
        result = StepResult(
            step_name="finding",
            status=StepStatus.COMPLETED,
            success=True,
            metadata=metadata
        )
        
        assert result.metadata == metadata
        assert result.metadata["custom_field"] == "value"
        assert result.metadata["count"] == 42


class TestExecutionContext:
    """Test cases for ExecutionContext dataclass."""
    
    def test_execution_context_initialization(self, mock_cli_config, mock_port_token):
        """Test ExecutionContext initialization."""
        context = ExecutionContext(
            config=mock_cli_config,
            port_token=mock_port_token
        )
        
        assert context.config == mock_cli_config
        assert context.port_token == mock_port_token
        assert context.armorcode_client is None
        assert context.direct_armorcode_client is None
        assert context.bulk_port_manager is not None
        assert context.retry_manager is not None
        assert context.filter_manager is None  # No filters path provided
        assert context.step_results == {}
        assert context.completed_steps == set()
        assert context.port_api_url == "https://api.us.getport.io/v1"
        # Check that blueprints_path ends with the correct relative path
        assert str(context.blueprints_path).endswith("armorcode_integration/blueprints")
    
    def test_execution_context_with_filters(self, mock_port_token):
        """Test ExecutionContext initialization with filter manager."""
        config = CLIConfig(
            steps=["finding"],
            after_key=None,
            finding_filters_path="test_filters.json",
            batch_size=10,
            retry_attempts=3,
            dry_run=False
        )
        
        context = ExecutionContext(
            config=config,
            port_token=mock_port_token
        )
        
        assert context.filter_manager is not None
    
    def test_step_completion_tracking(self, mock_execution_context):
        """Test step completion tracking methods."""
        context = mock_execution_context
        
        # Initially no steps completed
        assert not context.is_step_completed("product")
        assert context.completed_steps == set()
        
        # Mark step as completed
        context.mark_step_completed("product")
        assert context.is_step_completed("product")
        assert "product" in context.completed_steps
        
        # Mark another step
        context.mark_step_completed("subproduct")
        assert context.is_step_completed("subproduct")
        assert context.completed_steps == {"product", "subproduct"}
    
    def test_step_result_storage_and_retrieval(self, mock_execution_context):
        """Test step result storage and retrieval."""
        context = mock_execution_context
        
        # Initially no results
        assert context.get_step_result("product") is None
        
        # Store a result
        result = StepResult(
            step_name="product",
            status=StepStatus.COMPLETED,
            success=True
        )
        context.step_results["product"] = result
        
        # Retrieve the result
        retrieved_result = context.get_step_result("product")
        assert retrieved_result == result
        assert retrieved_result.step_name == "product"
    
    @pytest.mark.asyncio
    async def test_execution_context_async_context_manager(self, mock_cli_config, mock_port_token):
        """Test ExecutionContext as async context manager."""
        with patch.dict(os.environ, {"ARMORCODE_API_KEY": "mock-api-key"}):
            with patch('step_executor.ArmorCodeClient') as mock_ac_client, \
                 patch('step_executor.DirectArmorCodeClient') as mock_direct_client, \
                 patch('step_executor.BulkPortManager') as mock_bulk_manager:
                
                # Setup mocks
                mock_ac_instance = AsyncMock()
                mock_direct_instance = AsyncMock()
                mock_bulk_instance = AsyncMock()
                
                mock_ac_client.return_value = mock_ac_instance
                mock_direct_client.return_value = mock_direct_instance
                mock_bulk_manager.return_value = mock_bulk_instance
                
                context = ExecutionContext(
                    config=mock_cli_config,
                    port_token=mock_port_token
                )
                
                # Test context manager entry
                async with context as ctx:
                    assert ctx == context
                    assert context.armorcode_client == mock_ac_instance
                    assert context.direct_armorcode_client == mock_direct_instance
                    
                    # Verify clients were initialized
                    mock_ac_instance.__aenter__.assert_called_once()
                    mock_direct_instance.__aenter__.assert_called_once()
                    mock_bulk_instance.__aenter__.assert_called_once()
                
                # Verify cleanup was called
                mock_ac_instance.__aexit__.assert_called_once()
                mock_direct_instance.__aexit__.assert_called_once()
                mock_bulk_instance.__aexit__.assert_called_once()
    
    def test_execution_context_missing_api_key(self, mock_cli_config, mock_port_token):
        """Test ExecutionContext fails when ARMORCODE_API_KEY is missing."""
        with patch.dict(os.environ, {}, clear=True):
            context = ExecutionContext(
                config=mock_cli_config,
                port_token=mock_port_token
            )
            
            with pytest.raises(ValueError, match="ARMORCODE_API_KEY must be set"):
                asyncio.run(context.__aenter__())


class TestStepExecutor:
    """Test cases for StepExecutor class."""
    
    def test_step_executor_initialization(self):
        """Test StepExecutor initialization."""
        executor = StepExecutor()
        
        assert executor.AVAILABLE_STEPS == ["product", "subproduct", "finding"]
        assert len(executor._step_registry) == 3
        assert "product" in executor._step_registry
        assert "subproduct" in executor._step_registry
        assert "finding" in executor._step_registry
    
    def test_get_available_steps(self):
        """Test getting available steps."""
        executor = StepExecutor()
        steps = executor.get_available_steps()
        
        assert steps == ["product", "subproduct", "finding"]
        # Ensure it returns a copy
        steps.append("invalid")
        assert executor.get_available_steps() == ["product", "subproduct", "finding"]
    
    def test_validate_steps_all_valid(self):
        """Test step validation with all valid steps."""
        executor = StepExecutor()
        
        valid_steps = ["product", "subproduct"]
        invalid_steps = executor.validate_steps(valid_steps)
        
        assert invalid_steps == []
    
    def test_validate_steps_some_invalid(self):
        """Test step validation with some invalid steps."""
        executor = StepExecutor()
        
        mixed_steps = ["product", "invalid_step", "subproduct", "another_invalid"]
        invalid_steps = executor.validate_steps(mixed_steps)
        
        assert set(invalid_steps) == {"invalid_step", "another_invalid"}
    
    def test_validate_steps_all_invalid(self):
        """Test step validation with all invalid steps."""
        executor = StepExecutor()
        
        invalid_steps_input = ["invalid1", "invalid2"]
        invalid_steps = executor.validate_steps(invalid_steps_input)
        
        assert invalid_steps == invalid_steps_input
    
    @pytest.mark.asyncio
    async def test_execute_steps_invalid_steps(self, mock_execution_context):
        """Test execute_steps with invalid step names."""
        executor = StepExecutor()
        
        with pytest.raises(ValueError, match="Invalid step names: invalid_step"):
            await executor.execute_steps(mock_execution_context, ["product", "invalid_step"])
    
    @pytest.mark.asyncio
    async def test_execute_steps_already_completed(self, mock_execution_context):
        """Test execute_steps skips already completed steps."""
        executor = StepExecutor()
        context = mock_execution_context
        
        # Mark product step as already completed
        context.mark_step_completed("product")
        
        with patch.object(executor, '_execute_product_step') as mock_product_step:
            results = await executor.execute_steps(context, ["product"])
            
            # Should not call the step function
            mock_product_step.assert_not_called()
            assert results == {}
    
    @pytest.mark.asyncio
    async def test_execute_steps_successful_execution(self, mock_execution_context):
        """Test successful step execution."""
        executor = StepExecutor()
        context = mock_execution_context
        
        # Mock the step execution method
        async def mock_product_step(ctx, result):
            result.success = True
            result.message = "Product step completed"
            result.entities_processed = 5
            result.entities_successful = 5
        
        with patch.object(executor, '_execute_product_step', new=mock_product_step):
            results = await executor.execute_steps(context, ["product"])
            
            assert "product" in results
            result = results["product"]
            assert result.success is True
            assert result.status == StepStatus.COMPLETED
            assert result.message == "Product step completed"
            assert context.is_step_completed("product")
    
    @pytest.mark.asyncio
    async def test_execute_steps_failed_execution(self, mock_execution_context):
        """Test failed step execution."""
        executor = StepExecutor()
        context = mock_execution_context
        
        # Mock the step execution method to fail
        async def mock_product_step(ctx, result):
            result.success = False
            result.message = "Product step failed"
        
        with patch.object(executor, '_execute_product_step', new=mock_product_step):
            results = await executor.execute_steps(context, ["product"])
            
            assert "product" in results
            result = results["product"]
            assert result.success is False
            assert result.status == StepStatus.FAILED
            assert result.message == "Product step failed"
            assert not context.is_step_completed("product")
    
    @pytest.mark.asyncio
    async def test_execute_steps_exception_handling(self, mock_execution_context):
        """Test step execution with exception handling."""
        executor = StepExecutor()
        context = mock_execution_context
        
        # Mock the step execution method to raise an exception
        test_exception = Exception("Test exception")
        
        async def mock_product_step(ctx, result):
            raise test_exception
        
        with patch.object(executor, '_execute_product_step', side_effect=test_exception):
            results = await executor.execute_steps(context, ["product"])
            
            assert "product" in results
            result = results["product"]
            assert result.success is False
            assert result.status == StepStatus.FAILED
            assert result.error == test_exception
            assert "Unexpected error: Test exception" in result.message
            assert not context.is_step_completed("product")
    
    @pytest.mark.asyncio
    async def test_execute_steps_ordered_execution(self, mock_execution_context):
        """Test that steps are executed in the correct order."""
        executor = StepExecutor()
        context = mock_execution_context
        
        execution_order = []
        
        async def mock_finding_step(ctx, result):
            execution_order.append("finding")
            result.success = True
            
        async def mock_product_step(ctx, result):
            execution_order.append("product")
            result.success = True
            
        async def mock_subproduct_step(ctx, result):
            execution_order.append("subproduct")
            result.success = True
        
        with patch.object(executor, '_execute_finding_step', new=mock_finding_step), \
             patch.object(executor, '_execute_product_step', new=mock_product_step), \
             patch.object(executor, '_execute_subproduct_step', new=mock_subproduct_step):
            
            # Request steps in different order than AVAILABLE_STEPS
            await executor.execute_steps(context, ["finding", "product", "subproduct"])
            
            # Should execute in AVAILABLE_STEPS order
            assert execution_order == ["product", "subproduct", "finding"]


class TestStepExecutorIntegration:
    """Integration tests for step execution methods."""
    
    @pytest.mark.asyncio
    async def test_setup_blueprint_success(self, mock_execution_context):
        """Test successful blueprint setup."""
        executor = StepExecutor()
        context = mock_execution_context
        
        # Mock blueprint file content
        blueprint_data = {
            "identifier": "testBlueprint",
            "title": "Test Blueprint"
        }
        
        with patch("builtins.open", mock_open(read_data=json.dumps(blueprint_data))), \
             patch("pathlib.Path.exists", return_value=True), \
             patch("aiohttp.ClientSession") as mock_session:
            
            # Mock successful response
            mock_response = AsyncMock()
            mock_response.status = 201
            mock_session_instance = AsyncMock()
            mock_session_instance.post.return_value.__aenter__.return_value = mock_response
            mock_session.return_value.__aenter__.return_value = mock_session_instance
            
            await executor._setup_blueprint(context, "test.json")
            
            # Verify the request was made
            mock_session_instance.post.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_setup_blueprint_dry_run(self, mock_execution_context):
        """Test blueprint setup in dry run mode."""
        executor = StepExecutor()
        context = mock_execution_context
        context.config.dry_run = True
        
        blueprint_data = {"identifier": "testBlueprint"}
        
        with patch("builtins.open", mock_open(read_data=json.dumps(blueprint_data))), \
             patch("pathlib.Path.exists", return_value=True), \
             patch("aiohttp.ClientSession") as mock_session:
            
            await executor._setup_blueprint(context, "test.json")
            
            # Should not make any HTTP requests in dry run
            mock_session.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_setup_blueprint_file_not_found(self, mock_execution_context):
        """Test blueprint setup with missing file."""
        executor = StepExecutor()
        context = mock_execution_context
        
        with patch("pathlib.Path.exists", return_value=False):
            with pytest.raises(FileNotFoundError, match="Blueprint file not found"):
                await executor._setup_blueprint(context, "missing.json")


if __name__ == "__main__":
    pytest.main([__file__])