"""
Step Executor for ArmorCode Integration

This module provides the StepExecutor class for managing selective step execution
with shared execution context and result tracking.
"""

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from pathlib import Path

from acsdk import ArmorCodeClient
from ..clients.armorcode_client import DirectArmorCodeClient
from ..managers.bulk_port_manager import BulkPortManager
from ..managers.retry_manager import RetryManager
from ..managers.filter_manager import FilterManager
from ..cli.cli_controller import CLIConfig
from ..managers.logging_manager import get_logging_manager, OperationType, LoggingManager
from ..managers.progress_tracker import create_progress_tracker
from ..utils.data_transformers import transform_finding_timestamps


class StepStatus(Enum):
    """Status of step execution."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StepResult:
    """Result of a step execution with success/failure reporting."""
    step_name: str
    status: StepStatus
    success: bool
    message: str = ""
    error: Optional[Exception] = None
    entities_processed: int = 0
    entities_successful: int = 0
    entities_failed: int = 0
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.entities_processed == 0:
            return 0.0
        return (self.entities_successful / self.entities_processed) * 100


@dataclass
class ExecutionContext:
    """Execution context containing all shared resources and configuration."""
    # Configuration
    config: CLIConfig
    
    # API Clients
    port_token: str
    armorcode_client: Optional[ArmorCodeClient] = None
    direct_armorcode_client: Optional[DirectArmorCodeClient] = None
    
    # Managers
    bulk_port_manager: Optional[BulkPortManager] = None
    retry_manager: Optional[RetryManager] = None
    filter_manager: Optional[FilterManager] = None
    
    # Runtime state
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))
    step_results: Dict[str, StepResult] = field(default_factory=dict)
    completed_steps: Set[str] = field(default_factory=set)
    logging_manager: LoggingManager = field(default_factory=get_logging_manager)
    
    # Constants
    port_api_url: str = "https://api.us.getport.io/v1"
    blueprints_path: Path = field(default_factory=lambda: Path(__file__).parent.parent / "blueprints")
    
    def __post_init__(self):
        """Initialize managers after dataclass creation."""
        # Initialize retry manager
        self.retry_manager = RetryManager(
            max_attempts=self.config.retry_attempts,
            base_delay=1.0,
            max_delay=60.0
        )
        
        # Initialize bulk port manager
        self.bulk_port_manager = BulkPortManager(
            port_token=self.port_token,
            port_api_url=self.port_api_url
        )
        
        # Initialize filter manager only if needed
        self.filter_manager = FilterManager() if (self.config.finding_filters_path or self.config.after_key is not None) else None
    
    async def __aenter__(self):
        """Async context manager entry."""
        # Initialize ArmorCode clients
        api_key = os.getenv("ARMORCODE_API_KEY")
        if not api_key:
            raise ValueError("ARMORCODE_API_KEY must be set")
        
        self.armorcode_client = ArmorCodeClient(api_key)
        await self.armorcode_client.__aenter__()
        
        self.direct_armorcode_client = DirectArmorCodeClient(api_key)
        await self.direct_armorcode_client.__aenter__()
        
        # Initialize bulk port manager
        await self.bulk_port_manager.__aenter__()
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.armorcode_client:
            await self.armorcode_client.__aexit__(exc_type, exc_val, exc_tb)
        
        if self.direct_armorcode_client:
            await self.direct_armorcode_client.__aexit__(exc_type, exc_val, exc_tb)
        
        if self.bulk_port_manager:
            await self.bulk_port_manager.__aexit__(exc_type, exc_val, exc_tb)
    
    def is_step_completed(self, step_name: str) -> bool:
        """Check if a step has been completed successfully."""
        return step_name in self.completed_steps
    
    def mark_step_completed(self, step_name: str):
        """Mark a step as completed."""
        self.completed_steps.add(step_name)
    
    def get_step_result(self, step_name: str) -> Optional[StepResult]:
        """Get the result of a specific step."""
        return self.step_results.get(step_name)


class StepExecutor:
    """Step executor with selective step execution logic and result tracking."""
    
    # Available steps in execution order
    AVAILABLE_STEPS = ["product", "subproduct", "finding"]
    
    def __init__(self):
        """Initialize the step executor."""
        self.logging_manager = get_logging_manager()
        self.logger = self.logging_manager.get_logger("StepExecutor")
        self._step_registry: Dict[str, Callable] = {}
        self._register_steps()
    
    def _register_steps(self):
        """Register step execution functions in the step registry."""
        self._step_registry = {
            "product": self._execute_product_step,
            "subproduct": self._execute_subproduct_step,
            "finding": self._execute_finding_step,
        }
    
    def get_available_steps(self) -> List[str]:
        """Get list of available steps."""
        return self.AVAILABLE_STEPS.copy()
    
    def validate_steps(self, steps: List[str]) -> List[str]:
        """
        Validate step names and return any invalid ones.
        
        Args:
            steps: List of step names to validate
            
        Returns:
            List of invalid step names (empty if all valid)
        """
        return [step for step in steps if step not in self._step_registry]
    
    async def execute_steps(self, context: ExecutionContext, steps: List[str]) -> Dict[str, StepResult]:
        """
        Execute specified steps with the given execution context.
        
        Args:
            context: Execution context with shared resources
            steps: List of step names to execute
            
        Returns:
            Dictionary mapping step names to their results
        """
        self.logger.info(
            f"Starting execution of steps: {', '.join(steps)}",
            steps=steps,
            correlation_id=context.logging_manager.correlation_id
        )
        
        # Validate steps
        invalid_steps = self.validate_steps(steps)
        if invalid_steps:
            self.logger.error(
                f"Invalid step names provided: {', '.join(invalid_steps)}",
                invalid_steps=invalid_steps,
                valid_steps=self.AVAILABLE_STEPS
            )
            raise ValueError(f"Invalid step names: {', '.join(invalid_steps)}")
        
        # Execute steps in the order they appear in AVAILABLE_STEPS
        ordered_steps = [step for step in self.AVAILABLE_STEPS if step in steps]
        
        for step_name in ordered_steps:
            if context.is_step_completed(step_name):
                self.logger.info(
                    f"Step '{step_name}' already completed, skipping",
                    step_name=step_name,
                    status="skipped"
                )
                continue
            
            self.logger.info(
                f"Executing step: {step_name}",
                step_name=step_name,
                status="starting"
            )
            
            # Create initial result
            result = StepResult(
                step_name=step_name,
                status=StepStatus.IN_PROGRESS,
                success=False
            )
            context.step_results[step_name] = result
            
            try:
                # Execute the step
                step_function = self._step_registry[step_name]
                await step_function(context, result)
                
                # Mark as completed if successful
                if result.success:
                    result.status = StepStatus.COMPLETED
                    context.mark_step_completed(step_name)
                    self.logger.info(
                        f"Step '{step_name}' completed successfully",
                        step_name=step_name,
                        status="completed",
                        entities_processed=result.entities_processed,
                        entities_successful=result.entities_successful,
                        entities_failed=result.entities_failed,
                        success_rate=result.success_rate,
                        execution_time=result.execution_time
                    )
                else:
                    result.status = StepStatus.FAILED
                    troubleshooting_info = context.logging_manager.log_troubleshooting_info(
                        "StepExecutor", Exception(result.message), {"step_name": step_name}
                    )
                    self.logger.error(
                        f"Step '{step_name}' failed: {result.message}",
                        step_name=step_name,
                        status="failed",
                        error_message=result.message,
                        troubleshooting_info=troubleshooting_info
                    )
                
            except Exception as e:
                result.status = StepStatus.FAILED
                result.success = False
                result.error = e
                result.message = f"Unexpected error: {str(e)}"
                
                troubleshooting_info = context.logging_manager.log_troubleshooting_info(
                    "StepExecutor", e, {"step_name": step_name}
                )
                self.logger.error(
                    f"Step '{step_name}' failed with exception",
                    error=e,
                    step_name=step_name,
                    status="failed",
                    troubleshooting_info=troubleshooting_info
                )
        
        # Log execution summary
        successful_steps = [name for name, result in context.step_results.items() if result.success]
        failed_steps = [name for name, result in context.step_results.items() if not result.success]
        
        self.logger.info(
            "Step execution completed",
            total_steps=len(ordered_steps),
            successful_steps=len(successful_steps),
            failed_steps=len(failed_steps),
            success_rate=(len(successful_steps) / len(ordered_steps) * 100) if ordered_steps else 0,
            successful_step_names=successful_steps,
            failed_step_names=failed_steps
        )
        
        return context.step_results
    
    async def _execute_product_step(self, context: ExecutionContext, result: StepResult):
        """Execute the product ingestion step."""
        import time
        start_time = time.time()
        
        try:
            self.logger.info("Starting product ingestion...")
            
            # Setup blueprint first
            await self._setup_blueprint(context, "product.json")
            
            # Fetch products
            products = await context.armorcode_client.get_all_products()
            result.entities_processed = len(products)
            
            self.logger.info(f"Found {len(products)} products in ArmorCode")
            
            # Process products
            entities = []
            for product in products:
                try:
                    entity = {
                        "identifier": str(product["id"]),
                        "title": product.get("name"),
                        "properties": {
                            "name": product.get("name"),
                            "description": product.get("description"),
                            "businessOwner": product.get("businessOwnerName"),
                            "securityOwner": product.get("securityOwnerName"),
                        },
                    }
                    entities.append(entity)
                except Exception as e:
                    self.logger.error(f"Failed to process product {product.get('id')}: {e}")
            
            # Bulk upsert entities
            if not context.config.dry_run:
                bulk_result = await context.bulk_port_manager.create_entities_bulk(
                    "armorcodeProduct", entities, context.config.batch_size
                )
                result.entities_successful = bulk_result.success_count
                result.entities_failed = bulk_result.failure_count
            else:
                self.logger.info(f"[DRY RUN] Would upsert {len(entities)} product entities")
                result.entities_successful = len(entities)
                result.entities_failed = 0
            
            result.success = True
            result.message = f"Processed {result.entities_processed} products"
            
        except Exception as e:
            result.success = False
            result.message = f"Product step failed: {str(e)}"
            result.error = e
            raise
        finally:
            result.execution_time = time.time() - start_time
    
    async def _execute_subproduct_step(self, context: ExecutionContext, result: StepResult):
        """Execute the subproduct ingestion step."""
        import time
        start_time = time.time()
        
        try:
            self.logger.info("Starting subproduct ingestion...")
            
            # Setup blueprint first
            await self._setup_blueprint(context, "subproduct.json")
            
            # Fetch subproducts
            subproducts = await context.armorcode_client.get_all_subproducts()
            result.entities_processed = len(subproducts)
            
            self.logger.info(f"Found {len(subproducts)} subproducts in ArmorCode")
            
            # Process subproducts
            entities = []
            for subproduct in subproducts:
                try:
                    # Handle technologies field
                    technologies = []
                    if tech_str := subproduct.get("technologies"):
                        technologies = [tech.strip() for tech in tech_str.split(",")]
                    
                    entity = {
                        "identifier": str(subproduct["id"]),
                        "title": subproduct.get("name"),
                        "properties": {
                            "name": subproduct.get("name"),
                            "repoLink": subproduct.get("repoLink"),
                            "programmingLanguage": subproduct.get("programmingLanguage"),
                            "technologies": technologies,
                        },
                        "relations": {"product": str(subproduct["parent"])},
                    }
                    entities.append(entity)
                except Exception as e:
                    self.logger.error(f"Failed to process subproduct {subproduct.get('id')}: {e}")
            
            # Bulk upsert entities
            if not context.config.dry_run:
                bulk_result = await context.bulk_port_manager.create_entities_bulk(
                    "armorcodeSubProduct", entities, context.config.batch_size
                )
                result.entities_successful = bulk_result.success_count
                result.entities_failed = bulk_result.failure_count
            else:
                self.logger.info(f"[DRY RUN] Would upsert {len(entities)} subproduct entities")
                result.entities_successful = len(entities)
                result.entities_failed = 0
            
            result.success = True
            result.message = f"Processed {result.entities_processed} subproducts"
            
        except Exception as e:
            result.success = False
            result.message = f"Subproduct step failed: {str(e)}"
            result.error = e
            raise
        finally:
            result.execution_time = time.time() - start_time
    
    
    async def _execute_finding_step(self, context: ExecutionContext, result: StepResult):
        """Execute the finding ingestion step."""
        import time
        start_time = time.time()

        # Ensure DirectArmorCodeClient is initialized
        if context.direct_armorcode_client is None:
            raise RuntimeError(
                "DirectArmorCodeClient not initialized. Make sure to enter the ExecutionContext via 'async with'."
            )
        self.logger.debug(f"Using DirectArmorCodeClient: {context.direct_armorcode_client}")

        try:
            self.logger.info("Starting findings ingestion...")

            # Setup blueprint
            await self._setup_blueprint(context, "finding.json")

            # Prepare filters using FilterManager
            combined_filters = None
            try:
                if context.config.finding_filters_path or context.config.after_key is not None:
                    if context.filter_manager is None:
                        # Initialize filter manager if not already done
                        from ..managers.filter_manager import FilterManager
                        context.filter_manager = FilterManager()

                    # Load and validate filters from file if provided
                    file_filters = None
                    if context.config.finding_filters_path:
                        self.logger.info(f"Loading finding filters from: {context.config.finding_filters_path}")
                        file_filters = context.filter_manager.load_and_validate_filters(context.config.finding_filters_path)
                        self.logger.info(f"Successfully loaded and validated filters: {context.filter_manager.get_filter_summary(file_filters)}")

                    # Combine with afterKey parameter
                    combined_filters = context.filter_manager.combine_filters(file_filters, context.config.after_key)

                    if combined_filters:
                        self.logger.info(f"Applied filters: {context.filter_manager.get_filter_summary(combined_filters)}")

            except Exception as e:
                from ..managers.filter_manager import FilterValidationError
                if isinstance(e, FilterValidationError):
                    result.success = False
                    result.message = f"Filter validation failed: {e}"
                    self.logger.error(f"Filter validation failed: {e}")
                    return
                else:
                    result.success = False
                    result.message = f"Unexpected error while processing filters: {e}"
                    self.logger.error(f"Unexpected error while processing filters: {e}")
                    return

            # Fetch findings with optional filtering
            self.logger.info("Fetching all findings from ArmorCode (about to call get_all_findings)...")
            # Debug: record start of API call
            self.logger.debug("Calling get_all_findings() with filters: %s", combined_filters)
            findings = await context.direct_armorcode_client.get_all_findings(filters=combined_filters)
            result.entities_processed = len(findings)

            self.logger.info(f"Found {len(findings)} findings in ArmorCode")

            # Process findings
            finding_entities = []

            for idx, finding in enumerate(findings, start=1):
                try:
                    # Transform timestamps to RFC 3339 format for Port compatibility
                    transformed_finding = transform_finding_timestamps(finding)

                    # Create finding entity
                    finding_entity = {
                        "identifier": str(finding["id"]),
                        "title": finding.get("title", "Unknown Finding"),
                        "properties": {
                            "source": finding.get("source", "Unknown Tool"),
                            "description": finding.get("description", ""),
                            "mitigation": finding.get("mitigation", ""),
                            "severity": finding.get("severity", "UNKNOWN"),
                            "findingCategory": finding.get("findingCategory", ""),
                            "status": finding.get("status", "UNKNOWN"),
                            "productStatus": finding.get("productStatus", ""),
                            "subProductStatuses": finding.get("subProductStatuses", ""),
                            "title": finding.get("title", "Unknown Finding"),
                            "toolSeverity": finding.get("toolSeverity", ""),
                            "createdAt": transformed_finding.get("createdAt"),  # Use transformed timestamp
                            "lastUpdated": transformed_finding.get("lastUpdated"),  # Use transformed timestamp
                            "cwe": finding.get("cwe", []),
                            "cve": finding.get("cve", []),
                            "link": finding.get("findingUrl"),
                            "riskScore": finding.get("riskScore"),
                            "findingScore": finding.get("findingScore"),
                        },
                        "relations": {
                            "product": str(finding["product"]["id"]),
                            "subProduct": str(finding["subProduct"]["id"]),
                        },
                    }
                    finding_entities.append(finding_entity)
                    self.logger.info(f"Processed {idx} findings so far")
                    # Log progress every 1000 findings
                    if idx % 1000 == 0:
                        self.logger.info(f"Processed {idx} findings so far")

                except Exception as e:
                    self.logger.error(f"Failed to process finding {finding.get('id')}: {e}")

            # Bulk upsert entities
            if not context.config.dry_run:
                # Upsert findings
                if finding_entities:
                    finding_result = await context.bulk_port_manager.create_entities_bulk(
                        "armorcodeFinding", finding_entities, context.config.batch_size
                    )
                    result.entities_successful = finding_result.success_count
                    result.entities_failed = finding_result.failure_count
                    self.logger.info(f"Upserted {finding_result.success_count} findings")
            else:
                self.logger.info(f"[DRY RUN] Would upsert {len(finding_entities)} findings")
                result.entities_successful = len(finding_entities)
                result.entities_failed = 0

            result.success = True
            result.message = f"Processed {result.entities_processed} findings"
            result.metadata = {
                "findings_created": len(finding_entities)
            }

        except Exception as e:
            result.success = False
            result.message = f"Finding step failed: {str(e)}"
            result.error = e
            raise
        finally:
            result.execution_time = time.time() - start_time
    
    async def _setup_blueprint(self, context: ExecutionContext, blueprint_filename: str):
        """Setup a Port blueprint from JSON file."""
        import json
        import aiohttp
        
        blueprint_file = context.blueprints_path / blueprint_filename
        
        if not blueprint_file.exists():
            raise FileNotFoundError(f"Blueprint file not found: {blueprint_file}")
        
        with open(blueprint_file, "r") as f:
            blueprint_data = json.load(f)
        
        identifier = blueprint_data["identifier"]
        
        if context.config.dry_run:
            self.logger.info(f"[DRY RUN] Would setup blueprint '{identifier}'")
            return
        
        self.logger.info(f"Setting up blueprint '{identifier}'")
        
        headers = {"Authorization": f"Bearer {context.port_token}"}
        url = f"{context.port_api_url}/blueprints"
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=blueprint_data, headers=headers) as response:
                if response.status == 201:
                    self.logger.info(f"Blueprint '{identifier}' created successfully")
                elif response.status == 409:
                    # Blueprint exists, try to update
                    self.logger.info(f"Blueprint '{identifier}' already exists, updating...")
                    update_url = f"{url}/{identifier}"
                    async with session.put(update_url, json=blueprint_data, headers=headers) as update_response:
                        if update_response.status == 200:
                            self.logger.info(f"Blueprint '{identifier}' updated successfully")
                        else:
                            error_text = await update_response.text()
                            raise Exception(f"Failed to update blueprint: {update_response.status} - {error_text}")
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to create blueprint: {response.status} - {error_text}")