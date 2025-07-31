"""
Armorcode to Port Integration Script

This script provides a one-way synchronization of data from Armorcode to Port.
It performs the following actions:
1.  Connects to the Port and Armorcode APIs using credentials stored in a .env file.
2.  Creates or updates the necessary blueprints in Port for Armorcode data, including:
    - armorcodeProduct
    - armorcodeSubProduct
    - armorcodeFinding
3.  Ingests product, sub-product, and active finding data from Armorcode.
4.  Transforms the fetched data into Port entity format, establishing relations
    between them to create a connected software catalog.
5.  Upserts the entities into Port, ensuring the catalog is kept up-to-date
    on subsequent runs.


Configure the .env file with your Port and Armorcode API credentials.

```env
# Port API Credentials
PORT_CLIENT_ID="your-port-client-id"
PORT_CLIENT_SECRET="your-port-client-secret"

# Armorcode API Key
ARMORCODE_API_KEY="your-armorcode-api-key"
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the script:

```bash
# Using the installed package (recommended)
armorcode-integration

# Or using module execution
python -m armorcode_integration
```

"""
import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from acsdk import ArmorCodeClient
from ..clients.armorcode_client import DirectArmorCodeClient
from dotenv import load_dotenv

# --- Setup ---
from ..managers.logging_manager import get_logging_manager, OperationType
from ..managers.progress_tracker import create_progress_tracker

# Initialize global logging manager
logging_manager = get_logging_manager()
logger = logging_manager.get_logger("main")

# Disable root logger to prevent duplicate logs
logging.getLogger().handlers.clear()
logging.getLogger().setLevel(logging.WARNING)  # Only show warnings/errors from other libraries

# --- Constants ---
PORT_API_URL = "https://api.us.getport.io/v1"

# --- Port Client ---

async def get_port_api_token() -> str:
    """Gets a Port API access token."""
    import aiohttp
    
    client_id = os.getenv("PORT_CLIENT_ID")
    client_secret = os.getenv("PORT_CLIENT_SECRET")

    if not client_id or not client_secret:
        raise ValueError("PORT_CLIENT_ID and PORT_CLIENT_SECRET must be set.")

    credentials = {"clientId": client_id, "clientSecret": client_secret}
    
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{PORT_API_URL}/auth/access_token", json=credentials) as response:
            response.raise_for_status()
            result = await response.json()
            return result["accessToken"]

def upsert_blueprint(blueprint: Dict[str, Any], token: str, dry_run: bool = False):
    """Creates or updates a blueprint in Port."""
    identifier = blueprint["identifier"]
    if dry_run:
        logging.info(f"[DRY RUN] Would upsert blueprint '{identifier}'.")
        return

    logging.info(f"POSTing blueprint to {PORT_API_URL}/blueprints with identifier {identifier}")
    logging.debug(f"Payload: {json.dumps(blueprint, indent=2)}")
    headers = {"Authorization": f"Bearer {port_token}"}
    response = requests.post(f"{PORT_API_URL}/blueprints", json=blueprint, headers=headers)
    if response.status_code != 201:
        logging.error(f"Failed to create blueprint: {response.status_code} - {response.text}")
    
    if response.status_code == 409: # Conflict, blueprint already exists
        logging.info(f"Blueprint '{identifier}' already exists. Attempting update.")
        # NOTE: Port API for blueprint update is PUT, not PATCH.
        # This means the entire blueprint definition is replaced.
        update_response = requests.put(f"{PORT_API_URL}/blueprints/{identifier}", json=blueprint, headers=headers)
        update_response.raise_for_status()
        logging.info(f"Blueprint '{identifier}' updated successfully.")
    else:
        response.raise_for_status()
        logging.info(f"Blueprint '{identifier}' created successfully.")

def upsert_entity(blueprint_id: str, entity: Dict[str, Any], token: str, dry_run: bool = False):
    """Creates or updates an entity in Port."""
    identifier = entity["identifier"]
    if dry_run:
        logging.info(f"[DRY RUN] Would upsert entity '{identifier}' into blueprint '{blueprint_id}'.")
        return
        
    headers = {"Authorization": f"Bearer {token}"}
    params = {'upsert': 'true', 'merge': 'true'}
    
    response = requests.post(
        f"{PORT_API_URL}/blueprints/{blueprint_id}/entities", 
        json=entity, 
        headers=headers, 
        params=params
    )
    response.raise_for_status()
    logging.info(f"Upserted entity '{identifier}' in blueprint '{blueprint_id}'.")


# --- Ingestion Logic ---

async def ingest_products(ac_client: ArmorCodeClient, port_token: str, dry_run: bool = False, 
                        bulk_manager: Optional['BulkPortManager'] = None, 
                        retry_manager: Optional['RetryManager'] = None,
                        batch_size: int = 20):
    """
    Ingests Armorcode Products into Port using bulk operations and retry logic.
    
    Args:
        ac_client: ArmorCode client for API calls
        port_token: Port API authentication token
        dry_run: Whether to run in dry-run mode
        bulk_manager: BulkPortManager instance for batch operations
        retry_manager: RetryManager instance for API retry logic
        batch_size: Batch size for bulk operations (default: 20)
    """
    from ..managers.bulk_port_manager import BulkPortManager
    from ..managers.retry_manager import RetryManager
    
    # Get component logger with correlation ID
    component_logger = logging_manager.get_logger("product_ingestion")
    
    component_logger.info(
        "Starting enhanced product ingestion with bulk operations",
        operation_type=OperationType.PRODUCT_INGESTION,
        batch_size=batch_size,
        dry_run=dry_run
    )
    
    # Initialize managers if not provided
    if bulk_manager is None:
        bulk_manager = BulkPortManager(port_token, retry_manager=retry_manager)
    if retry_manager is None:
        retry_manager = RetryManager(max_attempts=3)
    
    # Fetch products with retry logic and error handling
    try:
        products = await retry_manager.with_retry(ac_client.get_all_products)
        component_logger.info(
            f"Successfully fetched {len(products)} products from ArmorCode",
            operation_type=OperationType.PRODUCT_INGESTION,
            products_count=len(products)
        )
    except Exception as e:
        troubleshooting_info = logging_manager.log_troubleshooting_info(
            "product_ingestion", e, {"operation": "fetch_products"}
        )
        component_logger.error(
            "Failed to fetch products from ArmorCode after retries",
            error=e,
            operation_type=OperationType.PRODUCT_INGESTION,
            troubleshooting_info=troubleshooting_info
        )
        return
    
    if not products:
        component_logger.info(
            "No products found, skipping product ingestion",
            operation_type=OperationType.PRODUCT_INGESTION
        )
        return
    
    if dry_run:
        component_logger.info(
            f"[DRY RUN] Would process {len(products)} products in batches of {batch_size}",
            operation_type=OperationType.PRODUCT_INGESTION,
            total_products=len(products),
            batch_size=batch_size
        )
        return
    
    # Start progress tracking
    progress_tracker = create_progress_tracker(
        OperationType.PRODUCT_INGESTION, 
        len(products), 
        logging_manager, 
        "product_ingestion"
    )
    
    # Start periodic progress reporting
    await progress_tracker.start_periodic_reporting()
    
    try:
        # Transform products to entities
        entities = []
        failed_transformations = 0
        
        for i, product in enumerate(products):
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
                
                # Update progress for successful transformation
                progress_tracker.update(processed_delta=1, successful_delta=1)
                
            except Exception as e:
                failed_transformations += 1
                progress_tracker.update(processed_delta=1, failed_delta=1)
                
                troubleshooting_info = logging_manager.log_troubleshooting_info(
                    "product_ingestion", e, {"product_id": product.get('id', 'unknown')}
                )
                component_logger.error(
                    f"Failed to transform product {product.get('id', 'unknown')}",
                    error=e,
                    operation_type=OperationType.PRODUCT_INGESTION,
                    troubleshooting_info=troubleshooting_info,
                    product_id=product.get('id', 'unknown')
                )
                # Continue processing other products instead of failing completely
        
        if failed_transformations > 0:
            component_logger.warning(
                f"Failed to transform {failed_transformations} products, continuing with {len(entities)} valid entities",
                operation_type=OperationType.PRODUCT_INGESTION,
                failed_transformations=failed_transformations,
                valid_entities=len(entities)
            )
        
        if not entities:
            component_logger.warning(
                "No valid product entities to process after transformation",
                operation_type=OperationType.PRODUCT_INGESTION
            )
            return
    
        # Use bulk operations to create entities
        component_logger.info(
            f"Creating {len(entities)} product entities using bulk operations",
            operation_type=OperationType.PRODUCT_INGESTION,
            entity_count=len(entities),
            batch_size=batch_size
        )
        
        try:
            async with bulk_manager:
                # Submit entities in bulk with retry logic
                bulk_result = await retry_manager.with_retry(
                    bulk_manager.create_entities_bulk,
                    "armorcodeProduct",
                    entities,
                    batch_size
                )
                
                # Log detailed completion statistics
                component_logger.info(
                    "Product ingestion completed successfully",
                    operation_type=OperationType.PRODUCT_INGESTION,
                    total_processed=bulk_result.total_processed,
                    successful=bulk_result.success_count,
                    failed=bulk_result.failure_count,
                    success_rate=bulk_result.success_rate,
                    transformation_failures=failed_transformations
                )
                
                # Log failed entities for troubleshooting
                if bulk_result.failed_entities:
                    component_logger.warning(
                        f"Failed to create {len(bulk_result.failed_entities)} product entities",
                        operation_type=OperationType.PRODUCT_INGESTION,
                        failed_count=len(bulk_result.failed_entities)
                    )
                    
                    # Log first few failed entities with details
                    for entity_id, error_msg in bulk_result.failed_entities[:5]:
                        troubleshooting_info = logging_manager.log_troubleshooting_info(
                            "product_ingestion", Exception(error_msg), {"entity_id": entity_id}
                        )
                        component_logger.error(
                            f"Failed to create product entity {entity_id}",
                            operation_type=OperationType.PRODUCT_INGESTION,
                            entity_id=entity_id,
                            error_message=error_msg,
                            troubleshooting_info=troubleshooting_info
                        )
                    
                    if len(bulk_result.failed_entities) > 5:
                        component_logger.warning(
                            f"... and {len(bulk_result.failed_entities) - 5} more failed entities (check logs for details)",
                            operation_type=OperationType.PRODUCT_INGESTION
                        )
                        
        except Exception as e:
            troubleshooting_info = logging_manager.log_troubleshooting_info(
                "product_ingestion", e, {"operation": "bulk_create_entities", "entity_count": len(entities)}
            )
            component_logger.error(
                "Bulk product ingestion failed after retries",
                error=e,
                operation_type=OperationType.PRODUCT_INGESTION,
                troubleshooting_info=troubleshooting_info,
                entity_count=len(entities)
            )
            # Don't re-raise to allow other ingestion steps to continue
        
    finally:
        # Stop progress reporting and get final summary
        await progress_tracker.stop_periodic_reporting()
        final_summary = progress_tracker.get_progress_summary()
        
        component_logger.info(
            "Product ingestion operation completed",
            operation_type=OperationType.PRODUCT_INGESTION,
            **final_summary
        )

async def ingest_subproducts(ac_client: ArmorCodeClient, port_token: str, dry_run: bool = False,
                          bulk_manager: Optional['BulkPortManager'] = None, 
                          retry_manager: Optional['RetryManager'] = None,
                          batch_size: int = 20):
    """
    Ingests Armorcode Sub-Products into Port using bulk operations and retry logic.
    
    Args:
        ac_client: ArmorCode client for API calls
        port_token: Port API authentication token
        dry_run: Whether to run in dry-run mode
        bulk_manager: BulkPortManager instance for batch operations
        retry_manager: RetryManager instance for API retry logic
        batch_size: Batch size for bulk operations (default: 20)
    """
    from ..managers.bulk_port_manager import BulkPortManager
    from ..managers.retry_manager import RetryManager
    
    logging.info("Starting enhanced sub-product ingestion with bulk operations...")
    
    # Initialize managers if not provided
    if bulk_manager is None:
        bulk_manager = BulkPortManager(port_token, retry_manager=retry_manager)
    if retry_manager is None:
        retry_manager = RetryManager(max_attempts=3)
    
    # Fetch subproducts with retry logic
    try:
        subproducts = await retry_manager.with_retry(ac_client.get_all_subproducts)
        logging.info(f"Found {len(subproducts)} sub-products in Armorcode.")
    except Exception as e:
        logging.error(f"Failed to fetch sub-products from ArmorCode after retries: {e}")
        return
    
    if not subproducts:
        logging.info("No sub-products found, skipping sub-product ingestion.")
        return
    
    if dry_run:
        logging.info(f"[DRY RUN] Would process {len(subproducts)} sub-products in batches of {batch_size}")
        return
    
    # Transform subproducts to entities
    entities = []
    failed_transformations = 0
    
    for subproduct in subproducts:
        try:
            # The 'technologies' field is a string, but the blueprint expects an array.
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
            failed_transformations += 1
            logging.error(f"Failed to transform sub-product {subproduct.get('id', 'unknown')}: {e}")
            # Continue processing other sub-products instead of failing completely
    
    if failed_transformations > 0:
        logging.warning(f"Failed to transform {failed_transformations} sub-products, "
                       f"continuing with {len(entities)} valid entities")
    
    if not entities:
        logging.warning("No valid sub-product entities to process after transformation")
        return
    
    # Use bulk operations to create entities
    logging.info(f"Creating {len(entities)} sub-product entities using bulk operations...")
    
    try:
        async with bulk_manager:
            # Submit entities in bulk with retry logic
            bulk_result = await retry_manager.with_retry(
                bulk_manager.create_entities_bulk,
                "armorcodeSubProduct",
                entities,
                batch_size
            )
            
            # Log detailed statistics
            logging.info(f"Sub-product ingestion completed:")
            logging.info(f"  - Total processed: {bulk_result.total_processed}")
            logging.info(f"  - Successful: {bulk_result.success_count}")
            logging.info(f"  - Failed: {bulk_result.failure_count}")
            logging.info(f"  - Success rate: {bulk_result.success_rate:.1f}%")
            
            # Log failed entities for troubleshooting
            if bulk_result.failed_entities:
                logging.warning("Failed sub-product entities:")
                for entity_id, error_msg in bulk_result.failed_entities[:10]:  # Limit to first 10
                    logging.warning(f"  - {entity_id}: {error_msg}")
                if len(bulk_result.failed_entities) > 10:
                    logging.warning(f"  ... and {len(bulk_result.failed_entities) - 10} more")
                    
    except Exception as e:
        logging.error(f"Bulk sub-product ingestion failed after retries: {e}")
        # Don't re-raise to allow other ingestion steps to continue
    
    logging.info("Sub-product ingestion finished.")

async def ingest_findings(ac_client: ArmorCodeClient, port_token: str, dry_run: bool = False,
                                            bulk_manager: Optional['BulkPortManager'] = None, 
                                            retry_manager: Optional['RetryManager'] = None,
                                            batch_size: int = 20,
                                            max_concurrent: int = 10,
                                            filters: Optional[Dict[str, Any]] = None,
                                            after_key: Optional[int] = None,
                                            finding_filters_path: Optional[str] = None):
    """
    Ingests ArmorCode Findings into Port using parallel processing.
    
    Args:
        ac_client: ArmorCode client for API calls
        port_token: Port API authentication token
        dry_run: Whether to run in dry-run mode
        bulk_manager: BulkPortManager instance for batch operations
        retry_manager: RetryManager instance for API retry logic
        batch_size: Batch size for bulk operations (default: 20)
        max_concurrent: Maximum concurrent findings processing (default: 10)
        filters: Optional filters for findings query (deprecated, use finding_filters_path)
        after_key: Optional afterKey parameter for pagination
        finding_filters_path: Path to JSON file containing FindingFiltersRequestDto format filters
    """
    from ..managers.bulk_port_manager import BulkPortManager
    from ..managers.retry_manager import RetryManager
    from ..utils.batch_accumulator import BatchAccumulator
    from ..managers.filter_manager import FilterManager, FilterValidationError
    from ..utils.data_transformers import transform_finding_timestamps
    
    logging.info("Starting enhanced findings and vulnerabilities ingestion with parallel processing...")
    
    # Initialize managers if not provided
    if bulk_manager is None:
        bulk_manager = BulkPortManager(port_token, retry_manager=retry_manager)
    if retry_manager is None:
        retry_manager = RetryManager(max_attempts=3)
    
    # Prepare filters using FilterManager
    combined_filters = None
    try:
        if finding_filters_path or after_key is not None or filters:
            filter_manager = FilterManager()
            
            # Load and validate filters from file if provided
            file_filters = None
            if finding_filters_path:
                logging.info(f"Loading finding filters from: {finding_filters_path}")
                file_filters = filter_manager.load_and_validate_filters(finding_filters_path)
                logging.info(f"Successfully loaded and validated filters: {filter_manager.get_filter_summary(file_filters)}")
            
            # Use file filters or fallback to legacy filters parameter
            base_filters = file_filters if file_filters is not None else filters
            
            # Combine with afterKey parameter
            combined_filters = filter_manager.combine_filters(base_filters, after_key)
            
            if combined_filters:
                logging.info(f"Applied filters: {filter_manager.get_filter_summary(combined_filters)}")
            
    except FilterValidationError as e:
        logging.error(f"Filter validation failed: {e}")
        logging.error("Please check your filter file format and try again.")
        return
    except Exception as e:
        logging.error(f"Unexpected error while processing filters: {e}")
        return
    
    # Fetch findings with retry logic and filtering support
    try:
        logging.info("Fetching all findings from ArmorCode (this may take several minutes)...")
        
        # Use DirectArmorCodeClient specifically for findings since acsdk has issues with this endpoint
        api_key = os.getenv("ARMORCODE_API_KEY")
        async with DirectArmorCodeClient(api_key) as direct_client:
            findings = await retry_manager.with_retry(
                direct_client.get_all_findings,
                filters=combined_filters
            )
            
        logging.info(f"Found {len(findings)} active findings in ArmorCode.")
    except Exception as e:
        logging.error(f"Failed to fetch findings from ArmorCode after retries: {e}")
        return
    
    if not findings:
        logging.info("No findings found, skipping findings ingestion.")
        return
    
    if dry_run:
        logging.info(f"[DRY RUN] Would process {len(findings)} findings with {max_concurrent} concurrent workers")
        return
    
    # Initialize batch accumulator for findings
        finding_accumulator = bulk_manager.create_batch_accumulator("armorcodeFinding", batch_size)
        
        # Thread-safe set for tracking processed vulnerabilities
        processed_vulnerabilities = set()
        processed_vulnerabilities_lock = asyncio.Lock()
        
        # Statistics tracking
        processed_findings = 0
        failed_findings = 0
        processing_lock = asyncio.Lock()
        
        # Semaphore for concurrency control (requirement 7.1)
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_finding(finding: Dict[str, Any], finding_index: int) -> None:
            """Process a single finding concurrently."""
            nonlocal processed_findings, failed_findings
            
            async with semaphore:
                try:
                    # Transform timestamps to RFC 3339 format for Port compatibility
                    transformed_finding = transform_finding_timestamps(finding)
                    
                    # Create the Finding entity with correct field mappings
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
                    
                    # Add to finding batch accumulator
                    await finding_accumulator.add_entity(finding_entity)
                    
                    # Update success statistics
                    async with processing_lock:
                        processed_findings += 1
                        
                except Exception as e:
                    logging.error(f"Failed to process finding {finding.get('id', 'unknown')}: {e}")
                    async with processing_lock:
                        failed_findings += 1
        
        # Process findings concurrently (requirement 7.1)
        logging.info(f"Processing {len(findings)} findings with {max_concurrent} concurrent workers...")
        
        # Create tasks for parallel processing
        tasks = []
        for i, finding in enumerate(findings):
            task = asyncio.create_task(process_single_finding(finding, i))
            tasks.append(task)
        
        # Log progress periodically
        total_findings = len(findings)
        progress_interval = min(100, max(1, total_findings // 10))
        
        async def log_progress():
            """Log progress periodically while processing."""
            while True:
                await asyncio.sleep(5)  # Log every 5 seconds
                async with processing_lock:
                    current_processed = processed_findings + failed_findings
                    if current_processed > 0:
                        percentage = (current_processed / total_findings) * 100
                        logging.info(f"Progress: {current_processed}/{total_findings} findings processed "
                                   f"({percentage:.1f}%) - {processed_findings} successful, {failed_findings} failed")
                    
                    # Stop logging when all tasks are done
                    if current_processed >= total_findings:
                        break
        
        # Start progress logging task
        progress_task = asyncio.create_task(log_progress())
        
        try:
            # Wait for all findings to be processed (requirement 7.1)
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Cancel progress logging
            progress_task.cancel()
            try:
                await progress_task
            except asyncio.CancelledError:
                pass
            
            # Flush remaining batches (requirement 7.4)
            logging.info("Flushing remaining batches...")
            
            finding_final_result = await finding_accumulator.flush_remaining()
            
            # Get final statistics
            finding_stats = finding_accumulator.get_stats()
            
            # Log comprehensive completion statistics (requirement 8.1)
            logging.info("Findings and vulnerabilities ingestion completed:")
            logging.info(f"  Findings processed: {processed_findings} successful, {failed_findings} failed")
            logging.info(f"  Unique vulnerabilities: {len(processed_vulnerabilities)}")
            logging.info(f"  Vulnerability batches: {vuln_stats.batches_submitted} submitted, "
                        f"{vuln_stats.successful_entities} successful, {vuln_stats.failed_entities} failed")
            logging.info(f"  Finding batches: {finding_stats.batches_submitted} submitted, "
                        f"{finding_stats.successful_entities} successful, {finding_stats.failed_entities} failed")
            logging.info(f"  Total processing time: {vuln_stats.processing_time:.2f}s")
            logging.info(f"  Average batch sizes: vulnerabilities={vuln_stats.average_batch_size:.1f}, "
                        f"findings={finding_stats.average_batch_size:.1f}")
            
            # Log any failed entities for troubleshooting
            if vuln_final_result and vuln_final_result.failed_entities:
                for entity_id, error_msg in vuln_final_result.failed_entities[:5]:  # Limit to first 5
                    logging.warning(f"  - {entity_id}: {error_msg}")
                    
            if finding_final_result and finding_final_result.failed_entities:
                logging.warning(f"Failed finding entities: {len(finding_final_result.failed_entities)}")
                for entity_id, error_msg in finding_final_result.failed_entities[:5]:  # Limit to first 5
                    logging.warning(f"  - {entity_id}: {error_msg}")
            
        except Exception as e:
            logging.error(f"Error during parallel findings processing: {e}")
            # Cancel progress logging
            progress_task.cancel()
            try:
                await progress_task
            except asyncio.CancelledError:
                pass
            raise
    
    logging.info("Enhanced findings and vulnerabilities ingestion finished.")


# --- Enhanced Main Execution ---

async def setup_blueprints_with_retry(port_token: str, retry_manager: 'RetryManager', dry_run: bool = False):
    """
    Setup Port blueprints with retry logic for API calls.
    
    Args:
        port_token: Port API authentication token
        retry_manager: RetryManager instance for API retry logic
        dry_run: Whether to run in dry-run mode
    """
    from ..managers.retry_manager import RetryManager
    import aiohttp
    
    logger.info("Setting up Port blueprints with retry logic...")
    
    blueprints_path = Path(__file__).parent.parent / "blueprints"
    ordered_blueprints = [
        "product.json",
        "subproduct.json", 
        "finding.json"
    ]
    
    if dry_run:
        logger.info(f"[DRY RUN] Would setup {len(ordered_blueprints)} blueprints")
        return
    
    headers = {"Authorization": f"Bearer {port_token}"}
    
    async with aiohttp.ClientSession() as session:
        for bp_filename in ordered_blueprints:
            blueprint_file = blueprints_path / bp_filename
            
            if not blueprint_file.exists():
                logger.error(f"Blueprint file not found: {blueprint_file}")
                continue
            
            with open(blueprint_file, "r") as f:
                blueprint_data = json.load(f)
            
            identifier = blueprint_data["identifier"]
            logger.info(f"Setting up blueprint '{identifier}' with retry logic...")
            
            try:
                # Use retry manager for blueprint creation
                async def create_blueprint():
                    async with session.post(f"{PORT_API_URL}/blueprints", json=blueprint_data, headers=headers) as response:
                        if response.status == 201:
                            logger.info(f"Blueprint '{identifier}' created successfully")
                            return True
                        elif response.status == 409:
                            # Blueprint exists, try to update
                            logger.info(f"Blueprint '{identifier}' already exists, updating...")
                            async with session.put(f"{PORT_API_URL}/blueprints/{identifier}", json=blueprint_data, headers=headers) as update_response:
                                if update_response.status == 200:
                                    logger.info(f"Blueprint '{identifier}' updated successfully")
                                    return True
                                else:
                                    error_text = await update_response.text()
                                    raise Exception(f"Failed to update blueprint: {update_response.status} - {error_text}")
                        else:
                            error_text = await response.text()
                            raise Exception(f"Failed to create blueprint: {response.status} - {error_text}")
                
                await retry_manager.with_retry(create_blueprint)
                
            except Exception as e:
                logger.error(f"Failed to setup blueprint '{identifier}' after retries: {e}")
                # Continue with other blueprints instead of failing completely
                continue


async def main():
    """Enhanced main function using CLIController and StepExecutor."""
    import time
    from ..cli.cli_controller import CLIController
    from .step_executor import StepExecutor, ExecutionContext
    from ..managers.retry_manager import RetryManager
    
    start_time = time.time()
    
    # Load environment variables
    load_dotenv()
    
    # 1. Parse and validate CLI configuration
    logger.info("Parsing command line arguments...")
    cli_controller = CLIController()
    
    try:
        config = cli_controller.parse_arguments()
    except SystemExit:
        # argparse calls sys.exit on error or help
        return
    
    # Print configuration summary
    cli_controller.print_configuration_summary(config)
    
    if config.dry_run:
        logger.info("--- Running in DRY RUN mode ---")
    
    # 2. Get Port Token with retry logic
    logger.info("Authenticating with Port...")
    retry_manager = RetryManager(max_attempts=config.retry_attempts)
    
    try:
        port_token = await retry_manager.with_retry(get_port_api_token)
        logger.info("Successfully authenticated with Port")
    except Exception as e:
        logger.error(f"Failed to authenticate with Port after {config.retry_attempts} attempts: {e}")
        return
    
    # 3. Setup blueprints with retry logic
    await setup_blueprints_with_retry(port_token, retry_manager, dry_run=config.dry_run)
    
    # 4. Initialize execution context and step executor
    logger.info("Initializing execution context...")
    
    async with ExecutionContext(
        config=config,
        port_token=port_token
    ) as context:
        
        step_executor = StepExecutor()
        
        # 5. Execute selected steps
        logger.info(f"Starting execution of steps: {', '.join(config.steps)}")
        step_start_time = time.time()
        
        try:
            step_results = await step_executor.execute_steps(context, config.steps)
            step_execution_time = time.time() - step_start_time
            
            # 6. Generate execution summary with detailed statistics and timing information
            total_execution_time = time.time() - start_time
            
            logger.info("--- EXECUTION SUMMARY ---")
            
            # Overall statistics
            successful_steps = [name for name, result in step_results.items() if result.success]
            failed_steps = [name for name, result in step_results.items() if not result.success]
            
            logger.info(f"Total execution time: {total_execution_time:.2f} seconds")
            logger.info(f"Step execution time: {step_execution_time:.2f} seconds")
            logger.info(f"Steps requested: {len(config.steps)}")
            logger.info(f"Steps successful: {len(successful_steps)}")
            logger.info(f"Steps failed: {len(failed_steps)}")
            logger.info(f"Overall success rate: {(len(successful_steps) / len(config.steps) * 100):.1f}%")
            
            # Configuration summary (if needed)
            if config.dry_run or config.after_key is not None or config.finding_filters_path:
                config_details = []
                if config.dry_run:
                    config_details.append("dry run mode")
                if config.after_key is not None:
                    config_details.append(f"after key: {config.after_key}")
                if config.finding_filters_path:
                    config_details.append(f"filters: {config.finding_filters_path}")
                logger.info(f"Configuration: {', '.join(config_details)}")
            
            # Step results summary
            total_entities_processed = 0
            total_entities_successful = 0
            total_entities_failed = 0
            
            for step_name in config.steps:
                result = step_results.get(step_name)
                if result:
                    status_symbol = "✓" if result.success else "✗"
                    summary_info = f"{step_name}: {result.entities_processed} processed"
                    if result.entities_processed > 0:
                        summary_info += f" ({result.success_rate:.1f}% success)"
                    if result.execution_time > 0:
                        summary_info += f" in {result.execution_time:.1f}s"
                    
                    if result.success:
                        logger.info(f"  {status_symbol} {summary_info}")
                    else:
                        logger.warning(f"  {status_symbol} {summary_info} - {result.message}")
                    
                    # Accumulate totals
                    total_entities_processed += result.entities_processed
                    total_entities_successful += result.entities_successful
                    total_entities_failed += result.entities_failed
                else:
                    logger.warning(f"  ? {step_name}: No result available")
            
            # Entity processing summary
            if total_entities_processed > 0:
                entity_success_rate = (total_entities_successful / total_entities_processed) * 100
                entities_per_second = total_entities_processed / step_execution_time if step_execution_time > 0 else 0
                rate_info = f" at {entities_per_second:.1f} entities/sec" if step_execution_time > 0 else ""
                logger.info(f"Total: {total_entities_processed} entities processed ({entity_success_rate:.1f}% success){rate_info}")
            
            # Error summary
            if failed_steps:
                logger.warning("Failed steps:")
                for step_name in failed_steps:
                    result = step_results[step_name]
                    logger.error(f"  {step_name}: {result.message}")
                    if result.error:
                        logger.error(f"    {result.error}")
            
            logger.info("--- END SUMMARY ---")
            
            # Final status message
            if failed_steps:
                logger.warning(f"Integration completed with {len(failed_steps)} failed steps. Check logs for details.")
            else:
                logger.info("Integration completed successfully! All steps executed without errors.")
                
        except Exception as e:
            total_execution_time = time.time() - start_time
            logger.error(f"Integration failed after {total_execution_time:.2f} seconds: {e}")
            logger.error("Check the logs above for detailed error information.")
            raise


if __name__ == "__main__":
    # Use the enhanced main function with CLI controller
    asyncio.run(main())