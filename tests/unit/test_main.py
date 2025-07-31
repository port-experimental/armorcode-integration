import pytest
from unittest.mock import patch, call, AsyncMock

# Because the functions we are testing are async, we need to use pytest-asyncio
from main import (
    ingest_products,
    ingest_subproducts,
    ingest_findings,
)


# --- Mock Data ---
MOCK_PRODUCTS = [
    {"id": "prod_1", "name": "E-Commerce Website", "description": "Main customer-facing website.", "businessOwnerName": "Alice", "securityOwnerName": "Bob"},
]
MOCK_SUBPRODUCTS = [
    {"id": "sub_101", "name": "payment-service", "repoLink": "https://github.com/example/payment-service", "programmingLanguage": "Java", "technologies": "Spring,Maven", "parent": "prod_1"},
]
MOCK_FINDINGS = [
    {
        "id": 1001,
        "title": "Log4j Vulnerability in payment-service",
        "source": "Snyk",
        "description": "Remote code execution vulnerability in Log4j library",
        "mitigation": "Upgrade to version 2.17.1 or later",
        "severity": "CRITICAL",
        "findingCategory": "Vulnerability",
        "status": "ACTIVE",
        "productStatus": "Active",
        "subProductStatuses": "Active",
        "toolSeverity": "CRITICAL",
        "createdAt": 1732564920000,  # 2024-11-25T12:35:20.000Z  
        "lastUpdated": 1752603329000,  # 2025-05-15T09:01:02.000Z
        "cwe": ["CWE-502"],
        "cve": ["CVE-2021-44228"],
        "findingUrl": "https://app.armorcode.com#/findings/1001",
        "riskScore": 85.5,
        "findingScore": 90.0,
        "product": {"id": "prod_1", "name": "E-Commerce Platform"},
        "subProduct": {"id": "sub_101", "name": "payment-service"},
    },
]

# --- Mock ArmorCode Client ---
class MockArmorCodeClient:
    """A mock client that simulates responses from the Armorcode API."""
    async def get_all_products(self):
        return MOCK_PRODUCTS

    async def get_all_subproducts(self):
        return MOCK_SUBPRODUCTS
    
    async def get_all_findings(self, status=None):
        if status:
            return [f for f in MOCK_FINDINGS if f["status"] == status]
        return MOCK_FINDINGS

# --- Tests ---

@pytest.mark.parametrize("dry_run", [True, False])
@pytest.mark.asyncio
async def test_ingest_products(dry_run):
    """Tests the product ingestion logic in both normal and dry-run modes."""
    mock_ac_client = MockArmorCodeClient()
    mock_port_token = "fake-token"
    
    with patch('main.upsert_entity') as mock_upsert:
        await ingest_products(mock_ac_client, mock_port_token, dry_run=dry_run)

        expected_entity = {
            "identifier": "prod_1",
            "title": "E-Commerce Website",
            "properties": {
                "name": "E-Commerce Website",
                "description": "Main customer-facing website.",
                "businessOwner": "Alice",
                "securityOwner": "Bob",
            },
        }
        mock_upsert.assert_called_once_with("armorcodeProduct", expected_entity, mock_port_token, dry_run=dry_run)

@pytest.mark.parametrize("dry_run", [True, False])
@pytest.mark.asyncio
async def test_ingest_subproducts(dry_run):
    """Tests the sub-product ingestion logic in both normal and dry-run modes with bulk operations."""
    mock_ac_client = MockArmorCodeClient()
    mock_port_token = "fake-token"

    # Mock the bulk manager and retry manager
    with patch('bulk_port_manager.BulkPortManager') as mock_bulk_class, \
         patch('retry_manager.RetryManager') as mock_retry_class:
        
        # Setup mock instances
        from bulk_port_manager import BulkResult
        mock_bulk_instance = AsyncMock()
        mock_bulk_result = BulkResult(
            successful_entities=["sub_101"],
            failed_entities=[],
            total_processed=1
        )
        mock_bulk_instance.create_entities_bulk.return_value = mock_bulk_result
        mock_bulk_instance.__aenter__.return_value = mock_bulk_instance
        mock_bulk_instance.__aexit__.return_value = None
        mock_bulk_class.return_value = mock_bulk_instance
        
        mock_retry_instance = AsyncMock()
        async def mock_with_retry(func, *args, **kwargs):
            return await func(*args, **kwargs)
        mock_retry_instance.with_retry.side_effect = mock_with_retry
        mock_retry_class.return_value = mock_retry_instance

        await ingest_subproducts(mock_ac_client, mock_port_token, dry_run=dry_run)

        if not dry_run:
            # Verify bulk manager was used
            mock_bulk_class.assert_called_once_with(mock_port_token)
            mock_retry_class.assert_called_once_with(max_attempts=3)
            
            # Verify the expected entity was processed
            expected_entity = {
                "identifier": "sub_101",
                "title": "payment-service",
                "properties": {
                    "name": "payment-service",
                    "repoLink": "https://github.com/example/payment-service",
                    "programmingLanguage": "Java",
                    "technologies": ["Spring", "Maven"],
                },
                "relations": {"product": "prod_1"},
            }
            
            # Check that bulk operations were called with the expected entity
            call_args = mock_retry_instance.with_retry.call_args_list[-1]
            entities_arg = call_args[0][2]  # Third argument (entities)
            assert len(entities_arg) == 1
            assert entities_arg[0] == expected_entity
        else:
            # In dry run mode, bulk operations should not be called
            mock_bulk_instance.create_entities_bulk.assert_not_called()

@pytest.mark.parametrize("dry_run", [True, False])
@pytest.mark.asyncio
async def test_ingest_findings(dry_run):
    """Tests the findings and vulnerabilities ingestion logic in both normal and dry-run modes."""
    mock_ac_client = MockArmorCodeClient()
    mock_port_token = "fake-token"
    
    with patch('main.upsert_entity') as mock_upsert:
        await ingest_findings(mock_ac_client, mock_port_token, dry_run=dry_run)

        expected_vuln_entity = {
            'identifier': 'vuln_1',
            'title': 'CVE-2021-44228 - Log4Shell',
            'properties': {
                'name': 'CVE-2021-44228 - Log4Shell',
                'cve': 'CVE-2021-44228',
                'severity': 'CRITICAL',
                'description': 'Remote code execution in Log4j.',
                'remediation': 'Upgrade to version 2.17.1 or later.'
            }
        }

        expected_finding_entity = {
            'identifier': 'find_1001',
            'title': 'Log4j Vulnerability in payment-service',
            'properties': {
                'title': 'Log4j Vulnerability in payment-service',
                'status': 'ACTIVE',
                'severity': 'CRITICAL',
                'age': 15,
                'tool': 'Snyk',
                'link': 'https://jira.example.com/TICKET-123'
            },
            'relations': {
                'subProduct': 'sub_101',
                            }
        }

        assert mock_upsert.call_count == 2
        mock_upsert.assert_has_calls([
            call('armorcodeFinding', expected_finding_entity, mock_port_token, dry_run=dry_run)
        ], any_order=True) 