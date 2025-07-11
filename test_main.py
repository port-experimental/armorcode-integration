import pytest
from unittest.mock import patch, call

# Because the functions we are testing are async, we need to use pytest-asyncio
from main import (
    ingest_products,
    ingest_subproducts,
    ingest_findings_and_vulnerabilities,
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
        "id": "find_1001", "title": "Log4j Vulnerability in payment-service", "status": "ACTIVE", "severity": "CRITICAL", "age": 15, "tool": {"name": "Snyk"}, "ticketUrl": "https://jira.example.com/TICKET-123", "subProduct": {"id": "sub_101"},
        "vulnerability": { "id": "vuln_1", "vulnerabilityName": "CVE-2021-44228 - Log4Shell", "cve": "CVE-2021-44228", "severity": "CRITICAL", "description": "Remote code execution in Log4j.", "remediation": "Upgrade to version 2.17.1 or later." }
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
    """Tests the sub-product ingestion logic in both normal and dry-run modes."""
    mock_ac_client = MockArmorCodeClient()
    mock_port_token = "fake-token"

    with patch('main.upsert_entity') as mock_upsert:
        await ingest_subproducts(mock_ac_client, mock_port_token, dry_run=dry_run)

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
        mock_upsert.assert_called_once_with("armorcodeSubProduct", expected_entity, mock_port_token, dry_run=dry_run)

@pytest.mark.parametrize("dry_run", [True, False])
@pytest.mark.asyncio
async def test_ingest_findings_and_vulnerabilities(dry_run):
    """Tests the findings and vulnerabilities ingestion logic in both normal and dry-run modes."""
    mock_ac_client = MockArmorCodeClient()
    mock_port_token = "fake-token"
    
    with patch('main.upsert_entity') as mock_upsert:
        await ingest_findings_and_vulnerabilities(mock_ac_client, mock_port_token, dry_run=dry_run)

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
                'vulnerability': 'vuln_1'
            }
        }

        assert mock_upsert.call_count == 2
        mock_upsert.assert_has_calls([
            call('armorcodeVulnerability', expected_vuln_entity, mock_port_token, dry_run=dry_run),
            call('armorcodeFinding', expected_finding_entity, mock_port_token, dry_run=dry_run)
        ], any_order=True) 