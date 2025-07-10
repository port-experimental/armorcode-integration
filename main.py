"""
Armorcode to Port Integration Script

This script provides a one-way synchronization of data from Armorcode to Port.
It performs the following actions:
1.  Connects to the Port and Armorcode APIs using credentials stored in a .env file.
2.  Creates or updates the necessary blueprints in Port for Armorcode data, including:
    - armorcodeProduct
    - armorcodeSubProduct
    - armorcodeVulnerability
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
python main.py
```

"""
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import requests
from acsdk import ArmorCodeClient
from dotenv import load_dotenv

# --- Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Constants ---
PORT_API_URL = "https://api.getport.io/v1"

# --- Port Client ---

def get_port_api_token() -> str:
    """Gets a Port API access token."""
    client_id = os.getenv("PORT_CLIENT_ID")
    client_secret = os.getenv("PORT_CLIENT_SECRET")

    if not client_id or not client_secret:
        raise ValueError("PORT_CLIENT_ID and PORT_CLIENT_SECRET must be set.")

    credentials = {"clientId": client_id, "clientSecret": client_secret}
    token_response = requests.post(f"{PORT_API_URL}/auth/access_token", json=credentials)
    token_response.raise_for_status()
    return token_response.json()["accessToken"]

def upsert_blueprint(blueprint: Dict[str, Any], token: str):
    """Creates or updates a blueprint in Port."""
    identifier = blueprint["identifier"]
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(f"{PORT_API_URL}/blueprints", json=blueprint, headers=headers)
    
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

def upsert_entity(blueprint_id: str, entity: Dict[str, Any], token: str):
    """Creates or updates an entity in Port."""
    identifier = entity["identifier"]
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

async def ingest_products(ac_client: ArmorCodeClient, port_token: str):
    """Ingests Armorcode Products into Port."""
    logging.info("Starting product ingestion...")
    products = await ac_client.get_all_products()
    logging.info(f"Found {len(products)} products in Armorcode.")

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
            upsert_entity("armorcodeProduct", entity, port_token)
        except Exception as e:
            logging.error(f"Failed to process product {product.get('id')}: {e}")

    logging.info("Product ingestion finished.")

async def ingest_subproducts(ac_client: ArmorCodeClient, port_token: str):
    """Ingests Armorcode Sub-Products into Port."""
    logging.info("Starting sub-product ingestion...")
    subproducts = await ac_client.get_all_subproducts()
    logging.info(f"Found {len(subproducts)} sub-products in Armorcode.")

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
            upsert_entity("armorcodeSubProduct", entity, port_token)
        except Exception as e:
            logging.error(f"Failed to process sub-product {subproduct.get('id')}: {e}")

    logging.info("Sub-product ingestion finished.")

async def ingest_findings_and_vulnerabilities(ac_client: ArmorCodeClient, port_token: str):
    """Ingests Armorcode Findings and their related Vulnerabilities into Port."""
    logging.info("Starting findings and vulnerabilities ingestion...")
    findings = await ac_client.get_all_findings(status="ACTIVE")
    logging.info(f"Found {len(findings)} active findings in Armorcode.")
    
    processed_vulnerabilities = set()

    for finding in findings:
        try:
            # 1. Upsert the unique Vulnerability entity
            vuln_data = finding.get("vulnerability")
            if vuln_data and vuln_data.get("id"):
                vuln_id = str(vuln_data["id"])
                if vuln_id not in processed_vulnerabilities:
                    vulnerability_entity = {
                        "identifier": vuln_id,
                        "title": vuln_data.get("vulnerabilityName") or vuln_data.get("name"),
                        "properties": {
                            "name": vuln_data.get("vulnerabilityName") or vuln_data.get("name"),
                            "cve": vuln_data.get("cve"),
                            "severity": vuln_data.get("severity"),
                            "description": vuln_data.get("description"),
                            "remediation": vuln_data.get("remediation"),
                        },
                    }
                    upsert_entity("armorcodeVulnerability", vulnerability_entity, port_token)
                    processed_vulnerabilities.add(vuln_id)

                # 2. Upsert the Finding entity
                finding_entity = {
                    "identifier": str(finding["id"]),
                    "title": finding.get("title"),
                    "properties": {
                        "title": finding.get("title"),
                        "status": finding.get("status"),
                        "severity": finding.get("severity"),
                        "age": finding.get("age"),
                        "tool": finding.get("tool", {}).get("name"),
                        "link": finding.get("ticketUrl"), # Assuming ticketUrl holds the link
                    },
                    "relations": {
                        "subProduct": str(finding["subProduct"]["id"]),
                        "vulnerability": vuln_id,
                    },
                }
                upsert_entity("armorcodeFinding", finding_entity, port_token)
        except Exception as e:
            logging.error(f"Failed to process finding {finding.get('id')}: {e}")
            
    logging.info("Findings and vulnerabilities ingestion finished.")

# --- Main Execution ---

async def main():
    """Main function to run the integration."""
    load_dotenv()
    
    # 1. Get Port Token
    logging.info("Authenticating with Port...")
    port_token = get_port_api_token()
    
    # 2. Setup Blueprints
    logging.info("Setting up Port blueprints...")
    blueprints_path = Path(__file__).parent / "blueprints"
    for blueprint_file in blueprints_path.glob("*.json"):
        with open(blueprint_file, "r") as f:
            blueprint_data = json.load(f)
            upsert_blueprint(blueprint_data, port_token)
    
    # 3. Initialize Armorcode Client
    logging.info("Authenticating with Armorcode...")
    api_key = os.getenv("ARMORCODE_API_KEY")
    if not api_key:
        raise ValueError("ARMORCODE_API_KEY must be set.")
    
    async with ArmorCodeClient(api_key) as ac_client:
        # 4. Run Ingestion
        await ingest_products(ac_client, port_token)
        await ingest_subproducts(ac_client, port_token)
        await ingest_findings_and_vulnerabilities(ac_client, port_token)

    logging.info("Integration run completed successfully!")

if __name__ == "__main__":
    asyncio.run(main()) 