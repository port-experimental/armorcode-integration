# Armorcode Port Integration

A Python-based integration that synchronizes Armorcode data with Port's developer portal, providing comprehensive visibility into your applications, repositories, and security findings.

## 🚀 Overview

This integration uses the [Armorcode Python SDK](https://github.com/armor-code/acsdk) to extract data and the [Port API](https://docs.getport.io/api/) to build a rich and connected software catalog. It automatically imports and maintains the following entities in Port:

*   **Products**: High-level applications or projects defined in Armorcode.
*   **Sub-Products**: Repositories or components linked to a product.
*   **Vulnerabilities**: A catalog of unique security vulnerabilities (e.g., CVEs).
*   **Findings**: Specific instances of a vulnerability detected on a sub-product.

## 📋 Prerequisites

*   Python 3.8+
*   An active Armorcode account with API access.
*   A Port account with API access credentials.

## 🛠️ Installation & Setup

### 1. Clone the Repository

```bash
git clone <this-repository-url>
cd armorcode-integration
```

### 2. Install Dependencies

It is recommended to use a virtual environment.

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the root of the project by copying the `example.env` content.

```env
# Port API Credentials
PORT_CLIENT_ID="your-port-client-id"
PORT_CLIENT_SECRET="your-port-client-secret"

# Armorcode API Key
ARMORCODE_API_KEY="your-armorcode-api-key"
```

Fill in the values with your actual credentials from Port and Armorcode.

## 🚀 Usage

To run the integration and perform a full synchronization, execute the main script:

```bash
python main.py
```

### What the Script Does

1.  **Authenticates**: Establishes secure connections to both Port and Armorcode.
2.  **Manages Blueprints**: Creates or updates the four required blueprints (`armorcodeProduct`, `armorcodeSubProduct`, `armorcodeVulnerability`, `armorcodeFinding`) in your Port instance.
3.  **Ingests Data**: Fetches all products, sub-products, and active findings from Armorcode.
4.  **Builds Catalog**: Transforms and loads the data into Port as entities, automatically creating the relationships between them.

The script is idempotent, meaning you can run it multiple times, and it will only update what has changed, without creating duplicates.

## 🏗️ Data Model

The integration creates the following blueprints and relationships in Port:

| Blueprint                 | Icon            | Description                                        | Relations                                                                    |
| ------------------------- | --------------- | -------------------------------------------------- | ---------------------------------------------------------------------------- |
| `armorcodeProduct`        | `Package`       | A top-level application in Armorcode.              | (None)                                                                       |
| `armorcodeSubProduct`     | `Git`           | A repository or component.                         | `product` (→ `armorcodeProduct`)                                             |
| `armorcodeVulnerability`  | `Vulnerability` | A unique security vulnerability (e.g., CVE).       | (None)                                                                       |
| `armorcodeFinding`        | `Bug`           | An instance of a vulnerability on a sub-product.   | `subProduct` (→ `armorcodeSubProduct`) <br/> `vulnerability` (→ `armorcodeVulnerability`) |

This creates the following hierarchy:
**Product → Sub-Product → Finding ← Vulnerability**
