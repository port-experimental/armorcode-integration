# ArmorCode Integration for Port.io

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A professional Python package for integrating ArmorCode security data with Port's developer portal, providing comprehensive visibility into your applications, repositories, and security findings.

## 🚀 Overview

This integration uses the [ArmorCode Python SDK](https://github.com/armor-code/acsdk) to extract data and the [Port API](https://docs.getport.io/api/) to build a rich and connected software catalog. It automatically imports and maintains the following entities in Port:

- **Products**: High-level applications or projects defined in ArmorCode
- **Sub-Products**: Repositories or components linked to a product
- **Findings**: Specific instances of a vulnerability detected on a sub-product

## 📦 Package Structure

```
armorcode-integration/
├── src/armorcode_integration/    # Main package
│   ├── core/                     # Core execution logic
│   │   ├── main.py              # Main entry point
│   │   └── step_executor.py     # Step-by-step execution
│   ├── managers/                 # Manager classes
│   │   ├── bulk_port_manager.py # Bulk API operations
│   │   ├── retry_manager.py     # Retry logic
│   │   ├── filter_manager.py    # Data filtering
│   │   ├── logging_manager.py   # Structured logging
│   │   └── progress_tracker.py  # Progress tracking
│   ├── clients/                  # External API clients
│   │   └── armorcode_client.py  # ArmorCode API client
│   ├── utils/                    # Utility functions
│   │   ├── batch_accumulator.py # Batch processing
│   │   ├── error_handler.py     # Error handling
│   │   └── streaming_processor.py # Stream processing
│   ├── cli/                      # Command-line interface
│   │   └── cli_controller.py    # CLI configuration
│   └── blueprints/              # Port blueprint definitions
│       ├── product.json
│       ├── subproduct.json
│       └── finding.json
├── tests/                        # Test suite
│   ├── unit/                    # Unit tests
│   └── integration/             # Integration tests
├── config/                       # Configuration files
├── pyproject.toml               # Modern Python packaging
├── setup.py                     # Backward compatibility
├── requirements.txt             # Core dependencies
├── requirements-dev.txt         # Development dependencies
└── Makefile                     # Development commands
```

## 📋 Prerequisites

- Python 3.8 or higher
- An active ArmorCode account with API access
- A Port account with API access credentials

## 🛠️ Installation

### Option 1: Install as a Package (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd armorcode-integration

# Install in development mode
pip install -e .

# Or install with development dependencies
pip install -r requirements-dev.txt
```

### Option 2: Traditional Installation

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## ⚙️ Configuration

Create a `.env` file in the project root:

```env
# Port API Credentials
PORT_CLIENT_ID="your-port-client-id"
PORT_CLIENT_SECRET="your-port-client-secret"

# ArmorCode API Key
ARMORCODE_API_KEY="your-armorcode-api-key"
```

## 🚀 Usage

### Using the Console Command (Recommended)

After installation, you can use the `armorcode-integration` command:

```bash
# Run full synchronization
armorcode-integration

# Run specific steps
armorcode-integration --steps product subproduct

# Show help
armorcode-integration --help
```

### Using Python Module

```bash
# Run as a module
python -m armorcode_integration

# With arguments
python -m armorcode_integration --dry-run --batch-size 10
```

### Advanced Usage

```bash
# Run with custom filters
armorcode-integration --finding-filters-path filters.json

# Run with retry configuration
armorcode-integration --retry-attempts 5 --batch-size 50

# Run specific steps only
armorcode-integration --steps finding --after-key 12345
```

## 🏗️ Data Model

The integration creates the following blueprints and relationships in Port:

| Blueprint                | Icon           | Description                                      | Relations                                                                   |
|--------------------------|----------------|--------------------------------------------------|-----------------------------------------------------------------------------|
| `armorcodeProduct`       | `Package`      | A top-level application in ArmorCode             | (None)                                                                      |
| `armorcodeSubProduct`    | `Git`          | A repository or component                        | `product` (→ `armorcodeProduct`)                                            |
| `armorcodeFinding`       | `Bug`          | A finding on a sub-product | `subProduct` (→ `armorcodeSubProduct`)|

This creates the following hierarchy:
**Product → Sub-Product → Finding ← Vulnerability**

## 🧪 Development

### Quick Start

```bash
# Install development dependencies
make install-dev

# Run all tests
make test

# Run specific test suites
make test-unit
make test-integration

# Format code
make format

# Run linting
make lint

# Show all available commands
make help
```

### Project Commands

```bash
# Development setup
make install-dev          # Install dev dependencies
make install             # Install package only

# Testing
make test                # Run all tests
make test-unit           # Unit tests only
make test-integration    # Integration tests only
make test-coverage       # Run with coverage report

# Code quality
make format              # Format with black & isort
make format-check        # Check formatting
make lint                # Run linting checks

# Build and distribution
make build               # Build package
make dist                # Create distribution
make clean               # Clean build artifacts

# Running
make run                 # Run the application
make run-help            # Show application help
```

### Testing

The project includes comprehensive unit and integration tests:

```bash
# Run unit tests
pytest tests/unit/ -v

# Run integration tests (requires credentials)
pytest tests/integration/ -v

# Run with coverage
pytest tests/ --cov=src/armorcode_integration --cov-report=html
```

## 📊 Features

### ✅ Core Features
- **Professional Package Structure**: Follows Python packaging best practices
- **Scalable Architecture**: Modular design with clear separation of concerns
- **Comprehensive Logging**: Structured logging with correlation IDs
- **Robust Error Handling**: Retry mechanisms and graceful failure handling
- **Bulk Operations**: Efficient batch processing for large datasets
- **Parallel Processing**: Concurrent handling of findings and vulnerabilities
- **Flexible Filtering**: Advanced filtering capabilities for data selection
- **Progress Tracking**: Real-time progress reporting for long-running operations

### ✅ Command Line Interface
- **Selective Execution**: Run specific integration steps
- **Dry Run Mode**: Preview operations without making changes
- **Configurable Batching**: Adjustable batch sizes for performance tuning
- **Retry Configuration**: Customizable retry attempts and strategies
- **Comprehensive Help**: Built-in documentation and examples

### ✅ Development Features
- **Type Hints**: Full type annotations for better IDE support
- **Code Quality Tools**: Black, isort, flake8, mypy integration
- **Comprehensive Testing**: Unit and integration test suites
- **Development Tools**: Makefile with common development tasks
- **Documentation**: Inline documentation and examples

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Install development dependencies (`make install-dev`)
4. Make your changes
5. Run tests (`make test`)
6. Format code (`make format`)
7. Run linting (`make lint`)
8. Commit your changes (`git commit -m 'Add amazing feature'`)
9. Push to the branch (`git push origin feature/amazing-feature`)
10. Open a Pull Request

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you've installed the package with `pip install -e .`
2. **Authentication Failures**: Verify your `.env` file contains valid credentials
3. **API Rate Limits**: Adjust batch sizes or retry configurations
4. **Memory Issues**: Reduce batch sizes for large datasets

### Debug Mode

Run with increased verbosity:

```bash
# Enable debug logging
PYTHONPATH=src python -m armorcode_integration --help

# Run specific steps with dry-run
armorcode-integration --dry-run --steps product
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [ArmorCode](https://armorcode.com/) for providing the security data platform
- [Port.io](https://getport.io/) for the developer portal platform
- The Python community for excellent packaging and development tools
