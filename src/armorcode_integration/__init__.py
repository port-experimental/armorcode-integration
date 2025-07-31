"""
ArmorCode Integration Package

A professional Python package for integrating ArmorCode security data with Port.io.
"""

__version__ = "1.0.0"
__author__ = "Port"
__description__ = "ArmorCode to Port.io integration for security data synchronization"

# Make commonly used classes available at package level
from .core.main import main
from .managers.bulk_port_manager import BulkPortManager
from .managers.retry_manager import RetryManager
from .clients.armorcode_client import DirectArmorCodeClient

__all__ = [
    "main",
    "BulkPortManager", 
    "RetryManager",
    "DirectArmorCodeClient",
    "__version__",
] 