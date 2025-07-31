"""
Client classes for external API communication.

This module contains client implementations for:
- ArmorCode API integration
- Port.io API communication
"""

from .armorcode_client import DirectArmorCodeClient, ArmorCodeAPIError

__all__ = ["DirectArmorCodeClient", "ArmorCodeAPIError"] 