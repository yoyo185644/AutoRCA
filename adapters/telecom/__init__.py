"""
Telecom Dataset Adapter

Wraps existing Telecom RCA logic into the DatasetAdapter interface.
This adapter demonstrates how to integrate legacy code without modification.
"""

from rca.adapters.telecom.adapter import TelecomAdapter

__all__ = ["TelecomAdapter"]
