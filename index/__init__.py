"""
Unified Index Interface

This module provides a unified interface for querying RCA results
across different datasets. All adapters must return results in the
UnifiedCandidate format.
"""

from rca.index.candidate import UnifiedCandidate
from rca.index.interface import PublicIndexInterface, SimpleIndexImpl

__all__ = [
    "UnifiedCandidate",
    "PublicIndexInterface",
    "SimpleIndexImpl",
]
