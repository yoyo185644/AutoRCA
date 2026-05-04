"""BASIL memory/indexing layer for RCA candidate retrieval.

The package initializer intentionally avoids importing query modules. Several
legacy solvers import submodules such as ``rca.memory.telemetry_cache`` while
``rca.component_candidates`` is still initializing; eager imports here would
create a circular dependency.
"""

__all__ = []
