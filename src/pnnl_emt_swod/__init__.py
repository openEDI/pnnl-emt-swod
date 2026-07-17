"""
OEDISI pnnl-emt-swod — Sliding Window Oscillation Detection (SWOD)

This component provides:
- HELICS co-simulation wrapper for the SWOD algorithm
"""

__version__ = "0.1.0"

from .federate import ComponentParameters, Federate, run_simulator
from . import plotting

__all__ = [
    "__version__",
    "run_simulator",
    "Federate",
    "ComponentParameters",
    "plotting",
]
