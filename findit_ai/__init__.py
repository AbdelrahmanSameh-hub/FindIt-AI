"""
FindIt AI Package
An intelligent multimodal retrieval system for lost & found items.
"""

from .algorithms import (
    BM25,
    ColorHistogramMatcher,
    PerceptualHasher,
    ReciprocalRankFusion
)
from .dataset import FOUND_DATABASE, CAMPUS_DISTANCES, EVALUATION_QUERIES
from .engine import FindItEngine

__version__ = "1.0.0"
__all__ = [
    "BM25",
    "ColorHistogramMatcher",
    "PerceptualHasher",
    "ReciprocalRankFusion",
    "FindItEngine",
    "FOUND_DATABASE",
    "CAMPUS_DISTANCES",
    "EVALUATION_QUERIES"
]
