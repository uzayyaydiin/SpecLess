"""SpecLess: photometry-based machine learning for galaxy classification."""

from .morphology import MorphologyClassifier
from .activity import ActivityClassifier

__all__ = [
    "MorphologyClassifier",
    "ActivityClassifier",
]

__version__ = "0.2.1"
