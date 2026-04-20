"""Febridge: Free energy difference calculation using neural network flows."""

# Core components
from .core import make_transformer
from .flows import make_flow
from .losses import make_loss
from .energy import make_free_energy

# Training
from .training import train_and_evaluate

# High-level API
from .api import model_train, model_inference

# Exceptions
from .utils import (
    FebridgeError,
    SampleSizeError,
    SizeMismatchError,
)

__all__ = [
    # Core
    "make_transformer",
    "make_flow",
    "make_loss",
    "make_free_energy",
    # Training
    "train_and_evaluate",
    # High-level API
    "model_train",
    "model_inference",
    # Exceptions
    "FebridgeError",
    "SampleSizeError",
    "SizeMismatchError",
]
