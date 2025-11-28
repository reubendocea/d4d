#!/usr/bin/env python3
"""
D4D - The Dresden Dataset for 4D Reconstruction

Dataset loader for D4D - The Dresden Dataset for 4D Reconstruction of Non-Rigid Abdominal Surgical Scenes.
Tools for loading and processing surgical stereo data.
"""

__version__ = "0.1.0"
__author__ = "Reuben Docea"

# Import main classes and functions for easy access
from .loader import (
    D4D,
    Specimen,
    Session,
    Clip,
    CameraInfo,
    StereoParameters,
    load_camera_info,
    load_image,
)

__all__ = [
    # Main dataset classes
    "D4D",
    "Specimen",
    "Session",
    "Clip",
    # Data classes
    "CameraInfo",
    "StereoParameters",
    # Utility functions
    "load_camera_info",
    "load_image",
]
