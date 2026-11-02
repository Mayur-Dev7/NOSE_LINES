"""
Cattle Muzzle AI Detection API Package
======================================
Modular FastAPI backend application for cattle muzzle identification,
segmentation, groove extraction, reference template consensus, and
deep learning biometric embedding matching.
"""

from .app import app

__all__ = ["app"]
