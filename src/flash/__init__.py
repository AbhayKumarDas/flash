"""FLASH — a generate-once, synthesise-many framework for reference-free synthetic anomaly
generation in industrial anomaly detection.

The pipeline runs as three notebooks (``notebooks/``); this package holds the operators they
import, so the algorithm lives in one place rather than being duplicated per notebook.

    from flash import config
    from flash.mrsp import generate_mrsp
    from flash.placement import place_original, place_adaptive
"""

__version__ = "0.1.0"

from flash import config  # noqa: F401

__all__ = ["config", "__version__"]
