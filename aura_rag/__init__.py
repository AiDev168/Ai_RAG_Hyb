"""AURA RAG - local document Q&A (modular layout).

Importing the package installs the network guard FIRST, before any
networking or model library is imported, so Offline mode is always in
force from the very start.
"""
import logging

from . import network  # noqa: F401  (must stay first: sets offline env flags + socket guard)

logging.basicConfig(level=logging.INFO)
