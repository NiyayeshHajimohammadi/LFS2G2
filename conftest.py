"""Ensures the repo root is importable during tests without an editable install."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
