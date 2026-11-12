"""Data utilities for the LDCT (low-dose to full-dose CT) experiments.

Kept import-light on purpose: ``ldct.data.utils`` needs only the standard library, so the
index builder runs without torch. Import the dataset classes from
``ldct.data.cached_dataset`` explicitly.
"""
