"""
Package principal du projet GPT-LLaMA Text Generation Comparison
Auteur: Dady Akrou Cyrille
Email: cyrilledady0501@gmail.com
"""

from .config import settings, get_settings

__version__ = "1.0.0"
__author__ = "Dady Akrou Cyrille"
__email__ = "cyrilledady0501@gmail.com"

# Exports principaux
__all__ = [
    "settings",
    "get_settings"
]