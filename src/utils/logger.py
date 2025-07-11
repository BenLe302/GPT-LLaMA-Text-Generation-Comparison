"""
Module de logging pour l'application
Auteur: Dady Akrou Cyrille
Email: cyrilledady0501@gmail.com
"""

import sys
from loguru import logger
from typing import Optional
from pathlib import Path

try:
    from ..config import settings
except ImportError:
    # Fallback si l'import relatif échoue
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from config import settings


def setup_logger(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    rotation: str = "10 MB",
    retention: str = "30 days"
) -> None:
    """
    Configure le système de logging avec loguru
    
    Args:
        log_level: Niveau de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Chemin vers le fichier de log (optionnel)
        rotation: Taille de rotation des logs
        retention: Durée de rétention des logs
    """
    # Supprimer la configuration par défaut
    logger.remove()
    
    # Configuration pour la console
    logger.add(
        sys.stdout,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
               "<level>{message}</level>",
        colorize=True
    )
    
    # Configuration pour le fichier (si spécifié)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_file,
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation=rotation,
            retention=retention,
            compression="zip"
        )


def get_logger(name: str = None):
    """
    Retourne une instance du logger configuré
    
    Args:
        name: Nom du logger (généralement __name__)
        
    Returns:
        Instance du logger loguru
    """
    # Configuration initiale si pas encore fait
    if not logger._core.handlers:
        setup_logger(
            log_level=getattr(settings, 'LOG_LEVEL', 'INFO'),
            log_file=getattr(settings, 'LOG_FILE', None)
        )
    
    # Retourner le logger avec le nom spécifié
    if name:
        return logger.bind(name=name)
    return logger


# Configuration automatique au chargement du module
try:
    setup_logger(
        log_level=getattr(settings, 'LOG_LEVEL', 'INFO'),
        log_file=getattr(settings, 'LOG_FILE', None)
    )
except Exception as e:
    # Fallback en cas d'erreur
    logger.add(sys.stdout, level="INFO")
    logger.warning(f"Configuration de logging par défaut utilisée: {e}")