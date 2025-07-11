"""
Configuration principale de l'application
Auteur: Dady Akrou Cyrille
Email: cyrilledady0501@gmail.com
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict


class Settings(BaseSettings):
    """Configuration de l'application avec validation Pydantic v2"""
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )
    
    # Informations du projet
    PROJECT_NAME: str = "Générateur de Texte IA - GPT & LLaMA"
    PROJECT_DESCRIPTION: str = "Application de génération de texte utilisant GPT-3.5-turbo et LLaMA"
    PROJECT_AUTHOR: str = "Dady Akrou Cyrille"
    PROJECT_EMAIL: str = "cyrilledady0501@gmail.com"
    VERSION: str = "1.0.0"
    
    # APIs de modèles
    OPENAI_API_KEY: str = Field(..., description="Clé API OpenAI")
    OPENAI_MODEL: str = "gpt-3.5-turbo"
    OPENAI_MAX_TOKENS: int = 2048
    OPENAI_TEMPERATURE: float = 0.7
    
    HF_API_TOKEN: Optional[str] = Field(None, description="Token Hugging Face")
    HF_MODEL_NAME: str = "meta-llama/Llama-3.1-8B"
    HF_CACHE_DIR: str = "./models/cache"
    
    # Configuration serveur
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_RELOAD: bool = True
    API_WORKERS: int = 4
    
    STREAMLIT_HOST: str = "0.0.0.0"
    STREAMLIT_PORT: int = 8502
    
    # Monitoring et logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    LOG_FILE: str = "logs/app.log"
    
    # Chemins
    DATA_DIR: str = "./data"
    MODELS_DIR: str = "./models"
    RESULTS_DIR: str = "./results"
    
    # Dataset TWCS
    TWCS_DATASET_PATH: str = "./dataset/twcs/twcs.csv"
    
    # Environnement
    ENVIRONMENT: str = "development"
    DEBUG: bool = True


# Instance globale des paramètres
settings = Settings()


def get_settings() -> Settings:
    """Retourne l'instance des paramètres de configuration"""
    return settings


def create_directories():
    """Crée les répertoires nécessaires s'ils n'existent pas"""
    directories = [
        settings.DATA_DIR,
        settings.MODELS_DIR,
        settings.RESULTS_DIR,
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Répertoire créé/vérifié: {directory}")


if __name__ == "__main__":
    # Test de la configuration
    print("🔧 Configuration du projet GPT-LLaMA")
    print(f"📝 Auteur: {settings.PROJECT_AUTHOR}")
    print(f"📧 Email: {settings.PROJECT_EMAIL}")
    print(f"🚀 Version: {settings.VERSION}")
    print(f"🌍 Environnement: {settings.ENVIRONMENT}")
    
    # Création des répertoires
    create_directories()
    
    print("✅ Configuration initialisée avec succès!")