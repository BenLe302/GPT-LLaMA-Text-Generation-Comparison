"""
Configuration principale de l'application
Auteur: Dady Akrou Cyrille
Email: cyrilledady0501@gmail.com
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Configuration de l'application avec validation Pydantic"""
    
    # Informations du projet
    PROJECT_NAME: str = "Générateur de Texte IA - GPT & LLaMA"
    PROJECT_DESCRIPTION: str = "Application de génération de texte utilisant GPT-4 et LLaMA"
    PROJECT_AUTHOR: str = "Dady Akrou Cyrille"
    PROJECT_EMAIL: str = "cyrilledady0501@gmail.com"
    VERSION: str = "1.0.0"
    
    # APIs de modèles
    OPENAI_API_KEY: str = Field(..., description="Clé API OpenAI")
    OPENAI_MODEL: str = "gpt-4-turbo-preview"
    OPENAI_MAX_TOKENS: int = 2048
    OPENAI_TEMPERATURE: float = 0.7
    
    HF_API_TOKEN: Optional[str] = Field(None, description="Token Hugging Face")
    HF_MODEL_NAME: str = "meta-llama/Llama-2-7b-chat-hf"
    HF_CACHE_DIR: str = "./models/cache"
    
    # Base de données
    DATABASE_URL: str = "postgresql://postgres:password@localhost:5432/gpt_llama_db"
    
    # Configuration API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_RELOAD: bool = True
    API_WORKERS: int = 1
    
    # Sécurité
    SECRET_KEY: str = Field(default="your-secret-key-here", description="Clé secrète pour JWT")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    ALGORITHM: str = "HS256"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/app.log"
    LOG_ROTATION: str = "10 MB"
    LOG_RETENTION: str = "30 days"
    
    # Chemins des données
    DATA_DIR: str = "data"
    RAW_DATA_DIR: str = "data/raw"
    PROCESSED_DATA_DIR: str = "data/processed"
    MODELS_DIR: str = "models"
    RESULTS_DIR: str = "results"
    EXPORTS_DIR: str = "exports"
    
    # Dataset TWCS
    TWCS_DATASET_PATH: str = "data/raw/twcs.csv"
    TWCS_PROCESSED_PATH: str = "data/processed/twcs_processed.csv"
    
    # Configuration des modèles
    MAX_SEQUENCE_LENGTH: int = 512
    BATCH_SIZE: int = 16
    LEARNING_RATE: float = 2e-5
    NUM_EPOCHS: int = 3
    
    # Limites de l'API
    RATE_LIMIT_PER_MINUTE: int = 60
    MAX_CONCURRENT_REQUESTS: int = 10
    REQUEST_TIMEOUT: int = 300
    
    # Configuration Gradio
    GRADIO_HOST: str = "0.0.0.0"
    GRADIO_PORT: int = 7860
    GRADIO_SHARE: bool = False
    
    # Configuration Streamlit
    STREAMLIT_HOST: str = "0.0.0.0"
    STREAMLIT_PORT: int = 8501
    
    # Métriques et monitoring
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    
    # Cache
    CACHE_TTL: int = 3600  # 1 heure
    CACHE_MAX_SIZE: int = 1000
    
    # Développement
    DEBUG: bool = False
    TESTING: bool = False
    
    # CORS
    CORS_ORIGINS: list = ["*"]
    CORS_METHODS: list = ["*"]
    CORS_HEADERS: list = ["*"]
    
    # Webhooks et notifications
    WEBHOOK_URL: Optional[str] = None
    SLACK_WEBHOOK_URL: Optional[str] = None
    
    # Configuration Docker
    DOCKER_REGISTRY: str = "localhost:5000"
    DOCKER_IMAGE_TAG: str = "latest"
    
    # Configuration de production
    ENVIRONMENT: str = "development"  # development, staging, production
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"  # Ignore les champs supplémentaires du .env


# Instance globale des paramètres
settings = Settings()


def get_settings() -> Settings:
    """Retourne l'instance des paramètres de configuration"""
    return settings


def create_directories():
    """Crée les répertoires nécessaires s'ils n'existent pas"""
    directories = [
        settings.DATA_DIR,
        settings.RAW_DATA_DIR,
        settings.PROCESSED_DATA_DIR,
        settings.MODELS_DIR,
        settings.RESULTS_DIR,
        settings.EXPORTS_DIR,
        "logs",
        "src/prompts",
        "src/templates"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Répertoire créé/vérifié: {directory}")


def validate_configuration():
    """Valide la configuration de l'application"""
    errors = []
    
    # Vérification des clés API
    if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY == "your-openai-api-key":
        errors.append("❌ OPENAI_API_KEY non configurée")
    
    # Vérification des répertoires
    if not os.path.exists(settings.DATA_DIR):
        errors.append(f"❌ Répertoire de données manquant: {settings.DATA_DIR}")
    
    if errors:
        print("🚨 Erreurs de configuration détectées:")
        for error in errors:
            print(f"  {error}")
        return False
    
    print("✅ Configuration validée avec succès")
    return True


if __name__ == "__main__":
    # Test de la configuration
    print("🔧 Test de la configuration...")
    print(f"📋 Projet: {settings.PROJECT_NAME}")
    print(f"👤 Auteur: {settings.PROJECT_AUTHOR}")
    print(f"📧 Email: {settings.PROJECT_EMAIL}")
    print(f"🔢 Version: {settings.VERSION}")
    print(f"🌍 Environnement: {settings.ENVIRONMENT}")
    
    # Création des répertoires
    create_directories()
    
    # Validation
    validate_configuration()