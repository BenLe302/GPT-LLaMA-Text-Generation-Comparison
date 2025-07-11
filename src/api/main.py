"""
API FastAPI pour la génération de texte avec GPT et LLaMA

Ce module implémente une API REST complète pour la génération de texte
utilisant les modèles GPT-3.5-turbo et LLaMA 3.1 8B.

Auteur: Dady Akrou Cyrille
Email: cyrilledady0501@gmail.com
Date: 2024
"""

import asyncio
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator

from src.config import settings
from src.models.model_manager import ModelManager
from src.utils.logger import get_logger

# ============================================================================
# Configuration de l'application
# ============================================================================

app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.PROJECT_DESCRIPTION,
    version=settings.PROJECT_VERSION,
    contact={
        "name": settings.PROJECT_AUTHOR,
        "email": settings.PROJECT_EMAIL
    },
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logger
logger = get_logger(__name__)

# Gestionnaire de modèles
model_manager = ModelManager()

# Sécurité optionnelle
security = HTTPBearer(auto_error=False) if settings.API_REQUIRE_AUTH else None


# ============================================================================
# Modèles Pydantic
# ============================================================================

class GenerationRequest(BaseModel):
    """Modèle pour les requêtes de génération de texte"""
    prompt: str = Field(..., min_length=1, max_length=2000, description="Texte d'entrée")
    model_type: str = Field(..., description="Type de modèle (gpt ou llama)")
    max_tokens: Optional[int] = Field(None, ge=1, le=4000, description="Nombre maximum de tokens")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Température de génération")
    system_message: Optional[str] = Field(None, max_length=500, description="Message système")
    context: Optional[str] = Field(None, max_length=1000, description="Contexte additionnel")
    
    @validator('model_type')
    def validate_model_type(cls, v):
        if v not in ['gpt', 'llama']:
            raise ValueError('model_type doit être "gpt" ou "llama"')
        return v


class ComparisonRequest(BaseModel):
    """Modèle pour les requêtes de comparaison de modèles"""
    prompt: str = Field(..., min_length=1, max_length=2000, description="Texte d'entrée")
    models: List[str] = Field(..., description="Liste des modèles à comparer")
    max_tokens: Optional[int] = Field(None, ge=1, le=4000, description="Nombre maximum de tokens")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Température de génération")
    system_message: Optional[str] = Field(None, max_length=500, description="Message système")
    
    @validator('models')
    def validate_models(cls, v):
        valid_models = ['gpt', 'llama']
        for model in v:
            if model not in valid_models:
                raise ValueError(f'Modèle "{model}" non supporté. Modèles valides: {valid_models}')
        if len(v) < 2:
            raise ValueError('Au moins 2 modèles requis pour la comparaison')
        return v


class GenerationResult(BaseModel):
    """Modèle pour les résultats de génération"""
    generated_text: str
    model_type: str
    generation_time: float
    tokens_used: int
    cost_estimate: float
    metadata: Dict[str, Any] = {}


class GenerationResponse(BaseModel):
    """Modèle pour les réponses de génération"""
    request_id: str
    result: GenerationResult
    timestamp: str
    success: bool = True


class ComparisonResponse(BaseModel):
    """Modèle pour les réponses de comparaison"""
    request_id: str
    results: Dict[str, Optional[GenerationResult]]
    comparison_metrics: Dict[str, Any]
    best_model: Optional[str]
    timestamp: str
    success: bool = True


class HealthResponse(BaseModel):
    """Modèle pour les réponses de santé"""
    status: str
    timestamp: str
    version: str
    models_status: Dict[str, str]
    uptime: float


class ErrorResponse(BaseModel):
    """Modèle pour les réponses d'erreur"""
    error: str
    error_code: str
    request_id: str
    timestamp: str
    success: bool = False


# ============================================================================
# Utilitaires
# ============================================================================

def generate_request_id() -> str:
    """Génère un ID unique pour la requête"""
    return str(uuid.uuid4())


def get_current_timestamp() -> str:
    """Retourne le timestamp actuel"""
    return datetime.now().isoformat()


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Authentification optionnelle"""
    if settings.API_REQUIRE_AUTH and not credentials:
        raise HTTPException(status_code=401, detail="Token d'authentification requis")
    return credentials


# ============================================================================
# Endpoints principaux
# ============================================================================

@app.get("/")
async def root():
    """Endpoint racine"""
    return {
        "message": f"Bienvenue sur {settings.PROJECT_NAME}",
        "author": settings.PROJECT_AUTHOR,
        "email": settings.PROJECT_EMAIL,
        "version": settings.PROJECT_VERSION,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Vérification de l'état de santé de l'API"""
    start_time = getattr(app.state, 'start_time', time.time())
    uptime = time.time() - start_time
    
    return HealthResponse(
        status="healthy",
        timestamp=get_current_timestamp(),
        version=settings.PROJECT_VERSION,
        models_status=model_manager.get_model_status(),
        uptime=uptime
    )


@app.post("/generate", response_model=GenerationResponse)
async def generate_text(
    request: GenerationRequest,
    background_tasks: BackgroundTasks,
    current_user=Depends(get_current_user)
):
    """
    Génère du texte avec le modèle spécifié
    
    - **prompt**: Texte d'entrée pour la génération
    - **model_type**: Type de modèle (gpt ou llama)
    - **max_tokens**: Nombre maximum de tokens (optionnel)
    - **temperature**: Température de génération (optionnel)
    - **system_message**: Message système (optionnel)
    - **context**: Contexte additionnel (optionnel)
    """
    request_id = generate_request_id()
    
    try:
        logger.info(f"🔄 Génération démarrée - ID: {request_id}, Modèle: {request.model_type}")
        
        # Préparation des paramètres
        generation_params = {
            "max_tokens": request.max_tokens or settings.OPENAI_MAX_TOKENS,
            "temperature": request.temperature or settings.OPENAI_TEMPERATURE
        }
        
        # Construction du prompt avec contexte et message système
        full_prompt = request.prompt
        if request.system_message:
            full_prompt = f"Système: {request.system_message}\n\nUtilisateur: {full_prompt}"
        if request.context:
            full_prompt = f"Contexte: {request.context}\n\n{full_prompt}"
        
        # Génération
        start_time = time.time()
        result = await model_manager.generate_text(
            prompt=full_prompt,
            model_type=request.model_type,
            **generation_params
        )
        generation_time = time.time() - start_time
        
        # Création de la réponse
        generation_result = GenerationResult(
            generated_text=result["text"],
            model_type=request.model_type,
            generation_time=generation_time,
            tokens_used=result.get("tokens_used", 0),
            cost_estimate=result.get("cost_estimate", 0.0),
            metadata=result.get("metadata", {})
        )
        
        # Log en arrière-plan
        background_tasks.add_task(
            log_generation_success,
            request_id,
            request.model_type,
            generation_time,
            generation_result.tokens_used
        )
        
        return GenerationResponse(
            request_id=request_id,
            result=generation_result,
            timestamp=get_current_timestamp()
        )
        
    except Exception as e:
        logger.error(f"❌ Erreur génération - ID: {request_id}, Erreur: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la génération: {str(e)}"
        )


@app.post("/compare", response_model=ComparisonResponse)
async def compare_models(
    request: ComparisonRequest,
    background_tasks: BackgroundTasks,
    current_user=Depends(get_current_user)
):
    """
    Compare plusieurs modèles sur le même prompt
    
    - **prompt**: Texte d'entrée pour la génération
    - **models**: Liste des modèles à comparer
    - **max_tokens**: Nombre maximum de tokens (optionnel)
    - **temperature**: Température de génération (optionnel)
    - **system_message**: Message système (optionnel)
    """
    request_id = generate_request_id()
    
    try:
        logger.info(f"🔄 Comparaison démarrée - ID: {request_id}, Modèles: {request.models}")
        
        # Préparation des paramètres
        generation_params = {
            "max_tokens": request.max_tokens or settings.OPENAI_MAX_TOKENS,
            "temperature": request.temperature or settings.OPENAI_TEMPERATURE
        }
        
        # Construction du prompt avec message système
        full_prompt = request.prompt
        if request.system_message:
            full_prompt = f"Système: {request.system_message}\n\nUtilisateur: {full_prompt}"
        
        # Génération parallèle pour tous les modèles
        tasks = []
        for model_type in request.models:
            task = asyncio.create_task(
                generate_with_model(model_type, full_prompt, generation_params)
            )
            tasks.append((model_type, task))
        
        # Attente des résultats
        results = {}
        for model_type, task in tasks:
            try:
                result = await task
                results[model_type] = result
            except Exception as e:
                logger.error(f"❌ Erreur modèle {model_type}: {e}")
                results[model_type] = None
        
        # Calcul des métriques de comparaison
        comparison_metrics = calculate_comparison_metrics(results)
        best_model = determine_best_model(results)
        
        # Log en arrière-plan
        background_tasks.add_task(
            log_comparison_success,
            request_id,
            request.models,
            comparison_metrics
        )
        
        return ComparisonResponse(
            request_id=request_id,
            results=results,
            comparison_metrics=comparison_metrics,
            best_model=best_model,
            timestamp=get_current_timestamp()
        )
        
    except Exception as e:
        logger.error(f"❌ Erreur comparaison - ID: {request_id}, Erreur: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la comparaison: {str(e)}"
        )


async def generate_with_model(model_type: str, prompt: str, params: Dict[str, Any]) -> GenerationResult:
    """Génère du texte avec un modèle spécifique"""
    start_time = time.time()
    
    result = await model_manager.generate_text(
        prompt=prompt,
        model_type=model_type,
        **params
    )
    
    generation_time = time.time() - start_time
    
    return GenerationResult(
        generated_text=result["text"],
        model_type=model_type,
        generation_time=generation_time,
        tokens_used=result.get("tokens_used", 0),
        cost_estimate=result.get("cost_estimate", 0.0),
        metadata=result.get("metadata", {})
    )


@app.get("/models", response_model=Dict[str, Any])
async def get_models_info():
    """Retourne les informations sur les modèles disponibles"""
    return {
        "available_models": ["gpt", "llama"],
        "models_info": model_manager.get_model_info(),
        "default_parameters": {
            "max_tokens": settings.OPENAI_MAX_TOKENS,
            "temperature": settings.OPENAI_TEMPERATURE
        }
    }


@app.post("/models/{model_type}/load")
async def load_model(model_type: str):
    """Charge un modèle spécifique"""
    try:
        if model_type not in ["gpt", "llama"]:
            raise HTTPException(status_code=400, detail="Modèle non supporté")
        
        model_manager.load_model(model_type)
        return {"message": f"Modèle {model_type} chargé avec succès"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/{model_type}/unload")
async def unload_model(model_type: str):
    """Décharge un modèle spécifique"""
    try:
        if model_type not in ["gpt", "llama"]:
            raise HTTPException(status_code=400, detail="Modèle non supporté")
        
        model_manager.unload_model(model_type)
        return {"message": f"Modèle {model_type} déchargé avec succès"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Fonctions utilitaires
# ============================================================================

def calculate_comparison_metrics(results: Dict[str, Optional[GenerationResult]]) -> Dict[str, Any]:
    """Calcule les métriques de comparaison entre modèles"""
    metrics = {
        "total_models": len(results),
        "successful_generations": sum(1 for r in results.values() if r is not None),
        "average_generation_time": 0,
        "total_tokens_used": 0,
        "total_cost_estimate": 0,
        "fastest_model": None,
        "most_efficient_model": None
    }
    
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if valid_results:
        # Temps moyen
        times = [r.generation_time for r in valid_results.values()]
        metrics["average_generation_time"] = sum(times) / len(times)
        
        # Total tokens et coût
        metrics["total_tokens_used"] = sum(r.tokens_used for r in valid_results.values())
        metrics["total_cost_estimate"] = sum(r.cost_estimate for r in valid_results.values())
        
        # Modèle le plus rapide
        fastest = min(valid_results.items(), key=lambda x: x[1].generation_time)
        metrics["fastest_model"] = fastest[0]
        
        # Modèle le plus efficace (ratio qualité/coût)
        if any(r.cost_estimate > 0 for r in valid_results.values()):
            most_efficient = min(
                valid_results.items(),
                key=lambda x: x[1].cost_estimate / max(x[1].tokens_used, 1)
            )
            metrics["most_efficient_model"] = most_efficient[0]
    
    return metrics


def determine_best_model(results: Dict[str, Optional[GenerationResult]]) -> Optional[str]:
    """Détermine le meilleur modèle basé sur plusieurs critères"""
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if not valid_results:
        return None
    
    # Score composite basé sur vitesse et efficacité
    scores = {}
    for model_name, result in valid_results.items():
        # Score basé sur la vitesse (inversé)
        speed_score = 1 / max(result.generation_time, 0.1)
        
        # Score basé sur l'efficacité (tokens/coût)
        if result.cost_estimate > 0:
            efficiency_score = result.tokens_used / result.cost_estimate
        else:
            efficiency_score = result.tokens_used  # LLaMA gratuit
        
        # Score composite
        scores[model_name] = speed_score * 0.3 + efficiency_score * 0.7
    
    return max(scores.items(), key=lambda x: x[1])[0]


async def log_generation_success(request_id: str, model_type: str, generation_time: float, tokens_used: int):
    """Log de succès de génération (tâche en arrière-plan)"""
    logger.info(f"📊 Génération terminée - ID: {request_id}, Modèle: {model_type}, "
                f"Temps: {generation_time:.2f}s, Tokens: {tokens_used}")


async def log_comparison_success(request_id: str, models: List[str], metrics: Dict[str, Any]):
    """Log de succès de comparaison (tâche en arrière-plan)"""
    logger.info(f"📊 Comparaison terminée - ID: {request_id}, Modèles: {models}, "
                f"Meilleur: {metrics.get('fastest_model', 'N/A')}")


# ============================================================================
# Gestionnaire d'erreurs
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Gestionnaire d'erreurs HTTP"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            error_code=f"HTTP_{exc.status_code}",
            request_id=generate_request_id(),
            timestamp=get_current_timestamp()
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Gestionnaire d'erreurs générales"""
    logger.error(f"❌ Erreur non gérée: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Erreur interne du serveur",
            error_code="INTERNAL_ERROR",
            request_id=generate_request_id(),
            timestamp=get_current_timestamp()
        ).dict()
    )


# ============================================================================
# Événements de démarrage/arrêt
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Événements au démarrage de l'application"""
    logger.info(f"🚀 Démarrage de {settings.PROJECT_NAME}")
    logger.info(f"👨‍💻 Développé par {settings.PROJECT_AUTHOR}")
    logger.info(f"📧 Contact: {settings.PROJECT_EMAIL}")
    logger.info(f"🌍 Environnement: {settings.ENVIRONMENT}")
    
    # Préchargement du modèle GPT (plus rapide)
    try:
        model_manager.load_model("gpt")
        logger.info("✅ Modèle GPT préchargé")
    except Exception as e:
        logger.warning(f"⚠️ Impossible de précharger GPT: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Événements à l'arrêt de l'application"""
    logger.info("🛑 Arrêt de l'application")
    
    # Déchargement des modèles
    try:
        model_manager.unload_all_models()
        logger.info("✅ Tous les modèles déchargés")
    except Exception as e:
        logger.error(f"❌ Erreur lors du déchargement: {e}")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
        workers=1 if settings.DEBUG else settings.API_WORKERS
    )