"""
Gestionnaire de modèles de langage pour GPT et LLaMA

Ce module implémente un gestionnaire unifié pour les modèles GPT-3.5-turbo
et LLaMA 3.1 8B, permettant la génération de texte et la comparaison.

Auteur: Dady Akrou Cyrille
Email: cyrilledady0501@gmail.com
Date: 2024
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import openai
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class GenerationResult:
    """Résultat d'une génération de texte"""
    text: str
    model_type: str
    generation_time: float
    tokens_used: int
    cost_estimate: float
    metadata: Dict[str, Any]


class ModelManager:
    """Gestionnaire unifié pour les modèles GPT et LLaMA"""
    
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.model_info = {
            "gpt": {
                "name": "GPT-3.5-turbo",
                "provider": "OpenAI",
                "loaded": False,
                "cost_per_token": 0.002 / 1000  # $0.002 per 1K tokens
            },
            "llama": {
                "name": "LLaMA 3.1 8B",
                "provider": "Meta/Hugging Face",
                "loaded": False,
                "cost_per_token": 0.0  # Gratuit en local
            }
        }
        
        # Configuration OpenAI
        if settings.OPENAI_API_KEY:
            openai.api_key = settings.OPENAI_API_KEY
        
        logger.info("🤖 ModelManager initialisé")
    
    def load_model(self, model_type: str) -> bool:
        """Charge un modèle spécifique"""
        try:
            if model_type == "gpt":
                return self._load_gpt_model()
            elif model_type == "llama":
                return self._load_llama_model()
            else:
                raise ValueError(f"Type de modèle non supporté: {model_type}")
                
        except Exception as e:
            logger.error(f"❌ Erreur chargement {model_type}: {e}")
            return False
    
    def _load_gpt_model(self) -> bool:
        """Charge le modèle GPT"""
        try:
            if not settings.OPENAI_API_KEY:
                raise ValueError("Clé API OpenAI manquante")
            
            # Test de connexion
            openai.api_key = settings.OPENAI_API_KEY
            
            # Marquer comme chargé
            self.model_info["gpt"]["loaded"] = True
            logger.info("✅ Modèle GPT-3.5-turbo chargé")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement GPT: {e}")
            return False
    
    def _load_llama_model(self) -> bool:
        """Charge le modèle LLaMA"""
        try:
            model_name = settings.LLAMA_MODEL_NAME
            
            logger.info(f"🔄 Chargement de {model_name}...")
            
            # Chargement du tokenizer
            self.tokenizers["llama"] = AutoTokenizer.from_pretrained(
                model_name,
                token=settings.HUGGINGFACE_TOKEN
            )
            
            # Configuration du device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Chargement du modèle
            self.models["llama"] = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=settings.HUGGINGFACE_TOKEN,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                low_cpu_mem_usage=True
            )
            
            # Création du pipeline
            self.models["llama_pipeline"] = pipeline(
                "text-generation",
                model=self.models["llama"],
                tokenizer=self.tokenizers["llama"],
                device=0 if device == "cuda" else -1
            )
            
            self.model_info["llama"]["loaded"] = True
            logger.info(f"✅ Modèle LLaMA chargé sur {device}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement LLaMA: {e}")
            return False
    
    def unload_model(self, model_type: str):
        """Décharge un modèle spécifique"""
        try:
            if model_type == "gpt":
                # GPT n'a pas besoin de déchargement explicite
                self.model_info["gpt"]["loaded"] = False
                
            elif model_type == "llama":
                if "llama" in self.models:
                    del self.models["llama"]
                if "llama_pipeline" in self.models:
                    del self.models["llama_pipeline"]
                if "llama" in self.tokenizers:
                    del self.tokenizers["llama"]
                
                # Nettoyage mémoire GPU
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                self.model_info["llama"]["loaded"] = False
            
            logger.info(f"🗑️ Modèle {model_type} déchargé")
            
        except Exception as e:
            logger.error(f"❌ Erreur déchargement {model_type}: {e}")
    
    def unload_all_models(self):
        """Décharge tous les modèles"""
        for model_type in ["gpt", "llama"]:
            self.unload_model(model_type)
    
    async def generate_text(
        self,
        prompt: str,
        model_type: str,
        max_tokens: int = None,
        temperature: float = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Génère du texte avec le modèle spécifié"""
        
        # Paramètres par défaut
        max_tokens = max_tokens or settings.OPENAI_MAX_TOKENS
        temperature = temperature or settings.OPENAI_TEMPERATURE
        
        start_time = time.time()
        
        try:
            if model_type == "gpt":
                result = await self._generate_with_gpt(prompt, max_tokens, temperature, **kwargs)
            elif model_type == "llama":
                result = await self._generate_with_llama(prompt, max_tokens, temperature, **kwargs)
            else:
                raise ValueError(f"Type de modèle non supporté: {model_type}")
            
            generation_time = time.time() - start_time
            
            # Calcul du coût
            tokens_used = result.get("tokens_used", 0)
            cost_estimate = tokens_used * self.model_info[model_type]["cost_per_token"]
            
            return {
                "text": result["text"],
                "tokens_used": tokens_used,
                "cost_estimate": cost_estimate,
                "generation_time": generation_time,
                "metadata": result.get("metadata", {})
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur génération {model_type}: {e}")
            raise
    
    async def _generate_with_gpt(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        **kwargs
    ) -> Dict[str, Any]:
        """Génération avec GPT"""
        
        if not self.model_info["gpt"]["loaded"]:
            raise RuntimeError("Modèle GPT non chargé")
        
        try:
            response = await asyncio.to_thread(
                openai.ChatCompletion.create,
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            generated_text = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            
            return {
                "text": generated_text,
                "tokens_used": tokens_used,
                "metadata": {
                    "model": "gpt-3.5-turbo",
                    "finish_reason": response.choices[0].finish_reason
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur génération GPT: {e}")
            raise
    
    async def _generate_with_llama(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        **kwargs
    ) -> Dict[str, Any]:
        """Génération avec LLaMA"""
        
        if not self.model_info["llama"]["loaded"]:
            raise RuntimeError("Modèle LLaMA non chargé")
        
        try:
            pipeline = self.models["llama_pipeline"]
            
            # Génération en thread séparé pour éviter le blocage
            result = await asyncio.to_thread(
                pipeline,
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=pipeline.tokenizer.eos_token_id,
                **kwargs
            )
            
            generated_text = result[0]["generated_text"]
            
            # Retirer le prompt original du résultat
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            # Estimation des tokens
            tokens_used = len(self.tokenizers["llama"].encode(generated_text))
            
            return {
                "text": generated_text,
                "tokens_used": tokens_used,
                "metadata": {
                    "model": settings.LLAMA_MODEL_NAME,
                    "device": "cuda" if torch.cuda.is_available() else "cpu"
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur génération LLaMA: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Retourne les informations sur les modèles"""
        return self.model_info.copy()
    
    def get_model_status(self) -> Dict[str, str]:
        """Retourne le statut des modèles"""
        return {
            model_type: "loaded" if info["loaded"] else "unloaded"
            for model_type, info in self.model_info.items()
        }
    
    def is_model_loaded(self, model_type: str) -> bool:
        """Vérifie si un modèle est chargé"""
        return self.model_info.get(model_type, {}).get("loaded", False)


# Instance globale du gestionnaire
_model_manager = None


def get_model_manager() -> ModelManager:
    """Retourne l'instance globale du gestionnaire de modèles"""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager