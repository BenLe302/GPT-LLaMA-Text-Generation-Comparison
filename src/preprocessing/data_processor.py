"""
Préprocesseur de données pour le dataset TWCS
Auteur: Dady Akrou Cyrille
Email: cyrilledady0501@gmail.com
"""

import pandas as pd
import numpy as np
import re
import json
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from loguru import logger

from ..config import settings


class TWCSDataProcessor:
    """Processeur pour le dataset Twitter Customer Support"""
    
    def __init__(self, dataset_path: str = None):
        self.dataset_path = dataset_path or settings.TWCS_DATASET_PATH
        self.processed_path = settings.TWCS_PROCESSED_PATH
        self.df = None
        self.conversations = None
        self.stats = {}
        
    def load_data(self) -> pd.DataFrame:
        """Charge le dataset TWCS"""
        try:
            logger.info(f"📂 Chargement du dataset: {self.dataset_path}")
            self.df = pd.read_csv(self.dataset_path)
            logger.info(f"✅ Dataset chargé: {len(self.df)} tweets")
            return self.df
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement: {e}")
            raise
    
    def analyze_data(self) -> Dict:
        """Analyse exploratoire du dataset"""
        if self.df is None:
            self.load_data()
        
        logger.info("🔍 Analyse exploratoire des données...")
        
        # Statistiques de base
        self.stats = {
            "total_tweets": len(self.df),
            "unique_authors": self.df['author_id'].nunique(),
            "inbound_tweets": len(self.df[self.df['inbound'] == True]),
            "outbound_tweets": len(self.df[self.df['inbound'] == False]),
            "tweets_with_responses": len(self.df[self.df['response_tweet_id'].notna()]),
            "tweets_as_responses": len(self.df[self.df['in_response_to_tweet_id'].notna()]),
        }
        
        # Analyse temporelle
        self.df['created_at'] = pd.to_datetime(self.df['created_at'])
        self.stats['date_range'] = {
            'start': self.df['created_at'].min().strftime('%Y-%m-%d'),
            'end': self.df['created_at'].max().strftime('%Y-%m-%d')
        }
        
        # Analyse du texte
        self.df['text_length'] = self.df['text'].str.len()
        self.stats['text_stats'] = {
            'avg_length': self.df['text_length'].mean(),
            'median_length': self.df['text_length'].median(),
            'max_length': self.df['text_length'].max(),
            'min_length': self.df['text_length'].min()
        }
        
        # Identification des compagnies (tweets sortants)
        companies = self.df[self.df['inbound'] == False]['author_id'].value_counts()
        self.stats['top_companies'] = companies.head(10).to_dict()
        
        logger.info("✅ Analyse terminée")
        return self.stats
    
    def clean_text(self, text: str) -> str:
        """Nettoie le texte des tweets"""
        if pd.isna(text):
            return ""
        
        # Conversion en string
        text = str(text)
        
        # Suppression des URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Suppression des mentions @username (remplacées par des IDs)
        text = re.sub(r'@\w+', '', text)
        
        # Suppression des hashtags mais conservation du texte
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Suppression des caractères spéciaux excessifs
        text = re.sub(r'[^\w\s.,!?;:\'"()-]', '', text)
        
        # Normalisation des espaces
        text = re.sub(r'\s+', ' ', text)
        
        # Suppression des espaces en début/fin
        text = text.strip()
        
        return text
    
    def extract_conversations(self) -> List[Dict]:
        """Extrait les conversations complètes du dataset"""
        if self.df is None:
            self.load_data()
        
        logger.info("💬 Extraction des conversations...")
        
        conversations = []
        
        # Grouper par conversation (tweets liés par response_tweet_id)
        conversation_map = {}
        
        for _, tweet in self.df.iterrows():
            tweet_id = tweet['tweet_id']
            response_to = tweet['in_response_to_tweet_id']
            
            if pd.isna(response_to):
                # Tweet initial
                conversation_map[tweet_id] = {
                    'conversation_id': tweet_id,
                    'tweets': [tweet.to_dict()],
                    'participants': [tweet['author_id']],
                    'start_time': tweet['created_at']
                }
            else:
                # Réponse à un tweet existant
                # Trouver la conversation parent
                parent_conv = None
                for conv_id, conv_data in conversation_map.items():
                    if any(t['tweet_id'] == response_to for t in conv_data['tweets']):
                        parent_conv = conv_id
                        break
                
                if parent_conv:
                    conversation_map[parent_conv]['tweets'].append(tweet.to_dict())
                    if tweet['author_id'] not in conversation_map[parent_conv]['participants']:
                        conversation_map[parent_conv]['participants'].append(tweet['author_id'])
        
        # Filtrer les conversations avec au moins un échange client-support
        for conv_id, conv_data in conversation_map.items():
            tweets = conv_data['tweets']
            if len(tweets) >= 2:  # Au moins 2 messages
                # Vérifier qu'il y a un mélange inbound/outbound
                inbound_count = sum(1 for t in tweets if t['inbound'])
                outbound_count = len(tweets) - inbound_count
                
                if inbound_count > 0 and outbound_count > 0:
                    # Trier par date
                    tweets.sort(key=lambda x: x['created_at'])
                    
                    # Nettoyer les textes
                    for tweet in tweets:
                        tweet['cleaned_text'] = self.clean_text(tweet['text'])
                    
                    conversations.append({
                        'conversation_id': conv_id,
                        'tweets': tweets,
                        'length': len(tweets),
                        'participants': conv_data['participants'],
                        'duration_minutes': self._calculate_duration(tweets),
                        'customer_messages': inbound_count,
                        'support_messages': outbound_count
                    })
        
        self.conversations = conversations
        logger.info(f"✅ {len(conversations)} conversations extraites")
        return conversations
    
    def _calculate_duration(self, tweets: List[Dict]) -> float:
        """Calcule la durée d'une conversation en minutes"""
        if len(tweets) < 2:
            return 0
        
        start_time = pd.to_datetime(tweets[0]['created_at'])
        end_time = pd.to_datetime(tweets[-1]['created_at'])
        duration = (end_time - start_time).total_seconds() / 60
        return round(duration, 2)
    
    def create_training_pairs(self) -> List[Dict]:
        """Crée des paires question-réponse pour l'entraînement"""
        if self.conversations is None:
            self.extract_conversations()
        
        logger.info("🎯 Création des paires d'entraînement...")
        
        training_pairs = []
        
        for conversation in self.conversations:
            tweets = conversation['tweets']
            
            for i in range(len(tweets) - 1):
                current_tweet = tweets[i]
                next_tweet = tweets[i + 1]
                
                # Paire client -> support
                if current_tweet['inbound'] and not next_tweet['inbound']:
                    training_pairs.append({
                        'conversation_id': conversation['conversation_id'],
                        'customer_message': current_tweet['cleaned_text'],
                        'support_response': next_tweet['cleaned_text'],
                        'customer_id': current_tweet['author_id'],
                        'support_id': next_tweet['author_id'],
                        'response_time_minutes': self._calculate_response_time(current_tweet, next_tweet),
                        'context': self._get_context(tweets, i),
                        'message_type': 'customer_to_support'
                    })
                
                # Paire support -> client (pour comprendre les suivis)
                elif not current_tweet['inbound'] and next_tweet['inbound']:
                    training_pairs.append({
                        'conversation_id': conversation['conversation_id'],
                        'support_message': current_tweet['cleaned_text'],
                        'customer_response': next_tweet['cleaned_text'],
                        'support_id': current_tweet['author_id'],
                        'customer_id': next_tweet['author_id'],
                        'response_time_minutes': self._calculate_response_time(current_tweet, next_tweet),
                        'context': self._get_context(tweets, i),
                        'message_type': 'support_to_customer'
                    })
        
        logger.info(f"✅ {len(training_pairs)} paires d'entraînement créées")
        return training_pairs
    
    def _calculate_response_time(self, tweet1: Dict, tweet2: Dict) -> float:
        """Calcule le temps de réponse entre deux tweets"""
        time1 = pd.to_datetime(tweet1['created_at'])
        time2 = pd.to_datetime(tweet2['created_at'])
        return round((time2 - time1).total_seconds() / 60, 2)
    
    def _get_context(self, tweets: List[Dict], current_index: int, context_length: int = 3) -> str:
        """Récupère le contexte précédent d'une conversation"""
        start_index = max(0, current_index - context_length)
        context_tweets = tweets[start_index:current_index]
        
        context = []
        for tweet in context_tweets:
            role = "Client" if tweet['inbound'] else "Support"
            context.append(f"{role}: {tweet['cleaned_text']}")
        
        return " | ".join(context)
    
    def save_processed_data(self, training_pairs: List[Dict] = None):
        """Sauvegarde les données traitées"""
        if training_pairs is None:
            training_pairs = self.create_training_pairs()
        
        # Créer le répertoire si nécessaire
        Path(self.processed_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder les paires d'entraînement
        df_pairs = pd.DataFrame(training_pairs)
        df_pairs.to_csv(self.processed_path, index=False)
        
        # Sauvegarder les statistiques
        stats_path = self.processed_path.replace('.csv', '_stats.json')
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False, default=str)
        
        # Sauvegarder les conversations
        conversations_path = self.processed_path.replace('.csv', '_conversations.json')
        with open(conversations_path, 'w', encoding='utf-8') as f:
            json.dump(self.conversations, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"✅ Données sauvegardées:")
        logger.info(f"   - Paires d'entraînement: {self.processed_path}")
        logger.info(f"   - Statistiques: {stats_path}")
        logger.info(f"   - Conversations: {conversations_path}")
    
    def generate_visualizations(self, save_path: str = None):
        """Génère des visualisations des données"""
        if self.df is None:
            self.load_data()
        
        if save_path is None:
            save_path = settings.RESULTS_DIR
        
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        # Configuration du style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Distribution des longueurs de texte
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.hist(self.df['text_length'], bins=50, alpha=0.7, edgecolor='black')
        plt.title('Distribution des Longueurs de Texte')
        plt.xlabel('Longueur (caractères)')
        plt.ylabel('Fréquence')
        
        # 2. Tweets entrants vs sortants
        plt.subplot(1, 2, 2)
        inbound_counts = self.df['inbound'].value_counts()
        plt.pie(inbound_counts.values, labels=['Support → Client', 'Client → Support'], 
                autopct='%1.1f%%', startangle=90)
        plt.title('Répartition des Types de Messages')
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/twcs_analysis_basic.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Activité par heure
        if 'created_at' in self.df.columns:
            self.df['hour'] = pd.to_datetime(self.df['created_at']).dt.hour
            
            plt.figure(figsize=(12, 6))
            hourly_activity = self.df.groupby(['hour', 'inbound']).size().unstack(fill_value=0)
            hourly_activity.plot(kind='bar', stacked=True, figsize=(12, 6))
            plt.title('Activité par Heure de la Journée')
            plt.xlabel('Heure')
            plt.ylabel('Nombre de Tweets')
            plt.legend(['Support → Client', 'Client → Support'])
            plt.xticks(rotation=0)
            plt.tight_layout()
            plt.savefig(f"{save_path}/twcs_hourly_activity.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Top compagnies
        if self.stats and 'top_companies' in self.stats:
            plt.figure(figsize=(12, 8))
            companies = list(self.stats['top_companies'].keys())[:10]
            counts = list(self.stats['top_companies'].values())[:10]
            
            plt.barh(range(len(companies)), counts)
            plt.yticks(range(len(companies)), [f"Company_{i+1}" for i in range(len(companies))])
            plt.xlabel('Nombre de Messages')
            plt.title('Top 10 des Compagnies par Volume de Messages')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(f"{save_path}/twcs_top_companies.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"✅ Visualisations sauvegardées dans {save_path}")
    
    def get_sample_conversations(self, n: int = 5) -> List[Dict]:
        """Retourne un échantillon de conversations pour inspection"""
        if self.conversations is None:
            self.extract_conversations()
        
        # Trier par longueur et prendre un échantillon varié
        sorted_convs = sorted(self.conversations, key=lambda x: x['length'], reverse=True)
        
        sample = []
        step = max(1, len(sorted_convs) // n)
        
        for i in range(0, min(n * step, len(sorted_convs)), step):
            conv = sorted_convs[i]
            sample.append({
                'conversation_id': conv['conversation_id'],
                'length': conv['length'],
                'duration_minutes': conv['duration_minutes'],
                'messages': [
                    {
                        'role': 'Client' if tweet['inbound'] else 'Support',
                        'text': tweet['cleaned_text'][:100] + '...' if len(tweet['cleaned_text']) > 100 else tweet['cleaned_text'],
                        'timestamp': tweet['created_at']
                    }
                    for tweet in conv['tweets']
                ]
            })
        
        return sample
    
    def process_all(self) -> Dict:
        """Traite complètement le dataset"""
        logger.info("🚀 Traitement complet du dataset TWCS")
        
        # 1. Chargement et analyse
        self.load_data()
        stats = self.analyze_data()
        
        # 2. Extraction des conversations
        conversations = self.extract_conversations()
        
        # 3. Création des paires d'entraînement
        training_pairs = self.create_training_pairs()
        
        # 4. Sauvegarde
        self.save_processed_data(training_pairs)
        
        # 5. Visualisations
        self.generate_visualizations()
        
        # 6. Rapport final
        report = {
            'dataset_stats': stats,
            'conversations_count': len(conversations),
            'training_pairs_count': len(training_pairs),
            'sample_conversations': self.get_sample_conversations(3),
            'processing_complete': True,
            'processed_by': 'Dady Akrou Cyrille',
            'processing_date': datetime.now().isoformat()
        }
        
        logger.info("✅ Traitement terminé avec succès!")
        return report


if __name__ == "__main__":
    # Test du processeur
    processor = TWCSDataProcessor()
    report = processor.process_all()
    
    print("\n📊 Rapport de Traitement TWCS")
    print("=" * 50)
    print(f"📈 Total tweets: {report['dataset_stats']['total_tweets']:,}")
    print(f"💬 Conversations: {report['conversations_count']:,}")
    print(f"🎯 Paires d'entraînement: {report['training_pairs_count']:,}")
    print(f"👥 Auteurs uniques: {report['dataset_stats']['unique_authors']:,}")
    print(f"📅 Période: {report['dataset_stats']['date_range']['start']} → {report['dataset_stats']['date_range']['end']}")
    
    print("\n💬 Échantillon de Conversations:")
    for i, conv in enumerate(report['sample_conversations'], 1):
        print(f"\n{i}. Conversation {conv['conversation_id']} ({conv['length']} messages, {conv['duration_minutes']} min)")
        for msg in conv['messages'][:3]:  # Afficher les 3 premiers messages
            print(f"   {msg['role']}: {msg['text']}")
        if len(conv['messages']) > 3:
            print(f"   ... et {len(conv['messages']) - 3} autres messages")
    
    print(f"\n✅ Traitement par: {report['processed_by']}")
    print(f"📅 Date: {report['processing_date']}")