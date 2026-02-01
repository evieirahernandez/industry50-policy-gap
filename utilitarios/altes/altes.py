"""
ALTES - Automatic Labeling of Topics with External Sources

Implementação modularizada do método descrito no paper 
"Automatic Labeling of Topics with External Sources".

Esta versão usa títulos de artigos como fonte externa em vez de Wikipedia.
"""

from typing import List, Dict, Optional, Any
import logging

import nltk
import numpy as np
import pandas as pd

from .similarity import (
    calculate_cosine_similarity,
    calculate_angular_similarity,
    calculate_euclidean_distance,
    minmax_norm,
)
from .text_processor import TitleProcessor

logger = logging.getLogger(__name__)


class ALTES:
    """
    Rotulação automática de tópicos usando fontes externas.
    
    O método funciona em 4 etapas:
    1. Processar títulos de artigos (fonte externa)
    2. Extrair candidatos via PMI (bigramas/trigramas)
    3. Ranquear candidatos por similaridade com palavras do tópico
    4. Selecionar melhor candidato como rótulo
    
    Example:
        >>> titles = ["Machine Learning in Industry 4.0", "Digital Transformation"]
        >>> topics = [{'topic_id': 0, 'words': ['machine', 'learning', 'industry']}]
        >>> altes = ALTES(titles=titles)
        >>> result = altes.label_all_topics(topics)
        >>> print(result['labels'])
    
    Attributes:
        titles: Lista de títulos de artigos
        pmi_threshold: Limiar mínimo de PMI para candidatos
        max_candidates: Número máximo de candidatos a considerar
    """
    
    DEFAULT_PMI_THRESHOLD = 6.0
    DEFAULT_MAX_CANDIDATES = 150
    DEFAULT_TOP_WORDS = 10
    
    def __init__(
        self,
        titles: List[str],
        embedding_model: Optional[Any] = None,
        pmi_threshold: float = DEFAULT_PMI_THRESHOLD,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
    ):
        """
        Inicializa o rotulador ALTES.
        
        Args:
            titles: Lista de títulos de artigos (fonte externa)
            embedding_model: Modelo de embeddings (compress_fasttext KeyedVectors).
                            Se None, tenta carregar automaticamente.
            pmi_threshold: Limiar mínimo de PMI para candidatos (default: 6.0)
            max_candidates: Número máximo de candidatos a processar (default: 150)
        """
        self.titles = titles
        self.pmi_threshold = pmi_threshold
        self.max_candidates = max_candidates
        self._embedding_model = embedding_model
    
    @property
    def embedding_model(self):
        """Carrega o modelo de embeddings sob demanda."""
        if self._embedding_model is None:
            self._embedding_model = self._load_default_model()
        return self._embedding_model
    
    def _load_default_model(self):
        """Carrega o modelo fastText comprimido padrão."""
        try:
            import compress_fasttext
            logger.info("Carregando modelo fastText comprimido...")
            model = compress_fasttext.models.CompressedFastTextKeyedVectors.load(
                'https://github.com/avidale/compress-fasttext/releases/'
                'download/gensim-4-draft/ft_cc.en.300_freqprune_400K_100K_pq_300.bin'
            )
            return model
        except ImportError:
            raise ImportError(
                "compress-fasttext não instalado. Instale com: pip install compress-fasttext"
            )
    
    def _generate_candidates(self, tokens: List[str]) -> List[tuple]:
        """
        Gera candidatos a rótulo usando PMI em bigramas e trigramas.
        
        Args:
            tokens: Lista de tokens processados
            
        Returns:
            Lista de tuplas (candidato, pmi_score) ordenada por PMI desc
        """
        if len(tokens) < 3:
            return []
        
        bigram_measures = nltk.collocations.BigramAssocMeasures()
        trigram_measures = nltk.collocations.TrigramAssocMeasures()
        
        bigram_finder = nltk.collocations.BigramCollocationFinder.from_words(tokens)
        trigram_finder = nltk.collocations.TrigramCollocationFinder.from_words(tokens)
        
        bigram_pmi = bigram_finder.score_ngrams(bigram_measures.pmi)
        trigram_pmi = trigram_finder.score_ngrams(trigram_measures.pmi)
        
        candidates = []
        for ngram, pmi in bigram_pmi:
            if pmi > self.pmi_threshold:
                candidates.append((ngram, pmi))
        for ngram, pmi in trigram_pmi:
            if pmi > self.pmi_threshold:
                candidates.append((ngram, pmi))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates
    
    def _get_average_vector(self, words: tuple) -> Optional[np.ndarray]:
        """
        Calcula o vetor médio para um conjunto de palavras.
        
        Args:
            words: Tupla de palavras
            
        Returns:
            Vetor médio ou None se nenhuma palavra encontrada
        """
        model = self.embedding_model
        vectors = [model[w] for w in words if w in model]
        if not vectors:
            return None
        return sum(vectors) / len(vectors)
    
    def _rank_candidates(
        self,
        topic_words: List[str],
        candidates: List[tuple],
    ) -> pd.DataFrame:
        """
        Ranqueia candidatos por similaridade com palavras do tópico.
        
        Args:
            topic_words: Lista de palavras do tópico
            candidates: Lista de candidatos (ngram, pmi)
            
        Returns:
            DataFrame com candidatos ranqueados por score composto
        """
        model = self.embedding_model
        
        # Vetores das palavras do tópico
        word_vectors = [model[w] for w in topic_words if w in model]
        if not word_vectors:
            logger.warning("Nenhuma palavra do tópico encontrada no modelo")
            return pd.DataFrame()
        
        # Limita número de candidatos e palavras
        candidates = candidates[:self.max_candidates]
        word_vectors = word_vectors[:self.DEFAULT_TOP_WORDS]
        
        results = []
        for ngram, pmi in candidates:
            cand_vector = self._get_average_vector(ngram)
            if cand_vector is None:
                continue
            
            # Calcula similaridades com cada palavra do tópico
            cos_scores = []
            ang_scores = []
            euc_scores = []
            
            for word_vec in word_vectors:
                try:
                    cos_scores.append(calculate_cosine_similarity(cand_vector, word_vec))
                    ang_scores.append(calculate_angular_similarity(cand_vector, word_vec))
                    euc_scores.append(calculate_euclidean_distance(cand_vector, word_vec))
                except Exception:
                    continue
            
            if not cos_scores:
                continue
            
            results.append({
                'label': ' '.join(ngram),
                'pmi': pmi,
                'cosine': np.mean(cos_scores),
                'angular': np.mean(ang_scores),
                'euclidean': np.mean(euc_scores),
            })
        
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        
        # Normaliza métricas
        df['cosine_n'] = minmax_norm(df['cosine'])
        df['angular_n'] = minmax_norm(df['angular'])
        df['euclidean_inv'] = 1 - minmax_norm(df['euclidean'])
        
        # Score composto: pondera cosseno mais alto
        df['score'] = (2 * df['cosine_n'] + df['angular_n'] + df['euclidean_inv']) / 4
        
        return df.sort_values(by='score', ascending=False).reset_index(drop=True)
    
    def label_topic(
        self,
        topic_words: List[str],
        weights: Optional[Dict[str, float]] = None,
        max_articles: int = 20,
    ) -> Dict[str, Any]:
        """
        Rotula um único tópico.
        
        Args:
            topic_words: Lista de palavras do tópico
            weights: Pesos das palavras (não usado atualmente, reservado para futuro)
            max_articles: Número máximo de artigos a usar
            
        Returns:
            Dict com:
                - label: Melhor rótulo encontrado
                - score: Score do rótulo
                - top_candidates: Top 5 candidatos com scores
        """
        # Processa títulos removendo palavras do tópico como stopwords extras
        processor = TitleProcessor(
            self.titles[:max_articles],
            extra_stopwords=topic_words
        )
        tokens = processor.get_tokens()
        
        if len(tokens) < 5:
            logger.warning("Poucos tokens para gerar candidatos")
            return {'label': None, 'score': 0.0, 'top_candidates': []}
        
        # Gera e ranqueia candidatos
        candidates = self._generate_candidates(tokens)
        if not candidates:
            logger.warning("Nenhum candidato com PMI acima do limiar")
            return {'label': None, 'score': 0.0, 'top_candidates': []}
        
        df_ranked = self._rank_candidates(topic_words, candidates)
        if df_ranked.empty:
            logger.warning("Ranking vazio - nenhum candidato válido")
            return {'label': None, 'score': 0.0, 'top_candidates': []}
        
        best = df_ranked.iloc[0]
        top5 = df_ranked.head(5).to_dict(orient='records')
        
        return {
            'label': best['label'],
            'score': float(best['score']),
            'top_candidates': top5,
        }
    
    def label_all_topics(
        self,
        topics: List[Dict],
        max_articles: int = 20,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Rotula múltiplos tópicos.
        
        Args:
            topics: Lista de dicts, cada um com:
                - topic_id: Identificador do tópico
                - words: Lista de palavras do tópico
                - weights: (opcional) Dict com pesos das palavras
            max_articles: Número máximo de artigos por tópico
            verbose: Se True, imprime progresso
            
        Returns:
            Dict com:
                - labels: Dict[topic_id, label]
                - scores: Dict[topic_id, score]
                - details: Dict[topic_id, top_candidates]
        """
        labels = {}
        scores = {}
        details = {}
        
        for topic in topics:
            topic_id = topic['topic_id']
            words = topic['words']
            weights = topic.get('weights')
            
            if verbose:
                logger.info(f"Processando tópico {topic_id}...")
            
            result = self.label_topic(words, weights, max_articles)
            
            labels[topic_id] = result['label']
            scores[topic_id] = result['score']
            details[topic_id] = result['top_candidates']
            
            if verbose:
                logger.info(f"  Rótulo: {result['label']} (score={result['score']:.3f})")
        
        return {
            'labels': labels,
            'scores': scores,
            'details': details,
        }
