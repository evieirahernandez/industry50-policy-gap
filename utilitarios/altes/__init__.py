"""
ALTES - Automatic Labeling of Topics with External Sources

Módulo para rotulação automática de tópicos usando fontes externas.
Baseado no paper "Automatic Labeling of Topics with External Sources".

Example:
    >>> from utilitarios.altes import ALTES
    >>> 
    >>> titles = ["Machine Learning in Industry 4.0", "Digital Transformation"]
    >>> topics = [{'topic_id': 0, 'words': ['machine', 'learning', 'industry']}]
    >>> 
    >>> altes = ALTES(titles=titles)
    >>> result = altes.label_all_topics(topics)
    >>> print(result['labels'])
"""

from .altes import ALTES
from .text_processor import TitleProcessor
from .similarity import (
    calculate_cosine_similarity,
    calculate_angular_similarity,
    calculate_euclidean_distance,
    minmax_norm,
)

__all__ = [
    'ALTES',
    'TitleProcessor',
    'calculate_cosine_similarity',
    'calculate_angular_similarity',
    'calculate_euclidean_distance',
    'minmax_norm',
]
