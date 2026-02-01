"""
Funções de cálculo de distância e similaridade para o método ALTES.

Fornece métricas de comparação entre vetores de embeddings.
"""

import math
import numpy as np
from scipy import spatial


def calculate_cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calcula a distância cosseno entre dois vetores.
    
    Args:
        a: Primeiro vetor
        b: Segundo vetor
        
    Returns:
        Distância cosseno (0 = idênticos, 2 = opostos)
    """
    return float(spatial.distance.cosine(a, b))


def calculate_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calcula a similaridade cosseno entre dois vetores.
    
    Args:
        a: Primeiro vetor
        b: Segundo vetor
        
    Returns:
        Similaridade cosseno (-1 a 1, onde 1 = idênticos)
    """
    return 1 - calculate_cosine_distance(a, b)


def calculate_angular_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calcula a distância angular entre dois vetores.
    
    Args:
        a: Primeiro vetor
        b: Segundo vetor
        
    Returns:
        Distância angular normalizada (0 a 1)
    """
    cosine_sim = calculate_cosine_similarity(a, b)
    # Limita valor entre -1 e 1 para evitar erros em acos
    clamped = max(-1, min(1, cosine_sim))
    return math.acos(clamped) / math.pi


def calculate_angular_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calcula a similaridade angular entre dois vetores.
    
    Args:
        a: Primeiro vetor
        b: Segundo vetor
        
    Returns:
        Similaridade angular (0 a 1, onde 1 = idênticos)
    """
    return 1 - calculate_angular_distance(a, b)


def calculate_euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calcula a distância euclidiana entre dois vetores.
    
    Args:
        a: Primeiro vetor
        b: Segundo vetor
        
    Returns:
        Distância euclidiana (>= 0)
    """
    return float(np.linalg.norm(a - b))


def minmax_norm(series) -> np.ndarray:
    """
    Aplica normalização min-max a uma série.
    
    Args:
        series: Array ou Series pandas
        
    Returns:
        Valores normalizados entre 0 e 1
    """
    min_val = series.min()
    max_val = series.max()
    range_val = max_val - min_val
    if range_val == 0:
        return series - min_val  # Retorna zeros se todos valores iguais
    return (series - min_val) / range_val
