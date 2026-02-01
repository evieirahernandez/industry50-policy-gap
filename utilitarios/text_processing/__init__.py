"""
Módulo de processamento de texto para análise de tópicos
Contém utilitários para carregamento, preprocessamento e análise de texto
"""

from .text_utils import (
    DataLoader,
    TextPreprocessor,
    TopicAnalysisUtils,
    configurar_ambiente
)

__all__ = [
    'DataLoader',
    'TextPreprocessor',
    'TopicAnalysisUtils',
    'configurar_ambiente'
]
