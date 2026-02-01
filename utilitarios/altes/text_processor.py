"""
Processador de texto para o método ALTES.

Responsável pela limpeza e tokenização de títulos de artigos.
"""

import re
from typing import List, Set, Optional

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Garante recursos NLTK disponíveis
def _ensure_nltk_resources():
    """Baixa recursos NLTK se necessário."""
    try:
        stopwords.words('english')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    try:
        WordNetLemmatizer().lemmatize('test')
    except LookupError:
        nltk.download('wordnet', quiet=True)


class TitleProcessor:
    """
    Processa títulos de artigos para uso no método ALTES.
    
    Realiza:
    - Remoção de números
    - Remoção de pontuação
    - Remoção de stopwords
    - Lematização
    
    Attributes:
        titles: Lista de títulos originais
        extra_stopwords: Palavras adicionais a remover
    """
    
    # Pontuação a remover
    PUNCTUATION = r'#!"…¨$%&\'\"\'""()\*+,-.–/_:;<=>?\[\\]^_`{|}~•@'
    
    def __init__(self, titles: List[str], extra_stopwords: Optional[List[str]] = None):
        """
        Inicializa o processador.
        
        Args:
            titles: Lista de títulos de artigos
            extra_stopwords: Palavras adicionais a remover (ex: palavras do tópico)
        """
        _ensure_nltk_resources()
        
        self.titles = titles
        self.extra_stopwords = set(extra_stopwords or [])
        self._stop_words = set(stopwords.words('english')) | self.extra_stopwords
        self._lemmatizer = WordNetLemmatizer()
    
    def clean(self, text: str) -> str:
        """
        Limpa um único texto.
        
        Args:
            text: Texto a limpar
            
        Returns:
            Texto limpo e tokenizado
        """
        text = text.lower()
        # Remove números
        text = re.sub(r'[0-9]+', '', text)
        # Remove pontuação
        text = re.sub(f'[{re.escape(self.PUNCTUATION)}]+', ' ', text)
        # Normaliza espaços
        text = re.sub(r'\s+', ' ', text).strip()
        # Tokeniza, remove stopwords e lematiza
        tokens = [
            self._lemmatizer.lemmatize(word)
            for word in text.split()
            if len(word) > 2 and word not in self._stop_words
        ]
        return ' '.join(tokens)
    
    def process(self) -> List[str]:
        """
        Processa todos os títulos.
        
        Returns:
            Lista de títulos processados
        """
        return [self.clean(title) for title in self.titles]
    
    def get_tokens(self) -> List[str]:
        """
        Retorna todos os tokens de todos os títulos processados.
        
        Returns:
            Lista plana de tokens
        """
        processed = self.process()
        return [word for title in processed for word in title.split()]
