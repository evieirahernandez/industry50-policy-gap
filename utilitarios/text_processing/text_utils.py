"""
Módulo de utilitários compartilhados para análise de tópicos
Contém funções de carregamento de dados, preprocessamento e análise
"""

import pandas as pd
import numpy as np
import spacy
import re
import os
from typing import List, Tuple, Dict, Any, Optional
from collections import Counter


class DataLoader:
    """Classe responsável pelo carregamento e validação dos dados"""

    @staticmethod
    def carregar_dados(caminho_arquivo: Optional[str] = None) -> pd.DataFrame:
        """
        Carrega os dados do arquivo CSV

        Args:
            caminho_arquivo: Caminho para o arquivo CSV (se None, usa o padrão)

        Returns:
            DataFrame com os dados carregados
        """
        print("Carregando dados...")

        if caminho_arquivo is None:
            # Encontra o diretório do projeto
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(script_dir))
            caminho_arquivo = os.path.join(project_root, 'dados', 'referencias_sem_duplicatas.csv')

        try:
            df = pd.read_csv(caminho_arquivo)
            print(f"Dados carregados: {df.shape[0]} registros, {df.shape[1]} colunas")
            return df
        except FileNotFoundError:
            raise FileNotFoundError(f"Arquivo não encontrado: {caminho_arquivo}")
        except Exception as e:
            raise Exception(f"Erro ao carregar dados: {str(e)}")

    @staticmethod
    def preprocessar_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Pré-processa o DataFrame combinando título e abstract

        Args:
            df: DataFrame com os dados originais

        Returns:
            DataFrame processado com coluna texto_completo
        """
        print("\n1. Concatenando título e abstract...")
        df_processado = df.copy()

        # Concatenar colunas abstract e titulo
        df_processado['texto_completo'] = (
            df_processado['Título'].fillna('') + ' ' +
            df_processado['Abstract'].fillna('')
        )

        # Verificar textos vazios
        textos_vazios = df_processado['texto_completo'].str.strip().eq('').sum()
        print(f"Textos vazios encontrados: {textos_vazios}")

        # Remover textos completamente vazios
        df_processado = df_processado[df_processado['texto_completo'].str.strip() != '']
        print(f"Registros após remoção de textos vazios: {df_processado.shape[0]}")

        return df_processado


class TextPreprocessor:
    """Classe responsável pelo preprocessamento de texto"""

    def __init__(self, modelo_spacy: str = 'en_core_web_sm'):
        """
        Inicializa o preprocessador

        Args:
            modelo_spacy: Nome do modelo spaCy a ser usado
        """
        self.modelo_spacy = modelo_spacy
        self.nlp = None
        self._carregar_modelo()

    def _carregar_modelo(self) -> None:
        """Carrega o modelo spaCy"""
        try:
            self.nlp = spacy.load(self.modelo_spacy)
            print(f"Modelo spaCy carregado: {self.modelo_spacy}")
        except OSError:
            print(f"Modelo {self.modelo_spacy} não encontrado. Será usado processamento básico.")
            self.nlp = None

    @staticmethod
    def limpar_texto(texto: str) -> str:
        """
        Limpa e normaliza o texto

        Args:
            texto: Texto a ser limpo

        Returns:
            Texto limpo e normalizado
        """
        if pd.isna(texto):
            return ""

        # Converter para minúsculas
        texto = texto.lower()

        # Remover caracteres especiais, mantendo apenas letras, números e espaços
        texto = re.sub(r'[^a-zA-Z0-9\s]', ' ', texto)

        # Remover múltiplos espaços
        texto = re.sub(r'\s+', ' ', texto)

        # Remover espaços no início e fim
        texto = texto.strip()

        return texto

    def preprocessar_textos(self, textos: List[str], usar_spacy: bool = True) -> List[str]:
        """
        Preprocessa uma lista de textos

        Args:
            textos: Lista de textos para processar
            usar_spacy: Se deve usar spaCy para processamento avançado

        Returns:
            Lista de textos processados
        """
        if usar_spacy and self.nlp is not None:
            return self._preprocessar_com_spacy(textos)
        else:
            return [self.limpar_texto(texto) for texto in textos]

    def _preprocessar_com_spacy(self, textos: List[str]) -> List[str]:
        """
        Preprocessa textos usando spaCy para inglês

        Args:
            textos: Lista de textos para processar

        Returns:
            Lista de textos processados
        """
        print(f"\n2. Processando textos com spaCy ({self.modelo_spacy})...")

        textos_processados = []

        for i, texto in enumerate(textos):
            if i % 100 == 0:
                print(f"Processando texto {i+1}/{len(textos)}")

            # Limpar texto básico primeiro
            texto_limpo = self.limpar_texto(texto)

            # Processar com spaCy
            doc = self.nlp(texto_limpo)

            # Extrair tokens relevantes (substantivos, adjetivos, verbos)
            tokens_relevantes = []
            for token in doc:
                if (token.pos_ in ['NOUN', 'ADJ', 'VERB'] and
                    not token.is_stop and
                    not token.is_punct and
                        len(token.text) > 2):
                    tokens_relevantes.append(token.lemma_)

            textos_processados.append(' '.join(tokens_relevantes))

        return textos_processados


class TopicAnalysisUtils:
    """Utilitários para análise de tópicos"""

    @staticmethod
    def filtrar_textos_validos(textos: List[str], df: pd.DataFrame) -> Tuple[List[str], pd.DataFrame]:
        """
        Filtra textos vazios e ajusta o DataFrame correspondente

        Args:
            textos: Lista de textos processados
            df: DataFrame original

        Returns:
            Tupla com textos válidos e DataFrame filtrado
        """
        # Filtrar textos vazios após processamento
        indices_validos = [i for i, texto in enumerate(textos) if texto.strip()]
        textos_nao_vazios = [textos[i] for i in indices_validos]
        df_filtrado = df.iloc[indices_validos].reset_index(drop=True)

        print(f"\nTextos válidos para análise: {len(textos_nao_vazios)}")

        if len(textos_nao_vazios) == 0:
            raise ValueError("Nenhum texto válido encontrado para análise!")

        return textos_nao_vazios, df_filtrado

    @staticmethod
    def analisar_distribuicao_topicos(topicos_dominantes: List[int],
                                      probabilidades: List[float] = None) -> Dict[int, int]:
        """
        Analisa a distribuição dos tópicos

        Args:
            topicos_dominantes: Lista com tópicos dominantes por documento
            probabilidades: Lista com probabilidades dos tópicos dominantes

        Returns:
            Dicionário com contagem por tópico
        """
        print("\n4. Analisando distribuição dos tópicos...")

        distribuicao = Counter(topicos_dominantes)
        total_docs = len(topicos_dominantes)

        print("\nDistribuição dos tópicos:")
        for topico, count in sorted(distribuicao.items()):
            percentual = (count / total_docs) * 100
            print(f"  Tópico {topico}: {count} documentos ({percentual:.1f}%)")

        return distribuicao

    @staticmethod
    def salvar_resultados(df: pd.DataFrame,
                          topicos_dominantes: List[int],
                          probabilidades: List[float],
                          caminho_saida: str) -> None:
        """
        Salva os resultados da análise de tópicos

        Args:
            df: DataFrame base
            topicos_dominantes: Lista de tópicos dominantes
            probabilidades: Lista de probabilidades
            caminho_saida: Caminho para salvar o arquivo
        """
        print("\n5. Salvando resultados...")

        df_resultado = df.copy()
        df_resultado['topico_dominante'] = topicos_dominantes
        df_resultado['probabilidade_topico'] = probabilidades

        df_resultado.to_csv(caminho_saida, index=False)
        print(f"Resultados salvos em: {caminho_saida}")


def configurar_ambiente():
    """Configura o ambiente para análise de tópicos"""
    print("Configurando ambiente para análise de tópicos...")
    print("=" * 60)
