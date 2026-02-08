#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Database Generator for Industry 5.0 Paper Exploration

Este módulo processa arquivos BibTeX de diferentes bases de dados acadêmicas,
remove duplicatas e gera um arquivo CSV consolidado com todas as referências.

Autor: Erika Vieira

"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import bibtexparser
import csv

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('database_generator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DatabaseNames:
    """Constantes para nomes das bases de dados."""
    SCOPUS = 'Scopus'
    SCIENCE_DIRECT = 'Science Direct'
    ACM = 'ACM'
    WEB_OF_SCIENCE = 'Web Of Science'
    IEEE = 'IEEE'
    SPRINGER_NATURE = 'Springer Nature'
    SPRINGER_NATURE = 'Springer Nature'
    OTHER = 'Outra'

MIN_YEAR = 2021


class BibTexProcessor:
    """Classe responsável pelo processamento de arquivos BibTeX."""

    ENCODINGS = ['utf-8', 'latin-1', 'cp1252']

    def __init__(self, data_directory: str):
        """
        Inicializa o processador BibTeX.

        Args:
            data_directory: Diretório contendo os arquivos BibTeX
        """
        self.data_directory = Path(data_directory)
        self._validate_directory()

    def _validate_directory(self) -> None:
        """Valida se o diretório existe."""
        if not self.data_directory.exists():
            raise FileNotFoundError(f"Diretório não encontrado: {self.data_directory}")
        if not self.data_directory.is_dir():
            raise NotADirectoryError(f"Caminho não é um diretório: {self.data_directory}")

    def read_bibtex_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Lê um arquivo BibTeX e retorna as entradas.

        Args:
            file_path: Caminho para o arquivo BibTeX

        Returns:
            Lista de entradas do arquivo BibTeX
        """
        for encoding in self.ENCODINGS:
            try:
                with open(file_path, 'r', encoding=encoding) as bibtex_file:
                    bib_database = bibtexparser.load(bibtex_file)
                    logger.info(f"Arquivo {file_path.name} lido com encoding {encoding}")
                    return bib_database.entries
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.error(f"Erro ao processar {file_path}: {e}")
                continue

        logger.error(f"Falha ao ler arquivo {file_path} com todos os encodings testados")
        return []

    def identify_database(self, filename: str) -> str:
        """
        Identifica a base de dados baseado no nome do arquivo.

        Args:
            filename: Nome do arquivo

        Returns:
            Nome da base de dados
        """
        filename_lower = filename.lower()

        database_mapping = {
            'scopus': DatabaseNames.SCOPUS,
            'sciencedirect': DatabaseNames.SCIENCE_DIRECT,
            'acm': DatabaseNames.ACM,
            'webofscience': DatabaseNames.WEB_OF_SCIENCE,
            'webofscience': DatabaseNames.WEB_OF_SCIENCE,
            'ieee': DatabaseNames.IEEE,
            'springernature': DatabaseNames.SPRINGER_NATURE
        }

        for prefix, database_name in database_mapping.items():
            if filename_lower.startswith(prefix):
                return database_name

        return DatabaseNames.OTHER


class CSVProcessor:
    """Classe responsável pelo processamento de arquivos CSV."""

    ENCODINGS = ['utf-8', 'latin-1', 'cp1252']

    def __init__(self, data_directory: str):
        """
        Inicializa o processador CSV.

        Args:
            data_directory: Diretório contendo os arquivos CSV
        """
        self.data_directory = Path(data_directory)
        self._validate_directory()

    def _validate_directory(self) -> None:
        """Valida se o diretório existe."""
        if not self.data_directory.exists():
            raise FileNotFoundError(f"Diretório não encontrado: {self.data_directory}")
        if not self.data_directory.is_dir():
            raise NotADirectoryError(f"Caminho não é um diretório: {self.data_directory}")

    def read_csv_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Lê um arquivo CSV e retorna as entradas.

        Args:
            file_path: Caminho para o arquivo CSV

        Returns:
            Lista de entradas do arquivo CSV
        """
        for encoding in self.ENCODINGS:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                logger.info(f"Arquivo CSV {file_path.name} lido com encoding {encoding}")
                return df.to_dict('records')  # type: ignore
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.error(f"Erro ao processar CSV {file_path}: {e}")
                continue

        logger.error(f"Falha ao ler arquivo CSV {file_path} com todos os encodings testados")
        return []

    def identify_database_csv(self, filename: str) -> str:
        """
        Identifica a base de dados baseado no nome do arquivo CSV.

        Args:
            filename: Nome do arquivo

        Returns:
            Nome da base de dados
        """
        filename_lower = filename.lower()

        if 'ieee' in filename_lower:
            return DatabaseNames.IEEE
        elif 'springer' in filename_lower:
            return DatabaseNames.SPRINGER_NATURE

        return DatabaseNames.OTHER


class ReferenceEntry:
    """Representa uma entrada de referência bibliográfica."""

    def __init__(self, bibtex_entry: Dict[str, Any], database: str):
        """
        Inicializa uma entrada de referência.

        Args:
            bibtex_entry: Entrada do BibTeX
            database: Nome da base de dados
        """
        self.id = bibtex_entry.get('ID', '')
        self.title = self._clean_title(bibtex_entry.get('title', ''))
        self.authors = bibtex_entry.get('author', '')
        self.year = bibtex_entry.get('year', '')
        self.journal = bibtex_entry.get('journal', bibtex_entry.get('booktitle', ''))
        self.doi = bibtex_entry.get('doi', '')
        self.abstract = bibtex_entry.get('abstract', '').strip()
        self.keywords = bibtex_entry.get('keywords', '')
        self.issn = bibtex_entry.get('issn', '').replace('-', '') # Normalize ISSN on entry
        self.issn = bibtex_entry.get('issn', '').replace('-', '') # Normalize ISSN on entry
        self.affiliation = bibtex_entry.get('affiliation', bibtex_entry.get('address', ''))
        self.database = database

    def _clean_title(self, title: str) -> str:
        """Remove caracteres especiais do título."""
        return title.replace('{', '').replace('}', '').strip()

    def is_valid(self) -> bool:
        """Verifica se a entrada tem dados mínimos necessários e não está vazia."""
        return bool(self.authors and self.title and self.abstract.strip() and len(self.abstract) > 0)

    def to_dict(self) -> Dict[str, str]:
        """Converte a entrada para dicionário."""
        return {
            'ID': self.id,
            'Título': self.title,
            'Autores': self.authors,
            'Ano': self.year,
            'Jornal/Conferência': self.journal,
            'DOI': self.doi,
            'Abstract': self.abstract,
            'Keywords': self.keywords,
            'ISSN': self.issn,
            'ISSN': self.issn,
            'Affiliation': self.affiliation,
            'Base': self.database
        }


class CSVReferenceEntry:
    """Representa uma entrada de referência bibliográfica de arquivo CSV."""

    def __init__(self, csv_entry: Dict[str, Any], database: str):
        """
        Inicializa uma entrada de referência CSV.

        Args:
            csv_entry: Entrada do CSV
            database: Nome da base de dados
        """
        self.database = database

        if database == DatabaseNames.IEEE:
            self._process_ieee_entry(csv_entry)
        elif database == DatabaseNames.SPRINGER_NATURE:
            self._process_springer_entry(csv_entry)
        else:
            raise ValueError(f"Base de dados não suportada: {database}")

    def _process_ieee_entry(self, entry: Dict[str, Any]) -> None:
        """Processa entrada do IEEE."""
        self.id = str(entry.get('Document Identifier', ''))
        self.title = self._clean_title(str(entry.get('Document Title', '')))
        self.authors = str(entry.get('Authors', ''))
        self.year = str(entry.get('Publication Year', ''))
        self.journal = str(entry.get('Publication Title', ''))
        self.doi = str(entry.get('DOI', ''))
        self.abstract = str(entry.get('Abstract', '')).strip()
        self.keywords = str(entry.get('Author Keywords', entry.get('IEEE Terms', ''))).strip()
        self.keywords = str(entry.get('Author Keywords', entry.get('IEEE Terms', ''))).strip()
        self.issn = str(entry.get('ISSN', '')).replace('-', '')
        self.affiliation = str(entry.get('Affiliations', entry.get('Author Affiliations', '')))

    def _process_springer_entry(self, entry: Dict[str, Any]) -> None:
        """Processa entrada do Springer Nature."""
        self.id = str(entry.get('Item DOI', ''))
        self.title = self._clean_title(str(entry.get('Item Title', '')))
        self.authors = str(entry.get('Authors', ''))
        self.year = str(entry.get('Publication Year', ''))
        self.journal = str(entry.get('Publication Title', ''))
        self.doi = str(entry.get('Item DOI', ''))
        self.abstract = str(entry.get('Abstract', '')).strip()
        self.keywords = ''  # Springer Nature CSV não tem keywords
        self.keywords = ''  # Springer Nature CSV não tem keywords
        self.issn = str(entry.get('Journal ISSN', '')).replace('-', '')
        self.affiliation = '' # Springer CSV typically lacks affiliation

    def _clean_title(self, title: str) -> str:
        """Remove caracteres especiais do título."""
        return title.replace('{', '').replace('}', '').strip()

    def is_valid(self) -> bool:
        """Verifica se a entrada tem dados mínimos necessários."""
        return bool(self.authors and self.title and self.abstract.strip() and len(self.abstract) > 0)

    def to_dict(self) -> Dict[str, str]:
        """Converte a entrada para dicionário."""
        return {
            'ID': self.id,
            'Título': self.title,
            'Autores': self.authors,
            'Ano': self.year,
            'Jornal/Conferência': self.journal,
            'DOI': self.doi,
            'Abstract': self.abstract,
            'Keywords': self.keywords,
            'ISSN': self.issn,
            'ISSN': self.issn,
            'Affiliation': self.affiliation,
            'Base': self.database
        }



class YearFilter:
    """Classe responsável por filtrar referências por ano."""

    @staticmethod
    def filter_dataframe(df: pd.DataFrame, min_year: int) -> pd.DataFrame:
        """
        Filtra DataFrame por ano mínimo.

        Args:
            df: DataFrame com as referências
            min_year: Ano mínimo (inclusivo)

        Returns:
            DataFrame filtrado
        """
        original_count = len(df)
        
        # Converte coluna Ano para numérico, forçando erros a NaN
        df['Ano_Num'] = pd.to_numeric(df['Ano'], errors='coerce')
        
        # Filtra anos válidos e maiores ou iguais ao min_year
        df_filtered = df[df['Ano_Num'] >= min_year].copy()
        
        # Remove coluna auxiliar
        df_filtered.drop(columns=['Ano_Num'], inplace=True)
        
        filtered_count = len(df_filtered)
        removed_count = original_count - filtered_count
        
        logger.info(f"Filtro de ano (>= {min_year}) aplicado: {filtered_count} mantidos, {removed_count} removidos")
        
        return df_filtered


class DuplicateRemover:
    """Classe responsável por remover duplicatas das referências."""

    @staticmethod
    def remove_duplicates_dataframe(df: pd.DataFrame, subset: Optional[List[str]] = None, keep: str = 'first') -> pd.DataFrame:
        """
        Remove duplicatas de um DataFrame.

        Args:
            df: DataFrame com as referências
            subset: Colunas para verificar duplicatas
            keep: Qual duplicata manter ('first', 'last', False)

        Returns:
            DataFrame sem duplicatas        """
        if subset is None:
            subset = ['Título']

        original_count = len(df)
        df_clean = df.drop_duplicates(subset=subset, keep=keep)  # type: ignore
        removed_count = original_count - len(df_clean)

        logger.info(f"Duplicatas removidas: {removed_count} de {original_count} entradas")
        return df_clean

    # remove_duplicates_two_stage removed as requested


class DatabaseStatistics:
    """Classe para gerar estatísticas das referências."""

    @staticmethod
    def generate_statistics(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Gera estatísticas completas do DataFrame.

        Args:
            df: DataFrame com as referências

        Returns:
            Dicionário com as estatísticas
        """
        total_refs = len(df)

        # Estatísticas por base
        database_counts = df['Base'].value_counts().to_dict()
        database_percentages = {
            base: (count / total_refs) * 100
            for base, count in database_counts.items()
        }

        # Estatísticas de abstracts
        refs_with_abstract = df[df['Abstract'].str.len() > 0]
        refs_without_abstract = df[df['Abstract'].str.len() == 0]

        abstract_stats = {
            'com_abstract': len(refs_with_abstract),
            'sem_abstract': len(refs_without_abstract),
            'percentual_com_abstract': (len(refs_with_abstract) / total_refs) * 100,
            'percentual_sem_abstract': (len(refs_without_abstract) / total_refs) * 100
        }

        # Estatísticas de keywords
        refs_with_keywords = df[df['Keywords'].str.len() > 0]
        refs_without_keywords = df[df['Keywords'].str.len() == 0]

        keywords_stats = {
            'com_keywords': len(refs_with_keywords),
            'sem_keywords': len(refs_without_keywords),
            'percentual_com_keywords': (len(refs_with_keywords) / total_refs) * 100,
            'percentual_sem_keywords': (len(refs_without_keywords) / total_refs) * 100
        }

        return {
            'total_referencias': total_refs,
            'min_year': int(df['Ano'].min()) if not df.empty and df['Ano'].notna().any() else 0,
            'por_base': {
                'contagem': database_counts,
                'percentual': database_percentages
            },
            'abstracts': abstract_stats,
            'keywords': keywords_stats
        }

    @staticmethod
    def print_statistics(stats: Dict[str, Any]) -> None:
        """Imprime as estatísticas de forma organizada."""
        print(f"\n=== ESTATÍSTICAS FINAIS ===")
        print(f"Total de referências: {stats['total_referencias']}")
        print(f"Ano mais antigo: {stats.get('min_year', 'N/A')}")

        print(f"\n=== DISTRIBUIÇÃO POR BASE ===")
        for base, count in stats['por_base']['contagem'].items():
            percentage = stats['por_base']['percentual'][base]
            print(f"{base}: {count} referências ({percentage:.2f}%)")

        print(f"\n=== ESTATÍSTICAS DE ABSTRACT ===")
        abstract_stats = stats['abstracts']
        print(f"Com abstract: {abstract_stats['com_abstract']} ({abstract_stats['percentual_com_abstract']:.2f}%)")
        print(f"Sem abstract: {abstract_stats['sem_abstract']} ({abstract_stats['percentual_sem_abstract']:.2f}%)")

        print(f"\n=== ESTATÍSTICAS DE KEYWORDS ===")
        keywords_stats = stats['keywords']
        print(f"Com keywords: {keywords_stats['com_keywords']} ({keywords_stats['percentual_com_keywords']:.2f}%)")
        print(f"Sem keywords: {keywords_stats['sem_keywords']} ({keywords_stats['percentual_sem_keywords']:.2f}%)")
        print(f"=" * 50)


class KeywordFilter:
    """Classe responsável por filtrar referências baseadas em palavras-chave."""

    # Palavras-chave para filtrar
    KEYWORDS = ['resilience', 'human-centricity', 'sustainability']

    @staticmethod
    def filter_dataframe_by_keywords(df: pd.DataFrame,
                                     keywords: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Filtra DataFrame por palavras-chave no título, abstract ou keywords.

        Args:
            df: DataFrame com as referências
            keywords: Lista de palavras-chave (usa padrão se None)

        Returns:
            DataFrame filtrado
        """
        if keywords is None:
            keywords = KeywordFilter.KEYWORDS

        original_count = len(df)
        keyword_pattern = '|'.join([kw.lower() for kw in keywords])

        # Cria máscaras para título, abstract e keywords
        title_mask = df['Título'].str.lower().str.contains(
            keyword_pattern,
            na=False,
            regex=True
        )

        abstract_mask = df['Abstract'].str.lower().str.contains(
            keyword_pattern,
            na=False,
            regex=True
        )

        keywords_mask = df['Keywords'].str.lower().str.contains(
            keyword_pattern,
            na=False,
            regex=True
        )

        # Filtra registros que têm palavras-chave no título, abstract OU keywords
        df_filtered = df[title_mask | abstract_mask | keywords_mask]

        filtered_count = len(df_filtered)
        logger.info(f"Filtro de palavras-chave aplicado: {filtered_count} de {original_count} referências mantidas")
        logger.info(f"Palavras-chave utilizadas: {', '.join(keywords)}")

        return df_filtered



class QualityFilter:
    """
    Classe responsável por enriquecer o dataset com dados do Scimago 
    e filtrar referências baseadas na qualidade do veículo (Q1/Q2).
    """

    @staticmethod
    def _normalize_text(text: pd.Series) -> pd.Series:
        """
        Remove pontuação, espaços extras e coloca em minúsculas para facilitar o merge.
        Ex: "IEEE Trans. on Ind. Inf." -> "ieeetransonindinf"
        """
        return (
            text.astype(str)
            .str.lower()
            .str.strip()
            .str.replace(r'[^\w\s]', '', regex=True) # Remove pontuação
            .str.replace(r'\s+', '', regex=True)     # Remove espaços
        )

    @staticmethod
    def filter_by_quality_proxy(df: pd.DataFrame, scimago_path: str = None) -> pd.DataFrame:
        """
        Filtra DataFrame por qualidade (Q1/Q2) cruzando com base externa do Scimago.
        Prioriza ISSN e depois Título.
        """
        if scimago_path is None:
            # Tenta localizar o arquivo padrão
            current_dir = os.path.dirname(os.path.dirname(__file__))
            scimago_path = os.path.join(current_dir, 'dados', 'ranking', 'scimago_rank_2024.csv')

        logger.info(f"=== Iniciando Enriquecimento e Filtro de Qualidade (Q1/Q2) ===")
        logger.info(f"Usando base Scimago: {scimago_path}")
        
        # 1. Verificar colunas necessárias
        source_col = 'Jornal/Conferência'
        if source_col not in df.columns:
            logger.warning(f"Coluna '{source_col}' não encontrada. Retornando dataset original.")
            return df
            
        # Garante que ISSN existe
        if 'ISSN' not in df.columns:
            df['ISSN'] = ''

        try:
            # 2. Carregar base Scimago
            # O separador do Scimago geralmente é ponto e vírgula (;), mas verifique seu arquivo.
            df_sjr = pd.read_csv(scimago_path, sep=';', on_bad_lines='skip')
            
            # O Scimago tem a coluna 'SJR Best Quartile' (Q1, Q2, etc) e 'Title'
            if 'SJR Best Quartile' not in df_sjr.columns:
                 # Tenta inferir se o CSV veio com outro formato ou separador errado
                 df_sjr = pd.read_csv(scimago_path, sep=',', on_bad_lines='skip')

            logger.info(f"Base Scimago carregada: {len(df_sjr)} revistas.")

        except FileNotFoundError:
            logger.error(f"Arquivo {scimago_path} não encontrado. Execute o filtro sem qualidade ou baixe o CSV.")
            return df

        # --- ESTRATÉGIA DE MERGE: 1. ISSN, 2. TÍTULO ---
        
        # Prepara df_sjr para explode de ISSNs (muitas vezes vem "xxxx-xxxx, yyyy-yyyy")
        # Vamos normalizar os ISSNs do SJR (remover hifens, separar por vírgula e explodir)
        df_sjr['Issn_Clean'] = df_sjr['Issn'].astype(str).str.replace('-', '').str.replace(' ', '')
        # Separar múltiplos ISSNs e explodir para linhas duplicadas (uma para cada ISSN)
        df_sjr_issn = df_sjr.assign(Issn_Clean=df_sjr['Issn_Clean'].str.split(',')).explode('Issn_Clean')
        
        # Normaliza ISSN do dataset principal
        df['ISSN_Clean'] = df['ISSN'].astype(str).str.replace('-', '').str.strip()
        
        # MERGE 1: Por ISSN
        # Faz left join onde ISSN não é vazio
        logger.info("Tentando match por ISSN...")
        
        # Separa os que têm ISSN válido para tentar match
        mask_has_issn = df['ISSN_Clean'].str.len() > 3 # Assumindo min 4 chars
        
        # Cria chaves
        df_merged_issn = df.merge(
            df_sjr_issn[['Issn_Clean', 'SJR Best Quartile']],
            left_on='ISSN_Clean',
            right_on='Issn_Clean',
            how='left',
            suffixes=('', '_issn')
        )
        
        # Se 'SJR Best Quartile' não estava no df original, renomeia a do merge
        if 'SJR Best Quartile' not in df_merged_issn.columns and 'SJR Best Quartile_issn' in df_merged_issn.columns:
             df_merged_issn.rename(columns={'SJR Best Quartile_issn': 'SJR Best Quartile'}, inplace=True)
        elif 'SJR Best Quartile_issn' in df_merged_issn.columns:
             # Preenche onde é nulo
             df_merged_issn['SJR Best Quartile'] = df_merged_issn['SJR Best Quartile'].fillna(df_merged_issn['SJR Best Quartile_issn'])

        # MERGE 2: Por Título (para os que ainda não tem Quartil)
        logger.info("Tentando match por Título para os restantes...")
        
        # Normaliza títulos
        df_merged_issn['join_key_title'] = QualityFilter._normalize_text(df_merged_issn[source_col])
        df_sjr['join_key_title'] = QualityFilter._normalize_text(df_sjr['Title'])
        
        # Remove duplicatas de titulo no SJR para não duplicar linhas no merge (pega o melhor ranking se houver duplicata de titulo)
        # Assumindo que o primeiro é o correto ou ordenando
        df_sjr_title = df_sjr.drop_duplicates(subset=['join_key_title'])

        df_final_merged = df_merged_issn.merge(
            df_sjr_title[['join_key_title', 'SJR Best Quartile']],
            on='join_key_title',
            how='left',
            suffixes=('', '_title')
        )
        
        # Consolida a coluna Quartile
        # Se Quartile (do ISSN) for nulo, pega do Título
        if 'SJR Best Quartile' in df_final_merged.columns:
             df_final_merged['SJR Best Quartile'] = df_final_merged['SJR Best Quartile'].fillna(df_final_merged['SJR Best Quartile_title'])
        else:
             df_final_merged.rename(columns={'SJR Best Quartile_title': 'SJR Best Quartile'}, inplace=True)

        # 5. Aplicar o Filtro
        quartiles_accepted = ['Q1', 'Q2']
        mask_accepted = df_final_merged['SJR Best Quartile'].isin(quartiles_accepted)
        
        df_filtered = df_final_merged[mask_accepted].copy()
        
        # Métricas
        total_inicial = len(df)
        total_final = len(df_filtered)
        sem_match = df_final_merged['SJR Best Quartile'].isna().sum()
        
        logger.info(f"Papers Iniciais: {total_inicial}")
        logger.info(f"Papers sem correspondência no Scimago: {sem_match}")
        logger.info(f"Papers Finais (Q1/Q2): {total_final}")

        # Limpeza final das colunas auxiliares
        cols_to_drop = ['Issn_Clean', 'join_key_title', 'SJR Best Quartile_title', 'SJR Best Quartile_issn', 'ISSN_Clean']
        df_filtered.drop(columns=[c for c in cols_to_drop if c in df_filtered.columns], inplace=True)
        
        # Garante que não temos linhas duplicadas introduzidas pelo merge (ex: same ISSN mapped twice in SJR exploded)
        df_filtered = df_filtered.drop_duplicates(subset=['ID', 'DOI', 'Título'])
        
        return df_filtered


class ComparativeMetrics:
    """Calcula métricas comparativas para análise do paper."""

    PILLARS = ['resilience', 'human-centricity', 'sustainability']

    @staticmethod
    def calculate_quality_alignment(df_organic: pd.DataFrame, df_quality: pd.DataFrame) -> Dict[str, Any]:
        """
        Quality Alignment Ratio: Compara frequência dos pilares entre D_total e D_quality.
        """
        logger.info("Calculando Quality Alignment Ratio...")
        
        def count_pillar_frequency(df):
            total = len(df)
            if total == 0: return {p: 0 for p in ComparativeMetrics.PILLARS}
            
            freqs = {}
            for pillar in ComparativeMetrics.PILLARS:
                pattern = pillar.lower()
                # Verifica presença em Title, Abstract ou Keywords
                mask = (
                    df['Título'].str.lower().str.contains(pattern, na=False) |
                    df['Abstract'].str.lower().str.contains(pattern, na=False) |
                    df['Keywords'].str.lower().str.contains(pattern, na=False)
                )
                count = mask.sum()
                freqs[pillar] = {
                    'count': int(count),
                    'ratio': float(count / total)
                }
            return freqs

        return {
            'organic': count_pillar_frequency(df_organic),
            'quality': count_pillar_frequency(df_quality)
        }



    @staticmethod
    def calculate_depth_of_adoption(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Depth of Adoption Analysis: Surface vs Core adoption.
        Surface: In Keywords/Title but NOT in Abstract.
        Core (Proxy): In Abstract. Recomputed after by verifying the presence in a reference BERTopic cluster
        """
        logger.info("Calculando Depth of Adoption...")
        stats = {}
        
        for pillar in ComparativeMetrics.PILLARS:
            pattern = pillar.lower()
            
            # Masks
            in_title_kw = (
                df['Título'].str.lower().str.contains(pattern, na=False) |
                df['Keywords'].str.lower().str.contains(pattern, na=False)
            )
            in_abstract = df['Abstract'].str.lower().str.contains(pattern, na=False)
            
            surface_mask = in_title_kw & (~in_abstract)
            core_proxy_mask = in_abstract # Usando Abstract presence como proxy para Core neste estágio
            
            stats[pillar] = {
                'surface_adoption_count': int(surface_mask.sum()),
                'core_adoption_proxy_count': int(core_proxy_mask.sum()),
                'total_mentions': int((in_title_kw | in_abstract).sum())
            }
            
        return stats

    @staticmethod
    def print_metrics(metrics: Dict[str, Any]) -> None:
        print("\n=== COMPARATIVE METRICS REPORT ===")
        
        print("\n--- Quality Alignment Ratio (Pillar adherence in Q1/Q2) ---")
        qa = metrics['quality_alignment']
        for pillar in ComparativeMetrics.PILLARS:
            org = qa['organic'].get(pillar, {'ratio': 0})
            qual = qa['quality'].get(pillar, {'ratio': 0})
            print(f"  {pillar.title()}: Organic {org['ratio']:.1%} vs Quality {qual['ratio']:.1%}")

        print("\n--- Depth of Adoption (Marketing vs Substance Gap) ---")
        da = metrics['depth_of_adoption']
        for pillar, data in da.items():
            surface = data['surface_adoption_count']
            core = data['core_adoption_proxy_count']
            print(f"  {pillar.title()}: Surface (Title/KW only) = {surface} | Core (Abstract) = {core}")

        print("==================================\n")



class DatabaseGenerator:

    """Classe principal para geração da base de dados consolidada."""

    def __init__(self, data_directory: Optional[str] = None, csv_directory: Optional[str] = None):
        """
        Inicializa o gerador de base de dados.

        Args:
            data_directory: Diretório com os arquivos BibTeX
            csv_directory: Diretório com os arquivos CSV        """
        if data_directory is None:
            data_directory = self._get_default_data_directory()

        if csv_directory is None:
            csv_directory = self._get_default_csv_directory()

        self.processor = BibTexProcessor(data_directory)
        self.csv_processor = CSVProcessor(csv_directory)
        self.year_filter = YearFilter()
        self.duplicate_remover = DuplicateRemover()
        self.statistics = DatabaseStatistics()
        self.keyword_filter = KeywordFilter()
        self.quality_filter = QualityFilter()
        self.comparative_metrics = ComparativeMetrics()

    def _get_default_data_directory(self) -> str:
        """Retorna o diretório padrão dos dados."""
        current_dir = os.path.dirname(os.path.dirname(__file__))
        return os.path.join(current_dir, 'dados', 'referencias_bibtex')

    def _get_default_csv_directory(self) -> str:
        """Retorna o diretório padrão dos arquivos CSV."""
        current_dir = os.path.dirname(os.path.dirname(__file__))
        return os.path.join(current_dir, 'dados', 'referencias_csv')

    def _get_output_path(self) -> str:
        """Retorna o caminho do arquivo de saída."""
        data_dir = os.path.dirname(self.processor.data_directory)
        return os.path.join(data_dir, 'referencias_sem_duplicatas.csv')

    def process_bibtex_files(self) -> List[ReferenceEntry]:
        """
        Processa arquivos BibTeX (somente bases primárias).
        Ignora arquivos 'periodicos*'.

        Returns:
            Lista de referências
        """
        bib_files = list(self.processor.data_directory.glob('*.bib'))
        logger.info(f"Encontrados {len(bib_files)} arquivos BibTeX")

        all_references = []

        for file_path in bib_files:
            # Ignora Periodicos CAPES explicitamente
            if 'periodicos' in file_path.name.lower():
                logger.info(f"Ignorando arquivo (Periodicos CAPES): {file_path.name}")
                continue
                
            logger.info(f"Processando: {file_path.name}")

            database_name = self.processor.identify_database(file_path.name)
            entries = self.processor.read_bibtex_file(file_path)

            references = [ReferenceEntry(entry, database_name) for entry in entries]
            valid_references = [ref for ref in references if ref.is_valid()]

            logger.info(f"Base: {database_name} - {len(valid_references)} referências válidas")
            all_references.extend(valid_references)

        return all_references

    def process_csv_files(self) -> List[CSVReferenceEntry]:
        """
        Processa todos os arquivos CSV.

        Returns:
            Lista de referências dos arquivos CSV
        """
        csv_files = list(self.csv_processor.data_directory.glob('*.csv'))
        logger.info(f"Encontrados {len(csv_files)} arquivos CSV")

        csv_references = []

        for file_path in csv_files:
            logger.info(f"Processando CSV: {file_path.name}")

            database_name = self.csv_processor.identify_database_csv(file_path.name)
            entries = self.csv_processor.read_csv_file(file_path)

            references = [CSVReferenceEntry(entry, database_name) for entry in entries]
            valid_references = [ref for ref in references if ref.is_valid()]

            logger.info(f"Base CSV: {database_name} - {len(valid_references)} referências válidas")
            csv_references.extend(valid_references)

        return csv_references

    def generate_database(self) -> pd.DataFrame:
        """
        Gera a base de dados consolidada incluindo BibTeX e CSV.

        Returns:
            DataFrame com todas as referências sem duplicatas
        """
        logger.info("Iniciando processamento de referências BibTeX e CSV...")

        try:
            # Processa arquivos BibTeX
            bibtex_refs = self.process_bibtex_files()

            # Processa arquivos CSV
            csv_refs = self.process_csv_files()

            # Converte CSV para ReferenceEntry para compatibilidade
            csv_as_refs = []
            for csv_ref in csv_refs:
                # Cria um dicionário BibTeX fake para ReferenceEntry
                fake_bibtex_entry = {
                    'ID': csv_ref.id,
                    'title': csv_ref.title,
                    'author': csv_ref.authors,
                    'year': csv_ref.year,
                    'journal': csv_ref.journal,
                    'doi': csv_ref.doi,
                    'abstract': csv_ref.abstract,
                    'issn': csv_ref.issn
                }
                ref_entry = ReferenceEntry(fake_bibtex_entry, csv_ref.database)
                csv_as_refs.append(ref_entry)

            # Combina tudo
            all_refs = bibtex_refs + csv_as_refs

            # --- SINGLE STAGE DE-DUPLICATION ---
            logger.info("=== PROCESSAMENTO DE DUPLICATAS (ETAPA ÚNICA) ===")
            
            # Converte todas as referências para DataFrame
            all_data = [ref.to_dict() for ref in all_refs if ref.is_valid()]
            if not all_data:
                logger.warning("Nenhuma referência válida encontrada!")
                return pd.DataFrame()
                
            df_complete = pd.DataFrame(all_data)
            logger.info(f"Total inicial de referências: {len(df_complete)}")

            # --- YEAR FILTER ---
            df_complete = self.year_filter.filter_dataframe(df_complete, MIN_YEAR)

            # Remove duplicatas
            df_final = self.duplicate_remover.remove_duplicates_dataframe(df_complete, keep='first')
            
            logger.info(f"=== RESULTADO FINAL ===")
            logger.info(f"Total final: {len(df_final)} referências únicas")

            # === DATASET A: ORGANIC (D_total) ===
            logger.info("=== GERANDO DATASET A: Organic Universe (D_total) ===")
            df_organic = df_final.copy()
            
            output_path_a = os.path.join(os.path.dirname(self._get_output_path()), 'dataset_organic.csv')
            print(f"DEBUG: Saving to {output_path_a}")
            logger.info(f"DEBUG PATH: {output_path_a}")
            df_organic.to_csv(output_path_a, index=False, encoding='utf-8')
            logger.info(f"Dataset A salvo em: {output_path_a} ({len(df_organic)} itens)")

            # === DATASET B: POLICY-ALIGNED (D_policy) ===
            logger.info("=== GERANDO DATASET B: Policy-Aligned (D_policy) ===")
            df_policy = self.keyword_filter.filter_dataframe_by_keywords(df_organic)
            
            output_path_b = os.path.join(os.path.dirname(self._get_output_path()), 'dataset_policy.csv')
            df_policy.to_csv(output_path_b, index=False, encoding='utf-8')
            logger.info(f"Dataset B salvo em: {output_path_b} ({len(df_policy)} itens)")
            
            # Gera estatísticas baseadas no Dataset B (Policy) ou A? O original fazia do final (que era o filtered).
            # Vou gerar estatísticas do dataset Organic para visão geral.
            stats = self.statistics.generate_statistics(df_organic)
            self.statistics.print_statistics(stats)

            # === DATASET C: HIGH-QUALITY (D_quality) ===
            logger.info("=== GERANDO DATASET C: High-Quality (D_quality) ===")
            df_quality = self.quality_filter.filter_by_quality_proxy(df_organic)
            
            output_path_c = os.path.join(os.path.dirname(self._get_output_path()), 'dataset_quality.csv')
            df_quality.to_csv(output_path_c, index=False, encoding='utf-8')
            logger.info(f"Dataset C salvo em: {output_path_c} ({len(df_quality)} itens)")

            # === COMPARATIVE METRICS ===
            metrics = {
                'quality_alignment': self.comparative_metrics.calculate_quality_alignment(df_organic, df_quality),

                'depth_of_adoption': self.comparative_metrics.calculate_depth_of_adoption(df_organic)
            }
            self.comparative_metrics.print_metrics(metrics)

            return df_organic

        except Exception as e:
            logger.error(f"Erro durante o processamento: {e}")
            raise


def main():
    """Função principal."""
    try:
        generator = DatabaseGenerator()
        df_result = generator.generate_database()
        print(f"\nProcessamento concluído! Total de {len(df_result)} referências únicas.")
        return df_result
    except Exception as e:
        print(f"Erro durante o processamento: {e}")
        return None


if __name__ == "__main__":
    main()
