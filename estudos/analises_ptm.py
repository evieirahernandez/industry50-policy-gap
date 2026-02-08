#!/usr/bin/env python3
"""
Análise de Tópicos usando PTM (Pseudo-document based Topic Model)

Este módulo implementa análise de tópicos usando o algoritmo PTM através da biblioteca
tomotopy. O PTM é especialmente eficaz para descobrir tópicos latentes em documentos
usando pseudo-documentos para melhor representação dos temas.

Author: Sistema de Análise de Tópicos
Date: 2025-08-31
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from collections import Counter
import time
import importlib.util

# Adicionar o diretório principal ao path para imports
script_dir = os.path.dirname(os.path.abspath(__file__))
projeto_root = os.path.dirname(script_dir)

# Importar módulos utilitários de forma robusta
text_utils_path = os.path.join(projeto_root, 'utilitarios', 'text_processing', 'text_utils.py')
spec = importlib.util.spec_from_file_location("text_utils", text_utils_path)
if spec and spec.loader:
    text_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(text_utils)
else:
    print("ERRO: Não foi possível carregar text_utils")
    sys.exit(1)

# Aliases para facilitar uso
DataLoader = text_utils.DataLoader
TextPreprocessor = text_utils.TextPreprocessor
TopicAnalysisUtils = text_utils.TopicAnalysisUtils


try:
    import tomotopy as tp
except ImportError:
    print("ERRO: tomotopy não está instalado. Execute: pip install tomotopy")
    sys.exit(1)


# Configuração dos datasets
DATASETS = {
    'organic': 'dataset_organic.csv',
    'policy': 'dataset_policy.csv',
    'quality': 'dataset_quality.csv',
    'default': 'referencias_com_topicos_ptm.csv' # Mantendo compatibilidade
}



def configurar_ambiente():
    """Configura o ambiente para análise PTM."""
    print("Configurando ambiente para análise PTM...")
    print(f"Versão do tomotopy: {tp.__version__}")


class PTMTopicAnalyzer:
    """
    Analisador de tópicos usando PTM (Pseudo-document based Topic Model).
    """

    def __init__(self, n_topicos: int = 10, n_palavras: int = 10, alpha: float = 1.0  # Seção 3.2 - Modelagem de Topicos em Textos Curtos: uma Avaliacão Experimental
                 , eta: float = 0.01  # Seção 3.2 - Modelagem de Topicos em Textos Curtos: uma Avaliacão Experimental
                 , p_pseudo_docs: Optional[int] = None, max_iter: int = 1000):
        """
        Inicializa o analisador PTM.

        Args:
            n_topicos: Número de tópicos a descobrir
            n_palavras: Número de palavras por tópico
            alpha: Parâmetro alpha (documento-tópico)
            eta: Parâmetro eta (tópico-palavra)
            p_pseudo_docs: Número de pseudo-documentos (padrão: 10 * n_topicos)
            max_iter: Número máximo de iterações
        """
        self.n_topicos = n_topicos
        self.n_palavras = n_palavras
        self.alpha = alpha
        self.eta = eta
        self.p_pseudo_docs = p_pseudo_docs or (10 * n_topicos)
        self.max_iter = max_iter
        self.auto_label = False  # Pode ser setado após instanciação
        self.titulos_fonte = []  # Pode ser setado após instanciação

        self.modelo = None
        self.documentos_processados = None
        self.rotulos_altes = {}

    def extrair_topicos_ptm(self, textos: List[str]) -> Tuple[List[Dict], Any]:
        """
        Extrai tópicos usando PTM.

        Args:
            textos: Lista de textos para análise

        Returns:
            Tupla com tópicos e modelo treinado
        """
        print(f"\n3. Aplicando PTM com {self.n_topicos} tópicos...")

        # Criar modelo PTM
        self.modelo = tp.PTModel(
            k=self.n_topicos,
            alpha=self.alpha,
            eta=self.eta,
            p=self.p_pseudo_docs,
            seed=42
        )

        print(f"Parâmetros: α={self.alpha}, η={self.eta}, pseudo-docs={self.p_pseudo_docs}")

        # Preprocessar documentos para tomotopy
        print("Preprocessando documentos...")
        documentos_tokenizados = []

        for i, texto in enumerate(textos):
            if i % 500 == 0:
                print(f"Processando documento {i+1}/{len(textos)}")

            # Tokenizar o texto (assumindo que já está preprocessado)
            tokens = texto.split() if isinstance(texto, str) else texto
            # Filtrar tokens válidos
            tokens_validos = [token for token in tokens if len(token) >= 2 and token.isalpha()]

            if tokens_validos:
                documentos_tokenizados.append(tokens_validos)

        self.documentos_processados = documentos_tokenizados
        print(f"Documentos processados: {len(documentos_tokenizados)}")

        # Adicionar documentos ao modelo
        print("Adicionando documentos ao modelo...")
        for doc_tokens in documentos_tokenizados:
            self.modelo.add_doc(doc_tokens)

        print(f"Vocabulário: {len(self.modelo.vocabs)} palavras únicas")

        # Treinar modelo
        print(f"Treinando modelo por {self.max_iter} iterações...")
        inicio = time.time()

        # Treinar em lotes para mostrar progresso
        batch_size = max(50, self.max_iter // 10)
        for i in range(0, self.max_iter, batch_size):
            iterations = min(batch_size, self.max_iter - i)
            self.modelo.train(iterations)

            if i % (batch_size * 2) == 0:
                ll = self.modelo.ll_per_word
                print(f"Iteração {i + iterations}: Log-likelihood = {ll:.4f}")

        tempo_total = time.time() - inicio
        print(f"Treinamento concluído em {tempo_total:.2f} segundos")
        print(f"Log-likelihood final: {self.modelo.ll_per_word:.4f}")

        # Extrair tópicos
        print("Extraindo tópicos...")
        topicos = []

        for topico_idx in range(self.n_topicos):
            # Obter palavras do tópico
            topic_words = self.modelo.get_topic_words(topico_idx, top_n=self.n_palavras)

            palavras = [word for word, prob in topic_words]
            pesos = [prob for word, prob in topic_words]

            topicos.append({
                'topico': topico_idx + 1,  # Começar do 1 para consistência
                'palavras': palavras,
                'pesos': pesos
            })

        return topicos, self.modelo

    def analisar_distribuicao_topicos(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Analisa a distribuição dos tópicos nos documentos.

        Args:
            df: DataFrame original

        Returns:
            Tupla com DataFrame atualizado e probabilidades dos tópicos
        """
        print("\n4. Analisando distribuição dos tópicos...")

        if not self.modelo:
            raise ValueError("Modelo deve ser treinado primeiro")

        # Obter distribuições de tópicos para cada documento
        doc_topic_probs = []
        topicos_dominantes = []
        probabilidades_dominantes = []

        for i in range(len(self.modelo.docs)):
            # Obter distribuição de tópicos para o documento
            topic_dist = self.modelo.docs[i].get_topic_dist()
            doc_topic_probs.append(topic_dist)

            # Encontrar tópico dominante
            topico_dominante = np.argmax(topic_dist)
            prob_dominante = topic_dist[topico_dominante]

            topicos_dominantes.append(topico_dominante + 1)  # +1 para consistência
            probabilidades_dominantes.append(prob_dominante)

        # Converter para array numpy
        doc_topic_probs_array = np.array(doc_topic_probs)

        # Adicionar ao dataframe
        df_analise = df.copy()

        # Ajustar tamanho se necessário (alguns documentos podem ter sido filtrados)
        n_docs_modelo = len(topicos_dominantes)
        if len(df_analise) > n_docs_modelo:
            df_analise = df_analise.iloc[:n_docs_modelo].copy()

        df_analise['topico_dominante'] = topicos_dominantes
        df_analise['probabilidade_topico'] = probabilidades_dominantes

        # Estatísticas da distribuição
        distribuicao = Counter(topicos_dominantes)
        print("\nDistribuição dos tópicos:")
        for topico, count in sorted(distribuicao.items()):
            print(f"  Tópico {topico}: {count} documentos ({count/len(df_analise)*100:.1f}%)")

        return df_analise, doc_topic_probs_array

    @staticmethod
    def exibir_topicos(topicos: List[Dict]) -> None:
        """
        Exibe os tópicos encontrados.

        Args:
            topicos: Lista de dicionários com informações dos tópicos
        """
        print("\n" + "="*60)
        print("TÓPICOS PRINCIPAIS IDENTIFICADOS - PTM")
        print("="*60)

        for topico in topicos:
            print(f"\nTÓPICO {topico['topico']}:")
            palavras_com_peso = zip(topico['palavras'], topico['pesos'])
            for palavra, peso in palavras_com_peso:
                print(f"  • {palavra} ({peso:.3f})")

    def rotular_topicos_altes(self, topicos: List[Dict]) -> Dict:
        """
        Aplica rotulação automática ALTES aos tópicos extraídos.

        Args:
            topicos: Lista de tópicos no formato {'topico', 'palavras', 'pesos'}

        Returns:
            Dict com rótulos por tópico
        """
        if not self.titulos_fonte:
            print("Aviso: Nenhum título de fonte fornecido para ALTES. Rotulação ignorada.")
            return {}

        try:
            from utilitarios.altes import ALTES

            print("\n5. Aplicando rotulação automática ALTES...")

            topics_altes = [
                {
                    'topic_id': t['topico'],
                    'words': t['palavras'],
                    'weights': dict(zip(t['palavras'], [float(p) for p in t['pesos']]))
                }
                for t in topicos
            ]

            altes = ALTES(titles=self.titulos_fonte)
            resultado = altes.label_all_topics(topics_altes, verbose=True)

            self.rotulos_altes = resultado['labels']

            for t in topicos:
                tid = t['topico']
                if tid in resultado['labels']:
                    t['rotulo_altes'] = resultado['labels'][tid]
                    t['score_altes'] = resultado['scores'][tid]

            print("\nRótulos ALTES gerados:")
            for tid, label in resultado['labels'].items():
                score = resultado['scores'][tid]
                print(f"  Tópico {tid}: {label} (score={score:.3f})")

            return resultado

        except ImportError:
            print("Aviso: Módulo ALTES não disponível. Instale com: pip install compress-fasttext")
            return {}
        except Exception as e:
            print(f"Erro na rotulação ALTES: {str(e)}")
            return {}


def executar_analise_ptm(dataset_name: str, dataset_filename: str, auto_label: bool = False):
    """
    Executa a análise PTM para um dataset específico.
    
    Args:
        dataset_name: Nome identificador do dataset (organic, policy, quality)
        dataset_filename: Nome do arquivo CSV (ex: dataset_organic.csv)
        auto_label: Se True, aplica rotulação ALTES
    """
    configurar_ambiente()
    print(f"\nANÁLISE DE TÓPICOS PRINCIPAIS - PTM [{dataset_name.upper()}]")
    print("="*50)
    if auto_label:
        print("Rotulação automática ALTES: ATIVADA")

    # Resolver caminhos
    script_dir = os.path.dirname(os.path.abspath(__file__))
    projeto_root = os.path.dirname(script_dir)
    dados_dir = os.path.join(projeto_root, 'dados')
    
    caminho_arquivo_entrada = os.path.join(dados_dir, dataset_filename)
    
    # Se for o default (referencias_sem_duplicatas), o DataLoader já sabe achar se passarmos None/default
    # Mas aqui vamos ser explícitos se possível, ou deixar o DataLoader se virar se o arquivo não existir na raiz de dados
    # No caso dos datasets novos, eles estão direto em dados/
    
    if dataset_name == 'default' and dataset_filename == 'referencias_com_topicos_ptm.csv':
        # Default behavior anterior: carregava referencias_sem_duplicatas.csv
        caminho_arquivo_entrada = None 
    
    # Carregar e preprocessar dados
    data_loader = DataLoader()
    try:
        df = data_loader.carregar_dados(caminho_arquivo_entrada)
    except FileNotFoundError:
        print(f"Arquivo não encontrado: {caminho_arquivo_entrada}")
        return None, None

    df_processado = data_loader.preprocessar_dataframe(df)

    # Preprocessar textos
    preprocessor = TextPreprocessor()
    textos_processados = preprocessor.preprocessar_textos(
        df_processado['texto_completo'].tolist(),
        usar_spacy=True
    )

    # Filtrar textos válidos
    textos_nao_vazios, df_filtrado = TopicAnalysisUtils.filtrar_textos_validos(
        textos_processados, df_processado
    )

    # Extrair títulos para ALTES
    # Tenta 'Título' ou 'titulo'
    coluna_titulo = 'Título' if 'Título' in df_filtrado.columns else 'titulo'
    titulos = df_filtrado[coluna_titulo].dropna().tolist() if coluna_titulo in df_filtrado.columns else []

    # Analisar tópicos com PTM
    ptm_analyzer = PTMTopicAnalyzer(n_topicos=3, n_palavras=10, max_iter=100, p_pseudo_docs=1000)
    ptm_analyzer.auto_label = auto_label
    ptm_analyzer.titulos_fonte = titulos
    topicos, ptm_model = ptm_analyzer.extrair_topicos_ptm(textos_nao_vazios)

    # Exibir resultados
    PTMTopicAnalyzer.exibir_topicos(topicos)

    # Aplicar rotulação automática ALTES se solicitado
    if auto_label and topicos:
        ptm_analyzer.rotular_topicos_altes(topicos)

    # Analisar distribuição
    df_com_topicos, doc_topic_probs = ptm_analyzer.analisar_distribuicao_topicos(df_filtrado)

    # Salvar resultados
    # Definir nome do arquivo de saída
    if dataset_name == 'default':
        nome_saida = 'referencias_com_topicos_ptm.csv'
    else:
        nome_saida = f'referencias_com_topicos_ptm_{dataset_name}.csv'
        
    caminho_saida = os.path.join(dados_dir, nome_saida)

    TopicAnalysisUtils.salvar_resultados(
        df_filtrado,
        df_com_topicos['topico_dominante'].tolist(),
        df_com_topicos['probabilidade_topico'].tolist(),
        caminho_saida
    )

    print(f"\nResultados salvos em: {caminho_saida}")
    print(f"Análise PTM [{dataset_name}] concluída com sucesso!")

    return df_com_topicos, topicos


def main(auto_label: bool = False, dataset_arg: str = 'default'):
    """Função principal para executar toda a análise PTM"""
    
    if dataset_arg == 'all':
        datasets_to_run = ['organic', 'policy', 'quality']
        resultados = {}
        for ds in datasets_to_run:
            print(f"\n\n>>> INICIANDO PROCESSAMENTO DO DATASET: {ds.upper()} <<<")
            df, topics = executar_analise_ptm(ds, DATASETS[ds], auto_label)
            resultados[ds] = (df, topics)
        return resultados, None # Retorna dict de resultados
    else:
        # Executar apenas um dataset
        if dataset_arg in DATASETS:
            filename = DATASETS[dataset_arg]
            # Se for default, a lógica interna trata de pegar o arquivo padrão
            return executar_analise_ptm(dataset_arg, filename, auto_label)
        else:
            print(f"Dataset desconhecido: {dataset_arg}")
            print(f"Opções válidas: {list(DATASETS.keys())} ou 'all'")
            sys.exit(1)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Análise de Tópicos usando PTM")
    parser.add_argument("--auto-label", action="store_true",
                        help="Aplica rotulação automática ALTES após extração")
    parser.add_argument("--dataset", type=str, default="default",
                        choices=["organic", "policy", "quality", "all", "default"],
                        help="Dataset a ser processado (default, organic, policy, quality, all)")
    args = parser.parse_args()
    
    # Executar
    main(auto_label=args.auto_label, dataset_arg=args.dataset)

