"""
Análise de Tópicos usando BERTopic
Implementação moderna com embeddings de transformers
"""

import os
import sys
import pandas as pd
from typing import List, Dict, Tuple, Any

# Dependências do BERTopic
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering

import importlib.util
import argparse
import sys
import os

# Adicionar diretório raiz do projeto ao sys.path para garantir que imports funcionem
# quando o script é executado de subdiretórios
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root) # Prioridade para o root do projeto


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


class BERTopicAnalyzer:
    """Classe para análise de tópicos usando BERTopic"""

    def __init__(self, nr_topics: int = 3, min_topic_size: int = 10, cluster_method: str = "hdbscan",
                 auto_label: bool = False, titulos_fonte: List[str] = None, high_quality: bool = False):
        """
        Inicializa o analisador BERTopic

        Args:
            nr_topics: Número de tópicos desejado (None para automático)
            min_topic_size: Tamanho mínimo do tópico
            cluster_method: Método de clustering ('hdbscan', 'kmeans', 'agglomerative')
            auto_label: Se True, aplica rotulação automática ALTES após extração
            titulos_fonte: Lista de títulos para usar como fonte externa no ALTES
            high_quality: Se True, usa embeddings roberta-large + mpnet (setup do paper original)
        """
        self.nr_topics = nr_topics
        self.min_topic_size = min_topic_size
        self.cluster_method = cluster_method.lower()
        self.auto_label = auto_label
        self.titulos_fonte = titulos_fonte or []
        self.high_quality = high_quality
        self.topic_model = None
        self.embeddings = None
        self.rotulos_altes = {}  # Rótulos gerados pelo ALTES

    def _criar_modelo_clustering(self):
        """
        Cria o modelo de clustering baseado no método escolhido

        Returns:
            Modelo de clustering configurado
        """
        if self.cluster_method == "hdbscan":
            return HDBSCAN(
                min_cluster_size=self.min_topic_size,
                metric='euclidean',
                cluster_selection_method='eom',
                prediction_data=True
            )
        elif self.cluster_method == "kmeans":
            # Para K-means, usar o número de tópicos como número de clusters
            n_clusters = self.nr_topics if self.nr_topics else 5
            return KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=10
            )
        elif self.cluster_method == "agglomerative":
            # Para Agglomerative, usar o número de tópicos como número de clusters
            n_clusters = self.nr_topics if self.nr_topics else 5
            return AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage='ward'
            )
        else:
            raise ValueError(f"Método de clustering não suportado: {self.cluster_method}")

    def extrair_topicos_bertopic(self, textos: List[str]) -> Tuple[List[Dict], Any, List[int], List[float]]:
        """
        Extrai tópicos usando BERTopic

        Args:
            textos: Lista de textos processados

        Returns:
            Tupla com tópicos, modelo BERTopic, tópicos por documento e probabilidades
        """
        print(f"\n3. Extraindo tópicos usando BERTopic...")

        try:
            # Configurar componentes do pipeline

            # 1. Modelo de embeddings (sentence-transformers ou WordDocEmbedder)
            if self.high_quality:
                try:
                    print("--> MODO HIGH QUALITY ATIVADO (Setup Original ALTES)")
                    
                    # Imports sob demanda para garantir que existem
                    from flair.embeddings import TransformerWordEmbeddings
                    from bertopic.backend import WordDocEmbedder
                    
                    print("    Carregando 'roberta-large' (word) e 'all-mpnet-base-v2' (doc)...")
                    print("    Isso pode demorar e consumir bastante memória.")
                    
                    # Setup original do paper
                    embedding_word = TransformerWordEmbeddings('roberta-large')
                    embedding_document = SentenceTransformer('all-mpnet-base-v2')
                    embedding_model = WordDocEmbedder(
                        embedding_model=embedding_document, 
                        word_embedding_model=embedding_word
                    )
                except ImportError:
                    print("ERRO: Bibliotecas 'flair' ou 'transformers' não encontradas.")
                    print("Instale com: pip install flair transformers")
                    raise
                except Exception as e:
                    print(f"ERRO ao carregar modelos High Quality: {str(e)}")
                    # Fallback
                    print("Revertendo para modelo padrão 'all-MiniLM-L6-v2'")
                    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            else:
                # Default (mais leve)
                embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

            # 2. Redução de dimensionalidade (UMAP)
            umap_model = UMAP(
                n_neighbors=15,
                n_components=5,
                min_dist=0.0,
                metric='cosine',
                random_state=42
            )

            # 3. Clustering (método configurável)
            clustering_model = self._criar_modelo_clustering()
            print(f"Usando método de clustering: {self.cluster_method}")

            # 4. Vetorização para representação de tópicos
            vectorizer_model = CountVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                min_df=2
            )

            # Criar modelo BERTopic
            # Calcular probabilidades apenas para HDBSCAN
            calculate_probs = self.cluster_method == "hdbscan"

            self.topic_model = BERTopic(
                embedding_model=embedding_model,
                umap_model=umap_model,
                hdbscan_model=clustering_model,
                vectorizer_model=vectorizer_model,
                nr_topics=self.nr_topics,
                verbose=True,
                calculate_probabilities=calculate_probs
            )

            # Ajustar modelo e predizer tópicos
            print("Gerando embeddings e extraindo tópicos...")

            if self.high_quality:
                # WordDocEmbedder não suporta .encode() direto da mesma forma que SentenceTransformer
                # Deixar o BERTopic gerenciar os embeddings internamente
                self.embeddings = None
                topics, probabilities = self.topic_model.fit_transform(textos)
            else:
                # Gerar embeddings primeiro para salvar para visualizações (padrão)
                self.embeddings = embedding_model.encode(textos, show_progress_bar=True)
                # Treinar o modelo com embeddings pré-computados
                topics, probabilities = self.topic_model.fit_transform(textos, self.embeddings)

            # Obter informações dos tópicos
            topic_info = self.topic_model.get_topic_info()
            print(f"Número de tópicos encontrados: {len(topic_info)}")

            # Debug: mostrar informações dos tópicos encontrados
            print("Tópicos encontrados:")
            for i, row in topic_info.iterrows():
                print(f"  Tópico {row['Topic']}: {row['Count']} documentos")

            # Formatear tópicos para estrutura compatível
            topicos_formatados = []
            for i, row in topic_info.iterrows():
                if row['Topic'] != -1:  # Ignorar tópico de outliers
                    topic_words = self.topic_model.get_topic(row['Topic'])
                    if topic_words:
                        palavras = [word for word, _ in topic_words[:10]]
                        pesos = [weight for _, weight in topic_words[:10]]

                        topicos_formatados.append({
                            'topico': row['Topic'],  # Manter numeração original (0, 1, 2...)
                            'palavras': palavras,
                            'pesos': pesos,
                            'count': row['Count']
                        })

            return topicos_formatados, self.topic_model, topics, probabilities

        except Exception as e:
            print(f"Erro ao executar BERTopic: {str(e)}")
            return [], None, [], []

    def rotular_topicos_altes(self, topicos: List[Dict]) -> Dict:
        """
        Aplica rotulação automática ALTES aos tópicos extraídos.

        Args:
            topicos: Lista de tópicos no formato {'topico', 'palavras', 'pesos', 'count'}

        Returns:
            Dict com rótulos por tópico
        """
        if not self.titulos_fonte:
            print("Aviso: Nenhum título de fonte fornecido para ALTES. Rotulação ignorada.")
            return {}

        try:
            # Importar ALTES sob demanda
            from utilitarios.altes import ALTES

            print("\n5. Aplicando rotulação automática ALTES...")

            # Preparar tópicos no formato esperado pelo ALTES
            topics_altes = [
                {
                    'topic_id': t['topico'],
                    'words': t['palavras'],
                    'weights': dict(zip(t['palavras'], t['pesos']))
                }
                for t in topicos
            ]

            # Aplicar ALTES
            altes = ALTES(titles=self.titulos_fonte)
            resultado = altes.label_all_topics(topics_altes, verbose=True)

            self.rotulos_altes = resultado['labels']

            # Adicionar rótulos aos tópicos originais
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

        except ImportError as e:
            import traceback
            traceback.print_exc()
            print(f"DEBUG: Erro específico de importação: {e}")
            print("Aviso: Módulo ALTES não disponível ou dependência faltando.")
            return {}
        except Exception as e:
            print(f"Erro na rotulação ALTES: {str(e)}")
            return {}

    def analisar_distribuicao_topicos(self, topics: List[int], probabilities: List[float],
                                      df: pd.DataFrame) -> Tuple[pd.DataFrame, List[float]]:
        """
        Analisa a distribuição dos tópicos nos documentos

        Args:
            topics: Lista de tópicos por documento
            probabilities: Lista de probabilidades
            df: DataFrame original

        Returns:
            Tupla com DataFrame atualizado e probabilidades
        """
        print("\n4. Analisando distribuição dos tópicos...")

        # Manter numeração original do BERTopic (0, 1, 2... e -1 para outliers)
        topicos_ajustados = []
        probs_ajustadas = []

        # Tratar o caso quando probabilities é None (K-means, Agglomerative)
        if probabilities is None:
            probabilities = [1.0] * len(topics)  # Atribuir probabilidade padrão
            print("Probabilidades não disponíveis para este método de clustering. Usando valores padrão.")

        for i, (topic, prob) in enumerate(zip(topics, probabilities)):
            if topic == -1:  # Outlier
                topicos_ajustados.append(-1)  # Manter -1 para outliers
                probs_ajustadas.append(prob if prob is not None else 0.0)
            else:
                topicos_ajustados.append(topic)  # Manter numeração original (0, 1, 2...)
                probs_ajustadas.append(prob if prob is not None else 1.0)

        # Adicionar ao dataframe
        df_analise = df.copy()
        df_analise['topico_dominante'] = topicos_ajustados
        df_analise['probabilidade_topico'] = probs_ajustadas

        # Estatísticas da distribuição
        distribuicao = TopicAnalysisUtils.analisar_distribuicao_topicos(
            topicos_ajustados, probs_ajustadas
        )

        return df_analise, probs_ajustadas

    @staticmethod
    def exibir_topicos(topicos: List[Dict]) -> None:
        """
        Exibe os tópicos encontrados

        Args:
            topicos: Lista de dicionários com informações dos tópicos
        """
        print("\n" + "="*60)
        print(f"TÓPICOS PRINCIPAIS IDENTIFICADOS - BERTopic")
        print("="*60)

        for topico in topicos:
            print(f"\nTÓPICO {topico['topico']} (Documentos: {topico.get('count', 'N/A')}):")
            palavras_com_peso = zip(topico['palavras'], topico['pesos'])
            for palavra, peso in palavras_com_peso:
                print(f"  • {palavra} ({peso:.3f})")

    def visualizar_topicos(self, textos: List[str]) -> None:
        """
        Cria visualizações dos tópicos

        Args:
            textos: Lista de textos usados na análise
        """
        if self.topic_model is None:
            print("Modelo BERTopic não foi treinado ainda.")
            return

        try:
            # Verificar se há tópicos válidos para visualizar
            info_topicos = self.topic_model.get_topic_info()
            print(f"Tópicos disponíveis: {len(info_topicos)}")

            if len(info_topicos) <= 1:  # Apenas tópico -1 (outliers)
                print("Não há tópicos suficientes para criar visualizações.")
                return

            # Visualizar tópicos - apenas se temos tópicos válidos
            try:
                # Verificar se temos dados suficientes para visualização
                topics = info_topicos[info_topicos['Topic'] != -1]
                if len(topics) >= 2:
                    fig1 = self.topic_model.visualize_topics()
                    if fig1:
                        fig1.show()
                        print("Visualização de tópicos criada com sucesso.")
                    else:
                        print("Não foi possível criar visualização de tópicos.")
                else:
                    print("Número insuficiente de tópicos para visualização.")
            except Exception as e:
                print(f"Erro na visualização de tópicos: {str(e)}")

            # Visualizar documentos - apenas se temos embeddings e tópicos
            if hasattr(self, 'embeddings') and self.embeddings is not None:
                try:
                    # Usar apenas uma amostra pequena para evitar problemas de memória
                    sample_size = min(50, len(textos))  # Reduzir ainda mais
                    if sample_size >= 10:  # Mínimo de documentos
                        sample_texts = textos[:sample_size]
                        sample_embeddings = self.embeddings[:sample_size]

                        fig2 = self.topic_model.visualize_documents(
                            sample_texts,
                            embeddings=sample_embeddings
                        )
                        if fig2:
                            fig2.show()
                            print("Visualização de documentos criada com sucesso.")
                        else:
                            print("Não foi possível criar visualização de documentos.")
                    else:
                        print("Amostra muito pequena para visualização de documentos.")
                except Exception as e:
                    print(f"Erro na visualização de documentos: {str(e)}")

            # Heatmap de tópicos - apenas se temos múltiplos tópicos
            try:
                valid_topics = info_topicos[info_topicos['Topic'] != -1]
                if len(valid_topics) >= 2:
                    fig3 = self.topic_model.visualize_heatmap()
                    if fig3:
                        fig3.show()
                        print("Heatmap de tópicos criado com sucesso.")
                    else:
                        print("Não foi possível criar heatmap.")
                else:
                    print("Número insuficiente de tópicos para heatmap.")
            except Exception as e:
                print(f"Erro no heatmap: {str(e)}")

        except Exception as e:
            print(f"Erro geral ao criar visualizações: {str(e)}")


def main():
    """Função principal para executar toda a análise BERTopic"""
    # Parser de argumentos de linha de comando
    parser = argparse.ArgumentParser(description="Análise de Tópicos usando BERTopic")
    parser.add_argument(
        "--cluster-method",
        choices=["hdbscan", "kmeans", "agglomerative"],
        default="hdbscan",
        help="Método de clustering a ser usado (default: hdbscan)"
    )
    parser.add_argument(
        "--nr-topics",
        type=int,
        default=3,
        help="Número de tópicos desejado (default: 3)"
    )
    parser.add_argument(
        "--min-topic-size",
        type=int,
        default=10,
        help="Tamanho mínimo do tópico para HDBSCAN (default: 10)"
    )
    parser.add_argument(
        "--high-quality",
        action="store_true",
        help="Usa setup do paper original (roberta-large + all-mpnet-base-v2). Requer 'flair'."
    )
    parser.add_argument(
        "--export-mock",
        action="store_true",
        help="Se definido, imprime estruturas prontas para mockar no notebook (bertopic_topics, percentuais, list_word_topic)."
    )
    parser.add_argument(
        "--mock-topic",
        type=int,
        default=1,
        help="ID do tópico a exportar como list_word_topic (default: 1)"
    )
    parser.add_argument(
        "--auto-label",
        action="store_true",
        help="Se definido, aplica rotulação automática ALTES após extração dos tópicos"
    )

    args = parser.parse_args()

    print("ANÁLISE DE TÓPICOS PRINCIPAIS - BERTopic")
    print("="*50)
    print(f"Método de clustering: {args.cluster_method}")
    print(f"Número de tópicos: {args.nr_topics}")
    if args.cluster_method == "hdbscan":
        print(f"Tamanho mínimo do tópico: {args.min_topic_size}")
    if args.high_quality:
        print("Modo High Quality: ATIVADO")
    if args.auto_label:
        print("Rotulação automática ALTES: ATIVADA")

    # Carregar e preprocessar dados
    data_loader = DataLoader()
    df = data_loader.carregar_dados()
    df_processado = data_loader.preprocessar_dataframe(df)

    # Preprocessar textos (menos agressivo para BERTopic)
    preprocessor = TextPreprocessor()
    textos_processados = preprocessor.preprocessar_textos(
        df_processado['texto_completo'].tolist(),
        usar_spacy=False  # Usar processamento básico para BERTopic
    )

    # Filtrar textos válidos
    textos_nao_vazios, df_filtrado = TopicAnalysisUtils.filtrar_textos_validos(
        textos_processados, df_processado
    )

    # Extrair títulos para usar como fonte externa no ALTES
    # Importante: Usar df_filtrado para garantir que os títulos alinhem com textos_nao_vazios se necessário,
    # embora ALTES use os títulos independentemente dos tópicos gerados.
    # Corrigido para procurar 'Título' ou 'titulo'
    coluna_titulo = 'Título' if 'Título' in df_filtrado.columns else 'titulo'
    titulos = df_filtrado[coluna_titulo].dropna().tolist() if coluna_titulo in df_filtrado.columns else []
    
    if args.auto_label and not titulos:
        print(f"AVISO: Coluna de títulos ('{coluna_titulo}') não encontrada ou vazia. ALTES não funcionará.")

    # Analisar tópicos com BERTopic usando os parâmetros configurados
    bertopic_analyzer = BERTopicAnalyzer(
        nr_topics=args.nr_topics,
        min_topic_size=args.min_topic_size,
        cluster_method=args.cluster_method,
        auto_label=args.auto_label,
        titulos_fonte=titulos,
        high_quality=args.high_quality
    )
    topicos, model, topics, probabilities = bertopic_analyzer.extrair_topicos_bertopic(textos_nao_vazios)

    if not topicos:
        print("Falha na análise BERTopic. Verifique as dependências.")
        return None, None

    # Exibir resultados
    BERTopicAnalyzer.exibir_topicos(topicos)

    # Aplicar rotulação automática ALTES se solicitado
    if args.auto_label and topicos:
        resultado_altes = bertopic_analyzer.rotular_topicos_altes(topicos)

    # Analisar distribuição
    df_com_topicos, probs = bertopic_analyzer.analisar_distribuicao_topicos(
        topics, probabilities, df_filtrado
    )

    # Salvar resultados com nome específico do método
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    dados_dir = os.path.join(project_root, 'dados')
    caminho_saida = os.path.join(dados_dir, f'referencias_com_topicos_bertopic_{args.cluster_method}.csv')

    TopicAnalysisUtils.salvar_resultados(
        df_filtrado,
        df_com_topicos['topico_dominante'].tolist(),
        df_com_topicos['probabilidade_topico'].tolist(),
        caminho_saida
    )

    # Exportação para mock no notebook, se solicitado
    if args.export_mock and topicos:
        try:
            print("\n# MOCK EXPORT START")
            # Lista completa de tópicos com palavras e pesos
            print("bertopic_topics = [")
            for t in topicos:
                # Empacotar palavras e pesos em tuplas (palavra, peso arredondado)
                word_weight_pairs = [(w, round(p, 3)) for w, p in zip(t['palavras'], t['pesos'])]
                print(f"    {{'topic': {t['topico']}, 'count': {t['count']}, 'words': {word_weight_pairs}}},")
            print("]")

            # Percentuais por tópico
            total_docs = sum(t['count'] for t in topicos if isinstance(t.get('count'), (int, float))) or 1
            print("percentuais = {")
            for t in topicos:
                pct = (t['count'] / total_docs) * 100 if total_docs else 0
                print(f"    {t['topico']}: {pct:.1f},")
            print("}")

            # Construir índice para acesso rápido ao tópico
            mapa_topicos = {t['topico']: t for t in topicos}
            chosen_id = args.mock_topic
            if chosen_id not in mapa_topicos:
                # Caso não exista, escolher o primeiro disponível
                chosen_id = topicos[0]['topico']
                print(f"# Aviso: Tópico {args.mock_topic} não encontrado. Usando {chosen_id}.")
            list_word_topic = mapa_topicos[chosen_id]['palavras']
            print(f"list_word_topic = {list_word_topic}")

            # Também exportar estrutura simples de pesos para wordcloud
            pesos_dict = {w: round(p, 3) for w, p in zip(mapa_topicos[chosen_id]['palavras'], mapa_topicos[chosen_id]['pesos'])}
            print(f"topic_words_with_weights = {pesos_dict}")
            print(f"topic_number = {chosen_id}")
            print("# MOCK EXPORT END\n")
        except Exception as e:
            print(f"Falha ao gerar exportação mock: {e}")

    # Criar visualizações (opcional)
    # try:
    #     print("\n6. Criando visualizações...")
    #     bertopic_analyzer.visualizar_topicos(textos_nao_vazios)
    # except Exception as e:
    #     print(f"Visualizações não puderam ser criadas: {str(e)}")

    return df_com_topicos, topicos


if __name__ == "__main__":
    df_resultados, topicos_encontrados = main()

"""
Exemplos de uso:

1. Usar HDBSCAN (padrão):
   python analises_bertopic.py

2. Usar K-means com 5 clusters:
   python analises_bertopic.py --cluster-method kmeans --nr-topics 5

3. Usar Agglomerative Clustering com 4 clusters:
   python analises_bertopic.py --cluster-method agglomerative --nr-topics 4

4. Usar HDBSCAN com tamanho mínimo personalizado:
   python analises_bertopic.py --cluster-method hdbscan --min-topic-size 15

5. Usar K-means com rotulação automática ALTES:
   python analises_bertopic.py --cluster-method kmeans --nr-topics 5 --auto-label
"""

