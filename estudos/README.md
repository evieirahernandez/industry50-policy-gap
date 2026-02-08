# Scripts de Análise de Tópicos

Este diretório contém os scripts para análise de tópicos utilizando **PTM (Pseudo-document based Topic Model)** e **BERTopic**.

Ambos os scripts foram atualizados para suportar o processamento de datasets específicos (`organic`, `policy`, `quality`).

## Instalação das Dependências

Certifique-se de que o ambiente virtual está ativo e as dependências instaladas:

```bash
pip install -r ../requirements.txt
python -m spacy download en_core_web_sm
```

## 1. Análise PTM (`analises_ptm.py`)

Usa o algoritmo PTM (do pacote `tomotopy`) para modelagem de tópicos, ideal para textos curtos.

### Uso Básico

```bash
# Processar o dataset padrão
python analises_ptm.py

# Processar um dataset específico (organic, policy, ou quality)
python analises_ptm.py --dataset organic

# Processar TODOS os datasets sequencialmente
python analises_ptm.py --dataset all
```

### Argumentos Opcionais

- `--dataset [nome]`: Escolhe o dataset (`organic`, `policy`, `quality`, `all`, `default`).
- `--auto-label`: Ativa a rotulação automática dos tópicos usando ALTES (requer `compress-fasttext`).

---

## 2. Análise BERTopic (`analises_bertopic.py`)

Usa embeddings de transformers (BERT/RoBERTa) e clustering para descobrir tópicos.

### Uso Básico

```bash
# Processar dataset padrão com HDBSCAN (padrão)
python analises_bertopic.py

# Processar dataset de qualidade
python analises_bertopic.py --dataset quality

# Processar todos
python analises_bertopic.py --dataset all
```

### Argumentos Opcionais

- `--dataset [nome]`: Escolhe o dataset (`organic`, `policy`, `quality`, `all`, `default`).
- `--cluster-method [metodo]`: Escolhe o algoritmo de clustering.
  - `hdbscan` (padrão): Detecta densidade, bom para ignorar outliers.
  - `kmeans`: Força um número fixo de clusters.
  - `agglomerative`: Clustering hierárquico.
- `--nr-topics [n]`: Define o número de tópicos desejado (para K-Means/Agglomerative ou redução no BERTopic).
- `--high-quality`: Usa modelos `roberta-large` e `all-mpnet-base-v2` (mais lento, melhor qualidade).
- `--auto-label`: Ativa rotulação ALTES.

### Exemplos Avançados

```bash
# Rodar K-Means com 5 tópicos no dataset policy
python analises_bertopic.py --dataset policy --cluster-method kmeans --nr-topics 5

# Rodar modo Alta Qualidade no dataset organic
python analises_bertopic.py --dataset organic --high-quality
```

## Saída

Os resultados são salvos na pasta `../dados/` com nomes indicando o dataset e método:
- `referencias_com_topicos_ptm_organic.csv`
- `referencias_com_topicos_bertopic_quality_hdbscan.csv`
etc.
