# Industry 5.0 Policy Gap Analysis

## Overview
This project analyzes the gap between academic research ("Organic Universe") and policy-aligned themes ("Policy-Aligned Universe") in the context of Industry 5.0. It leverages Natural Language Processing (NLP) techniques, specifically BERTopic and PTM (Pseudo-document based Topic Model), to identify and compare research topics.

Part of the paper "The Industry 5.0 Gap: Contrasting EU Policy Pillars with Global Research Trends using Automated Topic Modeling" 
## Project Structure
- **dados/**: Contains raw and processed data, including BibTeX/CSV references and generated datasets.
  - `dataset_organic.csv`: The complete "Organic" dataset of academic papers.
  - `dataset_policy.csv`: Papers aligned with specific policy keywords.
  - `dataset_quality.csv`: High-quality papers (e.g., Q1/Q2 journals).
- **estudos/**: Contains the main analysis scripts.
  - `analises_bertopic.py`: Topic modeling using BERTopic.
  - `analises_ptm.py`: Topic modeling using PTM (Tomotopy).
- **utilitarios/**: Helper scripts for data processing.
  - `database_generator.py`: Generates the consolidated database from raw BibTeX/CSV files.
  - `springer_abstracts.py`: Scraper for fetching missing abstracts from Springer Nature.
  - `altes/`: Implementation of the ALTES (Automatic Labeling of Topics with External Sources) method.
  - `text_processing/`: Text preprocessing utilities.

## Prerequisites
- Python 3.8+
- Chrome WebDriver (if using the Springer scraper)

## Installation

1. **Clone the repository** (if you haven't already).

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Language Models**:
   The project requires the spaCy English model.
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Usage

### 1. Database Generation
Consolidate raw references into the analysis datasets:
```bash
python utilitarios/database_generator.py
```
This script processes files in `dados/referencias_bibtex` and `dados/referencias_csv`, removes duplicates, and generates the three main datasets (`organic`, `policy`, `quality`) in the `dados/` folder.

### 2. Topic Modeling (BERTopic)
Run BERTopic analysis on a specific dataset:
```bash
python estudos/analises_bertopic.py --dataset organic
```
**Common Options:**
- `--dataset`: Choose `organic`, `policy`, `quality`, `default` (processed file), or `all`.
- `--nr-topics`: Number of topics to reduce to (default: 3).
- `--cluster-method`: Clustering algorithm (`hdbscan`, `kmeans`, `agglomerative`).
- `--auto-label`: Enable ALTES automatic labeling.
- `--high-quality`: Use a heavier, higher-quality embedding model (requires GPU recommended).

### 3. Topic Modeling (PTM)
Run PTM (Pseudo-document based Topic Model) analysis:
```bash
python estudos/analises_ptm.py --dataset organic
```
**Options:**
- `--dataset`: `organic`, `policy`, `quality`, or `all`.
- `--auto-label`: Enable ALTES automatic labeling.

### 4. Fetching Missing Abstracts
If you have a CSV file (e.g., from Springer Nature) with missing abstracts:
```bash
python utilitarios/springer_abstracts.py path/to/input.csv
```
This script uses Selenium to scrape abstract text from the article URLs.

## License
MIT
