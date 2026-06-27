# TraitAlign: Behaviorally Grounded Auxiliary Objectives for Recommendation

This repository contains the core codebase for the **TraitAlign** framework—a Multi-Task Learning approach for recommendation systems that uses behaviorally grounded auxiliary objectives.

## 📂 Codebase Overview

There are two primary files in this repository intended for execution on **Kaggle**:

### 1. `kaggle_master_prep.py`
**Purpose**: This script is used to pre-process the raw datasets and construct the engineered behavioral proxies. It parses raw interaction logs, calculates behavioral proxies (like Nostalgia, Taste Diversity, Discovery Rate), extracts MiniLM embeddings for textual metadata, and packages the results into structured Zip files (`movielens_traitalign_data.zip` and `lastfm_traitalign_data.zip`).
**Datasets Required for this Script**:
- [MovieLens 1M Original](https://www.kaggle.com/datasets/odedgolden/movielens-1m-dataset)
- [LastFM 360K Original](https://www.kaggle.com/datasets/neferfufi/lastfm)
- [MovieLens 25M (for embedding references)](https://www.kaggle.com/datasets/garymk/movielens-25m-dataset)

### 2. `generate_embeddings.ipynb`
**Purpose**: This notebook is responsible for generating the dense 384-dimensional semantic embeddings for all the movies in the MovieLens dataset. It processes the text string concatenations of movie titles and their associated genres by passing them through the `all-MiniLM-L6-v2` SentenceTransformer neural network. The resulting matrix (`P_i.npy`) is uploaded to Kaggle and attached to the master trainer.
**Datasets Required for this Notebook**:
- [MovieLens 25M Dataset](https://www.kaggle.com/datasets/garymk/movielens-25m-dataset) (used to fetch the large-scale `movies.csv` file containing the title strings and genres)

### 3. `traitalign_kaggle_trainer.ipynb`
**Purpose**: This is the master training notebook. It implements the GNN encoders (GraphSAGE, GCN, GAT), the Trait-Behavior Alignment Head (MLP), and runs the unified multi-task optimization. It dynamically compresses features, concatenates demographics, and trains the model while evaluating NDCG and HR metrics.
**Datasets Required for this Notebook**:
- [TraitAlign Processed MovieLens 1M](https://www.kaggle.com/datasets/keshavshaurya/traitalign-movielens1m)
- [TraitAlign Processed LastFM](https://www.kaggle.com/datasets/kkaushik06/traitalign-lastfm)
- [Personality 2018 (Original)](https://www.kaggle.com/datasets/arslanali4343/top-personality-dataset)
- [MovieLens MiniLM Embeddings](https://www.kaggle.com/datasets/kkaushik06/movielens-minilm-embeddings)

---

## 🚀 How to Run on Kaggle

To reproduce the experiments, you must execute the codebase within a Kaggle Notebook environment.

1. **Create a New Kaggle Notebook**: Upload `traitalign_kaggle_trainer.ipynb`.
2. **Attach Required Datasets**: Use the "Add Data" button in the Kaggle UI on the right panel to attach the datasets listed for the trainer notebook.
3. **Enable GPU**: Go to Settings on the right panel and set the Accelerator to **GPU T4x2** or **P100**.
4. **Configure Paths**: Ensure the `DATA_DIR` path configurations at the top of the notebook correctly point to the mounted Kaggle dataset paths (e.g., `/kaggle/input/...`).
5. **Execute**: Click "Run All" to begin the end-to-end model training, validation, and evaluation pipeline.
