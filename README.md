# TraitAlign: Behaviorally Grounded Auxiliary Objectives for Recommendation

This repository contains the core codebase for the **TraitAlign** framework—a Multi-Task Learning approach for recommendation systems that uses behaviorally grounded auxiliary objectives.

## 📂 Codebase Overview

There are two primary files in this repository intended for execution on **Kaggle**:

### 1. `kaggle_master_prep.py`
**Purpose**: This script is used to pre-process the raw datasets and construct the engineered behavioral proxies. It parses raw interaction logs, calculates behavioral proxies (like Nostalgia, Taste Diversity, Discovery Rate), extracts MiniLM embeddings for textual metadata, and packages the results into structured Zip files (`movielens_traitalign_data.zip` and `lastfm_traitalign_data.zip`).
**Datasets Required for this Script**:
- [MovieLens 1M Original](https://www.kaggle.com/datasets/odedgolden/movielens-1m-dataset)
- [LastFM 360K Original](https://www.kaggle.com/datasets/neferfufi/lastfm)

### 2. `generate_embeddings.ipynb`
**Purpose**: This notebook is responsible for generating the dense 384-dimensional semantic embeddings for all the movies in the MovieLens dataset. It processes the text string concatenations of movie titles and their associated genres by passing them through the `all-MiniLM-L6-v2` SentenceTransformer neural network. The resulting matrix (`P_i.npy`) is uploaded to Kaggle and attached to the master trainer.
**Datasets Required for this Notebook**:
- [MovieLens 25M Dataset](https://www.kaggle.com/datasets/garymk/movielens-25m-dataset) (Provides the `movies.csv` file, which contains the raw titles and genres that are passed into the `all-MiniLM-L6-v2` tokenizer to build the semantic strings.)

### 3. `traitalign_kaggle_trainer.ipynb`
**Purpose**: This is the master training notebook. It implements the GNN encoders (GraphSAGE, GCN, GAT), the Trait-Behavior Alignment Head (MLP), and runs the unified multi-task optimization. It dynamically compresses features, concatenates demographics, and trains the model while evaluating NDCG and HR metrics.
**Datasets Required for this Notebook**:
- [TraitAlign Processed MovieLens 1M](https://www.kaggle.com/datasets/keshavshaurya/traitalign-movielens1m) (Provides the processed interactions, MiniLM occupation embeddings, and the engineered behavioral proxies to train the MLP alignment head).
- [TraitAlign Processed LastFM](https://www.kaggle.com/datasets/kkaushik06/traitalign-lastfm) (Provides the processed play interactions, SVD country embeddings, and behavioral proxies for the LastFM evaluation).
- [Personality 2018 (Original)](https://www.kaggle.com/datasets/arslanali4343/top-personality-dataset) (Provides the raw user-item interactions and the 5-dimensional numerical Big Five (OCEAN) traits used directly as auxiliary features).
- [MovieLens MiniLM Embeddings](https://www.kaggle.com/datasets/kkaushik06/movielens-minilm-embeddings) (Provides the `P_i.npy` file, which contains the 384-dimensional dense semantic representations for all MovieLens items. These are loaded directly into the GNN as the initial node feature matrix).

---

## 🚀 How to Run on Kaggle

To reproduce the experiments, you must execute the codebase within a Kaggle Notebook environment.

1. **Create a New Kaggle Notebook**: Upload `traitalign_kaggle_trainer.ipynb`.
2. **Attach Required Datasets**: Use the "Add Data" button in the Kaggle UI on the right panel to attach the datasets listed for the trainer notebook.
3. **Enable GPU**: Go to Settings on the right panel and set the Accelerator to **GPU T4x2** or **P100**.
4. **Configure Paths**: Ensure the `DATA_DIR` path configurations at the top of the notebook correctly point to the mounted Kaggle dataset paths (e.g., `/kaggle/input/...`).
5. **Execute**: Click "Run All" to begin the end-to-end model training, validation, and evaluation pipeline.
