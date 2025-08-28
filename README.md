# DATASCI 266 - NLP Final Report (Summer 2025)
# SurgiRAG: A Domain-Adaptive Retrieval-Augmented Generation Framework for Procedural Text Understanding in Surgery

This repository contains all notebooks, scripts, and data preprocessing steps for our final project on retrieval-augmented generation (RAG) applied to surgical procedural understanding. The final model integrates domain-specific retrieval, reranking, and generation to answer medical questions grounded in procedural video transcripts and surgical literature.

**Final Notebook:**  
🔹 **`09_Full_RAG_w_ablation.ipynb`** – Our complete end-to-end RAG pipeline with full evaluation and ablation studies. This is the definitive notebook consolidating all prior development stages into a reproducible and modular framework.

---

## Repository Structure

### 📁 Folders
- **`data/`** – Contains all data used in the final notebook, including processed surgical documents and transcripts.
  - Core project data used in model training and evaluation.
  - **`video_branch_helpers/`** – Contains helper notebooks for extracting, cleaning, and preprocessing surgical video content and transcripts (detailed below).
- **`literature/`** – Supporting literature for background research and domain understanding.

### 📓 Notebooks

| Notebook | Description |
|----------|-------------|
| `00_simple_rag_preprocessing.ipynb` | Basic preprocessing of surgical documents and question data for initial RAG setup. |
| `01_video_preprocessing.ipynb` | Converts surgical videos into transcript format using ASR, segments them, and prepares them for retrieval. |
| `02_rag_text_branch.ipynb` | Implements a simple text-only RAG pipeline. |
| `03_combined_text_branch_RAG.ipynb` | Merges multiple text sources and integrates utility functions for modularity. |
| `04_retriever_training.ipynb` | Trains a domain-adapted dense retriever using BioBERT embeddings. |
| `05_retriever_fine_tuning.ipynb` | Fine-tunes the retriever with domain-specific QA pairs (e.g., BioASQ). |
| `06_generator_fine_tuning.ipynb` | Fine-tunes the generator for medical QA generation using retrieved context. |
| `07_e2e_RAG_w_ablation.ipynb` | Early end-to-end pipeline with preliminary ablation results. |
| `08_gen_finetune_rag_optimized.ipynb` | Generator fine-tuning with optimized hyperparameters for fluency and factual grounding. |
| `09_Full_RAG_w_ablation.ipynb` | **Final model notebook**: Full RAG system with detailed ablation, final retrieval/generation setup, and automated evaluation. |
| `README.md` | You're here. |
| `utils.py` | Utility functions for preprocessing, chunking, retrieval, and evaluation used across notebooks. |

---

### Video Branch Helper Notebooks (`data/video_branch_helpers/`)

| Notebook | Description |
|----------|-------------|
| `video_pull.ipynb` | Downloads laparoscopic cholecystectomy surgical videos from YouTube. |
| `transcript_pull.ipynb` | Uses AssemblyAI to generate high-confidence transcripts from downloaded surgical videos. |
| `video_preprocessing.ipynb` | Segments and timestamps transcripts into structured chunks, aligning them with procedural phases. |
| `07_e2e_RAG_w_ablation_video_sooyeon.ipynb` | Alternative end-to-end pipeline for multimodal data testing using video-derived data. |

---


## Key Features

- **BioBERT-powered Retriever** trained on domain-specific QA pairs
- **Cross-encoder Reranker** for high-precision evidence selection
- **Generator Fine-tuning** using surgical QA datasets
- **Comprehensive Evaluation** with BLEU, ROUGE, BERTScore, FactCC, Entailment, and GPT-4-based metrics
- **Token-aware chunking**, temporal grounding, and metadata-rich preprocessing

---

## How to Reproduce

1. Clone this repository.
2. Follow the preprocessing steps in `00_` and `02_` notebooks.
3. Train retriever ( `05_`), then generator (`06_`).
4. Run the full pipeline in `09_Full_RAG_w_ablation.ipynb`.

---

