# SurgiRAG: A Retrieval-Augmented Generation System for Surgical Question Answering

**SurgiRAG** is an open-source Retrieval-Augmented Generation (RAG) system designed to answer complex surgical questions about laparoscopic cholecystectomy (LC) procedures. The current version focuses on a robust **text-based pipeline**, with a **multimodal video+text extension** under active development.

---

## Key Features

- **Biomedical Text RAG Pipeline**: Leverages domain-specific retrieval and generation components.
- **Dense Retrieval with BioBERT**: Uses high-quality medical embeddings for relevant context retrieval.
- **Cross-Encoder Re-ranking**: BGE-based reranking enhances precision.
- **Text-Conditioned Generation**: Powered by an LLaMA-based vision-instruct model (used in text-only mode).
- **Comprehensive Evaluation**: Includes BLEU, ROUGE, BERTScore, FactCC, Entailment, and GPT-4 feedback.
- **Modular Design**: Built for easy extension to multimodal inputs.
---
  
## 📂 Project Structure
SurgiRAG/
├── retriever/ # Retrieval models, FAISS index, reranker
├── generator/ # LLM-based generation logic
├── evaluation/ # Scripts for evaluation and analysis
├── data/ # Surgical QA corpora
├── configs/ # Configuration files
├── scripts/ # Main pipeline and utilities
└── README.md



---

## 🧠 Text Branch Overview

### 🔍 Dense Retrieval

- **Retriever**: A fine-tuned `BioBERT`/`Sentence-BioBERT` model encodes both queries and contexts.
- **Indexing**: Contexts are embedded and indexed using FAISS with inner product similarity.
- **Re-ranking**: A `BGE` cross-encoder reranks top-k results for better relevance.

### 🧾 Generation

- **LLM Generator**: Uses a 11B LLaMA-based vision-instruct model for text generation.
- **Input**: `[Question] + [Top-k Contexts]` → `Answer`
- **Optimization**: Supports batched generation and inference on a single A100 GPU.

---

## 📊 Evaluation

- **Automatic Metrics**: BLEU, ROUGE-L, BERTScore
- **Factual Consistency**: FactCC and Natural Language Inference (Entailment)
- **Per-Query Reports**: Output includes granular logs for error analysis
- **LLM Review**: GPT-4-based scoring of helpfulness, accuracy, and hallucinations
---

## 🔭 Roadmap: Multimodal Expansion

The next version of SurgiRAG will incorporate **surgical video understanding** for enhanced multimodal reasoning.

### Planned Features
- **Video Retrieval**: Use CLIP, VideoMAE, or Video-LLaMA to encode laparoscopic videos
- **Joint Context Fusion**: Combine top-k visual and text contexts for generation
- **Multimodal LLMs**: Add support for visual + text conditioned generation
- **Extended Evaluation**: Human and benchmark evaluation for multimodal QA

---


