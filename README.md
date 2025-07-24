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
