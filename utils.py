
from __future__ import annotations
import json
import math
import pathlib
import torch
import torch.nn.functional as F         
from typing import List, Dict, Tuple
import re
from dataclasses import dataclass
from typing import Generator, Iterable, List, Sequence, Tuple
import numpy as np
import fitz
import spacy
import html, unicodedata, logging
import torch
import tiktoken
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import faiss
import os, warnings, faiss, torch, numpy as np
from typing import List, Dict, Tuple
import urllib.parse



# ------ Preprocessing ---------
nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")

html_tag_re = re.compile(r"<[^>]+>")
ws_re       = re.compile(r"\s+")
doi_re      = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b")
bullet_re   = re.compile(r"[•●▪■◦◆▶►‣❖⚫⬤➤➣➢\u2022\u25CF\u25A0]")
dehyphen_re = re.compile(r"(\w+)-\s*\n\s*(\w+)")
nbsp_re     = re.compile(r"\u00A0+")
header_footer_re = re.compile(r"""
    ^(
        [•●▪■◦◆▶►‣❖⚫⬤➤➣➢]?\s*    
        (page|p)\s*\d+(\s*/\s*\d+)? |  
        [•●▪■◦◆▶►‣❖⚫⬤➤➣➢]?\s*
        figure\s+\d+               |  
        [•●▪■◦◆▶►‣❖⚫⬤➤➣➢]?\s*
        table\s+\d+                |   
        ©.+?$                      |
        (all\s+rights\s+reserved)  |
        this\s+page\s+intentionally\s+left\s+blank |
        confidential               |
        copyright                  |
        doi:\s*10\.\d{4,9}/\S+     |
        \s*$                           
    )
""", re.IGNORECASE | re.VERBOSE)


def infer_source(doc: fitz.Document, pdf_path: str | pathlib.Path) -> str:
    """
    Get source label for a PDF

    Priority:
    1. DOI in XMP or first page          → 'DOI:10.1145/1234567'
    2. Domain extracted from URL         → 'arxiv.org' / 'springer.com'
    3. Base filename                     → 'my_paper.pdf'
    """
    meta_text = " ".join(str(v) for v in doc.metadata.values() if v)
    page1_text = doc[0].get_text() if len(doc) else ""
    haystack = f"{meta_text} {page1_text}"

    if (m := doi_re.search(haystack)):
        return f"DOI:{m.group(0)}"

    # Look for HTTPS path
    url = str(pdf_path)
    if urllib.parse.urlparse(url).scheme in {"http", "https"}:
        domain = urllib.parse.urlparse(url).netloc
        if domain:
            return domain

    # fallback to filename
    return pathlib.Path(url).name


def open_and_read_pdfs(pdf_paths, *, start_doc_num=0, text_formatter=lambda x: x):
    pages = []
    doc_num = start_doc_num
    enc = tiktoken.get_encoding("cl100k_base")

    for path in pdf_paths:
        with fitz.open(path) as doc:
            doc_source = infer_source(doc, path)
            for page_num, page in enumerate(doc):
                txt = text_formatter(page.get_text())
                pages.append({
                    "doc_num": doc_num,
                    "doc_source": doc_source,  #Doc name
                    "page_number": page_num,
                    "page_char_count": len(txt),
                    "page_word_count": len(txt.split()),
                    "page_sentence_count_raw": len(txt.split(". ")),
                    "page_token_count": len(enc.encode(txt)),
                    "text": txt
                })
        doc_num += 1
    return pages

def _normalize_whitespace(txt: str) -> str:
    txt = nbsp_re.sub(" ", txt)
    txt = bullet_re.sub(" ", txt)
    txt = unicodedata.normalize("NFKC", txt)
    return " ".join(txt.split()).strip()


def strip_html(text: str) -> str:
    """Remove HTML/XML tags."""
    return html_tag_re.sub(" ", text)


def normalize_ws(text: str) -> str:
    """Collapse consecutive whitespace to single spaces & trim ends."""
    return ws_re.sub(" ", text).strip()


def clean_text(text: str) -> str:
    """Canonical text cleanup used across corpus / query processing."""
    text = strip_html(text)
    text = normalize_ws(text)
    text = _normalize_whitespace(text)
    return text

def split_sentences(text: str) -> List[str]:
    """ spaCy-based sentence splitter """
    text = dehyphen_re.sub(r"\1\2", text)         # undo line-break hyphenation
    text = html.unescape(text)                     # &amp; -> &
    text = _normalize_whitespace(text)
    if nlp:
        doc = nlp(text)
        return [s.text.strip() for s in doc.sents if s.text.strip()]
    else:
        crude = re.split(r"(?<=[.!?])\s{1,}", text)
        return [s.strip() for s in crude if s.strip()]

def clean_sentence(sentence: str) -> str:
    """
    Remove headers, footers, boilerplate, short artifacts,
    and normalize whitespace.  Returns '' if the sentence should be skipped.
    """
    sentence = _normalize_whitespace(sentence)
    
    # strip lines that are obviously boilerplate
    if header_footer_re.match(sentence):
        return ""
    
    # very short fragments
    if len(sentence) < 6:
        return ""
    
    return sentence


def clean_pages(pages: list[dict]) -> list[dict]:
    """
    For every page in *pages*:
      • split every raw text chunk into sentences
      • run clean_sentence() on each sentence
      • drop blanks ('')
    """
    cleaned_pages = []

    for page in pages:
        raw_chunks = page["text"]        
        cleaned_sents: list[str] = []

        for chunk in raw_chunks:
            for sent in split_sentences(chunk):    
                sent = clean_sentence(sent)
                if sent:                        
                    cleaned_sents.append(sent)

       
        cleaned_page = {**page, "text": cleaned_sents}
        cleaned_pages.append(cleaned_page)

    return cleaned_pages
    

#----- Chunking ------

def token_chunks(
    sentences: Iterable[str],
    chunk_size: int = 300,
    overlap: int = 80,
    slack: int = 20,                    # how much we tolerate going over / under
    encoding_name: str = "cl100k_base"
) -> list[str]:
    """
    Build chunks of `chunk_size` tokens without splitting sentences.
    The chunk boundary is shifted so that the last sentence ends within
    ±`slack` tokens of the target size.  Keeps `overlap` tokens between chunks.
    """
    # tokenizer
    enc = tiktoken.get_encoding(encoding_name)


    # if no tokenizer listsed
    if enc is None:
        words  = " ".join(sentences).split()
        stride = chunk_size - overlap
        return [
            " ".join(words[i : i + chunk_size])
            for i in range(0, len(words), stride)
        ]


    chunks, cur_tokens = [], []
    cur_len = 0

    sentence_tokens = [enc.encode(s) for s in sentences]

    for sent_tok in sentence_tokens:
        sent_len = len(sent_tok)
        # If adding this sentence would exceed chunk_size by > slack -> close chunk
        if cur_len and cur_len + sent_len > chunk_size + slack:
            # flush current chunk
            chunks.append(enc.decode(sum(cur_tokens, [])))
            # seed next chunk with last `overlap` tokens for context
            overlap_toks = cur_tokens[-1] if cur_tokens else []
            if overlap:                       # take tail of previous chunk
                overlap_toks = sum(cur_tokens, [])[-overlap:]
            cur_tokens = [overlap_toks] if overlap_toks else []
            cur_len    = len(overlap_toks)
        # add sentence to current chunk
        cur_tokens.append(sent_tok)
        cur_len += sent_len

    # flush any remaining tokens
    if cur_tokens:
        chunks.append(enc.decode(sum(cur_tokens, [])))

    return chunks



def ensure_metadata_container(d):
    """
    Return a writeable metadata dict no matter whether *d* is a LangChain
    Document (with .metadata) or a plain dict.
    """
    if hasattr(d, "metadata"):           # LangChain Document
        return d.metadata
    if isinstance(d, dict):              # plain dict -> create key if missing
        return d.setdefault("metadata", {})
    raise TypeError("Unsupported object type: %s" % type(d))



# -------------------------------------------------------
# ---------------- Get final preprocessed pages --------------------

def get_cleaned_and_processed_pages(pdf_paths):
    pages = open_and_read_pdfs(pdf_paths)
    for page in pages:
            page['text'] = clean_text(page['text'])
            page['text'] = split_sentences(page['text'])
    pages = clean_pages(pages)
    
    processed_pages = []
    
    for page in pages:
        raw_text = (
            page["text"]
            if isinstance(page, dict)
            else getattr(page, "page_content", "")
        )
        if isinstance(raw_text, list):               # flatten list into single str
            raw_text = " ".join(raw_text)

        # sentence split -> clean -> token‑aligned chunks
        sentences = [
            s for s in map(clean_sentence, split_sentences(raw_text)) if s
        ]
        chunks = token_chunks(sentences, chunk_size=300, overlap=80)

        # 3) store results in metadata container
        meta = ensure_metadata_container(page)
        meta["sentences"]        = sentences
        meta["token_chunks"]     = chunks
        meta["num_token_chunks"] = len(chunks)

        processed_pages.append(page)
        
    return processed_pages









# ------ Embedding Corpus ------

def embed_text(
    processed_pages: List[Dict],
    embedding_model,
    *,
    batch_size: int = 64,
) -> Tuple[torch.Tensor, List[str], List[Dict]]:
    """
    Flatten every ``token_chunks`` list inside ``processed_pages`` and produce an
    L2-normalized embedding matrix.

    Parameters:
    processed_pages : list[dict]
        Each dict must contain a ``metadata`` key with ``"token_chunks"`` plus
        top-level keys ``doc_num``, ``page_number`` and ``doc_source``.
    embedding_model : SentenceTransformer
        Any sentence-transformers model that exposes ``encode``.
    batch_size : int, optional
        How many chunks to encode per forward pass.

    Returns:
    embeddings : torch.Tensor  # shape (N_chunks, dim)
    text_chunks : list[str]    # same order as rows in ``embeddings``
    chunk_meta  : list[dict]   # per-chunk provenance (doc/page/chunk ids, source)
    """

    #flatten
    text_chunks, chunk_meta = [], []
    for page in processed_pages:
        for i, chunk in enumerate(page["metadata"].get("token_chunks", [])):
            text_chunks.append(chunk)
            chunk_meta.append(
                {
                    "doc_num":     page["doc_num"],
                    "page_number": page["page_number"],
                    "chunk_id":    i,
                    "source":      page["doc_source"],
                }
            )

    print(f"Total chunks: {len(text_chunks):,}")

    #batched embedding
    emb_batches: List[torch.Tensor] = []
    for start in range(0, len(text_chunks), batch_size):
        batch_txts = text_chunks[start : start + batch_size]

        # encode without implicit normalisation
        vecs = embedding_model.encode(
            batch_txts,
            convert_to_tensor=True,
            normalize_embeddings=False,
        )

        # explicit L2 normalisation
        vecs = F.normalize(vecs, p=2, dim=1)
        emb_batches.append(vecs)

    embeddings = torch.cat(emb_batches, dim=0)
    print("Embeddings tensor shape:", embeddings.shape)

    return embeddings, text_chunks, chunk_meta
    




# ------ Build Index --------

def build_index(
    embeddings,
    *,
    index_path: str | None = None,
    use_gpu: bool = False,
    hnsw_m: int | None = None,
) -> faiss.Index:
    """
    Build (or load) a FAISS index for cosine-similarity search.
    Works on both faiss-cpu and faiss-gpu wheels.

    Parameters:
    embeddings : np.ndarray | torch.Tensor
        Raw or already-normalized vectors (N, dim).  Torch tensors are
        detached -> CPU -> NumPy automatically.
    index_path : str | None, default None
        If a file exists, the index is loaded from disk; otherwise the
        newly-built index is saved there.
    use_gpu : bool, default False
        If True and faiss-gpu is available, the index is moved
        to CUDA device 0.
    hnsw_m : int | None
        If set, builds an HNSW index with that `M`.
        Otherwise builds an exact IndexFlatIP.
    """

    # Takes both torch or NumPy
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().float().numpy()
    elif not isinstance(embeddings, np.ndarray):
        raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(embeddings)}")

    # load from disk if it exists
    if index_path and os.path.isfile(index_path):
        print(f"Loading FAISS index from: {index_path}")
        index = faiss.read_index(index_path)
        if use_gpu and hasattr(faiss, "StandardGpuResources"):
            res   = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        elif use_gpu:
            warnings.warn("faiss-gpu not found; continuing on CPU.")
        return index

    # float32 and L2-normalize
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)
    faiss.normalize_L2(embeddings)

    # index type
    dim = embeddings.shape[1]
    if hnsw_m is not None:
        print(f"Building HNSW index (M={hnsw_m}) …")
        index = faiss.IndexHNSWFlat(dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 200
    else:
        print("Building exact IndexFlatIP …")
        index = faiss.IndexFlatIP(dim)

    index.add(embeddings)
    print(f"Index built with {index.ntotal:,} vectors  |  dim = {dim}")

    # Move to GPU if available
    if use_gpu and hasattr(faiss, "StandardGpuResources"):
        res   = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
        print("Index moved to GPU")
    elif use_gpu:
        warnings.warn("faiss-gpu not found; staying on CPU.")

    # persist to disk (CPU copy)
    if index_path:
        idx_to_save = faiss.index_gpu_to_cpu(index) if \
                      (use_gpu and hasattr(faiss, "index_gpu_to_cpu")) else index
        faiss.write_index(idx_to_save, index_path)
        print(f"Index saved to: {index_path}")

    return index
