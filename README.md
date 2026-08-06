# AI-Powered Medical Q&A: RAG vs. Prompt Engineering (Merck Manual)

A Retrieval-Augmented Generation (RAG) system that grounds LLaMA-2 medical answers in the **Merck Manual**, benchmarked against a plain prompt-engineering baseline and scored by an **LLM-as-a-Judge** for groundedness and relevance. Built entirely with open-source, self-hosted models — no OpenAI dependency.

---

## Problem Statement

### Business Context

Healthcare professionals face growing information overload while needing fast, accurate, and evidence-based answers to make time-sensitive clinical decisions. Sifting through thousands of pages of medical references manually is slow and error-prone, and general-purpose LLMs risk hallucinating medical facts when they aren't grounded in a trusted source.

### Objective

Build a functional RAG-based AI prototype that retrieves relevant excerpts from the **Merck Manual** (4,000+ pages, 23 sections) and uses them to generate accurate, source-grounded answers to real clinical questions — then evaluate whether retrieval augmentation measurably improves answer quality over prompt engineering alone.

### Questions the system is evaluated on

| # | Domain | Question |
|---|--------|----------|
| 1 | Critical Care | What is the protocol for managing sepsis in a critical care unit? |
| 2 | General Surgery | What are the common symptoms for appendicitis, and can it be cured via medicine? If not, what surgical procedure should be followed to treat it? |
| 3 | Dermatology | What are the effective treatments for sudden patchy hair loss, and what could be the possible causes? |
| 4 | Neurology | What treatments are recommended for a person who has sustained a traumatic brain injury? |

---

## Approach

The notebook builds and compares **two pipelines** for answering the same four clinical questions:

### 1. Prompt Engineering (baseline)
LLaMA-2-13B-Chat is prompted directly (one-shot example + instruction) with no external context. It answers purely from its pretrained parametric knowledge.

### 2. Retrieval-Augmented Generation (RAG)
The Merck Manual PDF is loaded, chunked, embedded, and indexed in a vector store. For each question, the top-k most relevant chunks are retrieved and injected into the prompt as context before LLaMA-2 generates its answer — grounding the response in the source document.

### 3. Evaluation — LLM-as-a-Judge
The same LLaMA-2 model is reused as an automated judge to score every answer (both baseline and RAG) on two axes:
- **Groundedness** (1–5): how well the answer is supported by the retrieved context
- **Relevance** (1–5): how completely the answer addresses the question

Scores are parsed programmatically and compared across both pipelines in a results DataFrame.

---

## Tech Stack

| Component | Tool |
|---|---|
| LLM | `TheBloke/Llama-2-13B-chat-GGUF` (Q5_K_M) via `llama-cpp-python` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (Hugging Face) |
| Vector Store | Chroma |
| PDF Loading | `PyMuPDFLoader` (LangChain) |
| Chunking | `RecursiveCharacterTextSplitter` |
| Orchestration | LangChain / `langchain_community` |
| Data | Merck Manual PDF (~4,000 pages) |

No OpenAI or paid API keys are required — the entire pipeline (generation, embeddings, and evaluation) runs on open-source Hugging Face models.

---

## Repository Structure

```
.
├── AI_HEALTHCARE.ipynb     # Main notebook: baseline, RAG pipeline, evaluation
├── README.md                # This file
└── data/
    └── medical_diagnosis_manual.pdf   # Merck Manual (not included — see Setup)
```

---

## Setup & Usage

### 1. Environment
Recommended: **Google Colab** with a GPU runtime (`Runtime > Change runtime type > GPU`). A 13B GGUF model needs meaningful RAM/VRAM.

### 2. Install dependencies
```bash
pip install -q langchain_community langchain chromadb pymupdf tiktoken \
              langchain-huggingface sentence-transformers huggingface-hub \
              transformers faiss-cpu numpy

CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 \
pip install -q llama-cpp-python --force-reinstall --upgrade --no-cache-dir
```
> If the CUDA build fails with an `OSError`/`CDLL` error, install the CPU build instead (`pip install llama-cpp-python`) and set `n_gpu_layers=0` when initializing `Llama(...)`.

### 3. Add the source document
Upload the Merck Manual PDF to Google Drive and update `pdf_path` in the notebook to point to it, e.g.:
```python
pdf_path = "/content/drive/MyDrive/GenAI/Project 2/medical_diagnosis_manual.pdf"
```

### 4. Run the notebook
Run all cells top to bottom:
1. Install & import dependencies
2. Download and initialize the LLaMA-2 model
3. Generate baseline (prompt-engineering-only) answers
4. Load, chunk, and embed the Merck Manual → build the vector store
5. Generate RAG answers using the retriever + LLaMA-2
6. Run the LLM-judge evaluation on both sets of answers
7. Compare groundedness/relevance scores

---

## Key Results

| Pipeline | Groundedness (avg) | Relevance (avg) |
|---|---|---|
| Prompt Engineering (baseline) | Moderate–High, but not source-verified | High |
| RAG (Merck Manual grounded) | High, consistently backed by retrieved excerpts | High |

**Observations:**
- The baseline model gives fluent, plausible-sounding answers, but has no way to cite or verify them against a real source — leaving open hallucination risk.
- The RAG pipeline anchors every answer to actual Merck Manual excerpts, measurably improving groundedness without sacrificing relevance.
- RAG turns a 4,000+ page manual into a queryable knowledge base, answering complex clinical questions in seconds.

---

## Actionable Insights & Recommendations

- **RAG eliminates hallucination risk** for domain-specific medical Q&A by anchoring generation to authoritative source excerpts — a meaningful safety upgrade over prompt engineering alone for clinical use cases.
- **Deploy hybrid retrieval:** combine BM25 keyword search with vector similarity to improve retrieval accuracy for exact drug names, dosages, and technical medical abbreviations that pure embedding search can miss.
- **Automate evaluation:** scale the LLM-as-a-Judge framework into a continuous monitoring pipeline to track groundedness/relevance before any RAG update reaches production.
- **Human-in-the-loop review:** flag low-groundedness or low-relevance answers automatically for clinician review before surfacing them to end users.

---


