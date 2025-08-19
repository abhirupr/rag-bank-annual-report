# RAG Bank Annual Report

End-to-end Retrieval-Augmented Generation (RAG) pipeline in Databricks for ingesting, parsing, chunking, embedding, indexing, and querying Bank Annual Reports. It supports hybrid hierarchical chunking (narrative + tables), Databricks Vector Search indices, OpenAI + Databricks embedding backends, and enhanced retrieval with optional reranking.

## ✨ Key Features
- PDF parsing with [Docling](https://github.com/docling-project) (OCR + table structure extraction)
- Custom hierarchical + hybrid chunker (`CustomHybridChunker`) preserving multi-level headings
- Dual embedding generation: OpenAI (`text-embedding-3-large`) + Databricks model endpoint (`databricks-gte-large-en`)
- Delta tables with Change Data Feed enabled for incremental updates
- Automated Vector Search index creation & (re)building
- Unified or hybrid retrieval modes with optional FlashRank reranking
- Retrieval QA prompt tailored for financial analysis (FinanceGPT style)

## 🗂 Project Structure

```
scripts/
    parser_chunker.py          # Ingest + parse PDFs, generate chunks + embeddings, persist to Delta
    vector_store.py            # Manage Vector Search indices (rebuild, sync) based on embeddings
rag_app.ipynb              # Interactive experimentation / querying
helper/
   custom_chunker.py           # Custom hierarchical + hybrid chunker implementation
   functions.py                # Databricks + retrieval utilities (index mgmt, retrievers, QA classes)
config/
   training_config.example.yaml  # Example config (copy to training_config.yaml and edit)
requirements.txt
README.md
```

## 🔧 Installation

1. (Recommended) Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## ⚙️ Configuration
Copy `config/training_config.example.yaml` to `config/training_config.yaml` and set:

```yaml
data:
   file_path: /abs/path/to/pdf/folder
   catalog_name: <databricks_catalog>
   schema_name: <databricks_schema>
endpoint:
   databricks_embeddings_model_endpoint: <dbx_embedding_endpoint>
   openai_embeddings_model_endpoint: text-embedding-3-large
auth:
   databricks_host: https://<workspace>.cloud.databricks.com
   databricks_token: <pat_or_secret>
   openai_api_key: sk-...
```

Ensure secrets are stored securely (Databricks secrets scope or environment variables) when running in production.

## ▶️ Usage

### 1. Generate chunks + embeddings
```bash
python parser_chunker.py
```
This will:
1. Scan PDF directory and upsert metadata table
2. Parse each PDF with Docling (OCR + tables)
3. Chunk using custom hierarchical hybrid chunker
4. Build a Spark DataFrame of chunks
5. Generate Databricks + OpenAI embeddings (if table not already populated)
6. Persist to Delta table (with CDF enabled)

### 2. Build / rebuild Vector Search indices
```bash
python vector_store.py
```
Controls:
- Set `FORCE_REBUILD_INDICES = True` in `vector_store.py` to drop & recreate indices
- Embedding provider currently set to `openai` (adjust for Databricks native embeddings)

### 3. Experiment & Query
Open `rag_app.ipynb` and construct QA pipelines using:
- `EnhancedRetrievalQA` (unified index)
- `EnhancedHybridRetrievalQA` (separate narrative + table indices, if extended)

## 📦 Embedding Columns
- Databricks embeddings stored in: `embedding` (dimension 1024)
- OpenAI embeddings stored in: `openai_embedding` (dimension 3072)

## 🔍 Retrieval Modes
- Basic similarity: `get_retriever`
- Hybrid (headers + content): `get_enhanced_retriever`
- Reranked (FlashRank): `get_enhanced_retriever_with_reranking`
- Hybrid + reranking (narrative + tables): `get_enhanced_hybrid_retriever_with_reranking`

FlashRank is optional; install with:
```bash
pip install flashrank
```

## 🧩 Custom Chunking
`CustomHybridChunker` wraps a modified `CustomHierarchicalChunker` that:
- Flattens headings across levels for richer context
- Seals heading scopes after emitting content (prevents heading bleed)
- Supports table + narrative continuity

## 🛡 Prerequisites
- Python 3.10+ recommended
- Databricks Workspace (for Delta + Vector Search)
- OpenAI API key (unless only Databricks embeddings are used)

## 🧪 Testing & Validation
Add PDFs to the configured `file_path` and run the pipeline. After execution, validate:
- Delta tables exist in `catalog.schema`
- Vector Search index is ONLINE
- Sample query returns relevant passages

## 🚑 Troubleshooting
| Issue | Possible Cause | Fix |
|-------|----------------|-----|
| Table not created | Wrong catalog/schema | Check config YAML values |
| OpenAI auth error | Missing key | Set `OPENAI_API_KEY` env var or YAML entry |
| Index not rebuilding | `FORCE_REBUILD_INDICES` False | Set to True and rerun `vector_store.py` |
| FlashRank warning | Package missing | `pip install flashrank` |
| OCR slow | Large PDFs / OCR enabled | Disable OCR (`pipeline_options.do_ocr = False`) if text layer present |
