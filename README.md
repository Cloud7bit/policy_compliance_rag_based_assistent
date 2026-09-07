# Policy Compliance Assistant

A hybrid RAG system built on Databricks that answers AML/CFT compliance questions by combining **semantic search over policy PDFs** with **structured SQL querying of KPI data** — powered by a **Databricks Genie agent** with a custom RAG instruction set.

## 🎯 Project Overview

Compliance teams at banks need to answer two very different kinds of questions in the same conversation:

- **Analytical questions** — "What was the STR SLA breach count in Gulshan Branch for March 2025?"
- **Policy questions** — "What are the KYC requirements for high-risk customers?"

This project unifies both behind a single conversational interface by giving a **Databricks Genie agent** access to:

1. A vector-searchable knowledge base of AML/CFT policy documents (chunked PDFs)
2. Structured KPI tables covering STR/CTR filing, sanctions screening, KYC/EDD, and training compliance

A custom instruction set steers Genie to behave as a RAG system — querying document chunks for policy questions, aggregating KPI tables for analytical questions, and combining both when a question needs it.

## 🏗️ Architecture

```
┌─────────────────────┐        ┌──────────────────────┐
│  Policy PDFs (10)    │        │  Synthetic KPI CSVs  │
│  Bank Asia AML/CFT   │        │  STR/CTR/KYC/EDD/     │
│                      │        │  Sanctions/Training  │
└──────────┬───────────┘        └───────────┬──────────┘
           │                                │
           ▼                                ▼
┌──────────────────────┐        ┌──────────────────────┐
│  Data Ingestion       │        │  Unity Catalog        │
│  PyMuPDF chunking →   │        │  KPI Delta Tables      │
│  1,339 text chunks    │        │                        │
└──────────┬────────────┘        └───────────┬───────────┘
           │                                  │
           ▼                                  │
┌──────────────────────┐                      │
│ document_chunks table │                      │
└──────────┬────────────┘                      │
           │                                   │
           ▼                                   │
┌──────────────────────┐                       │
│ Vector Search Index    │                      │
│ (Delta Sync, HYBRID)   │                      │
│ databricks-gte-large-en│                      │
└──────────┬────────────┘                       │
           │                                    │
           └───────────────┬────────────────────┘
                            ▼
                ┌────────────────────────┐
                │  Databricks Genie Agent │
                │  + custom RAG instruction│
                └────────────┬─────────────┘
                             ▼
                ┌────────────────────────┐
                │   Conversational UI     │
                │  (Review App / Genie)   │
                └─────────────────────────┘
```

## 🔧 How It Works

Rather than hand-rolling a LangChain retrieval chain behind a Streamlit app, this project uses a **Databricks Genie agent** as the orchestration layer. Genie is given:

- Query access to the `document_chunks` table (policy text, chunked and embedded)
- Query access to the five KPI tables (STR, CTR, sanctions, KYC/EDD, training)
- A custom instruction set that defines its behavior, scope, and response format

### Custom Genie Instruction

```
You are an compliance analytics assistant. Answer questions ONLY using data from the knowledge base.

**Available Data:**
- CTR/STR filing metrics (volumes, timeliness, SLA compliance)
- Sanctions screening KPIs (hit rates, false positives, resolution times)
- KYC/EDD metrics (completion rates, overdue reviews)
- Training compliance (staff completion rates)
- Policy documents (10 AML/compliance PDFs with 1,339 text chunks)

**Document Search:**
For policy questions, query document_chunks table using LIKE '%keyword%' on chunk_text. Always cite source_file in responses.

**Out of Scope:**
If information isn't in the knowledge base, respond: "I don't have this information in my knowledge base."

**Response Format:**
Cite specific metrics, time periods, banks/branches. Use tables for comparisons. Highlight SLA breaches and compliance gaps.
```

This lets Genie route naturally between structured aggregation (SQL over KPI tables) and unstructured retrieval (keyword search over `chunk_text`), while enforcing citation discipline and a defined "I don't know" fallback — the core behaviors of a RAG system, without a separate retrieval chain to maintain.

## 📁 Project Structure

```
aml-cft-compliance-assistant/
├── README.md                    # This file
├── Data_Ingestion.py             # Extracts & chunks PDFs into document_chunks table
└── Vector_Search_Serving.py      # Creates the Vector Search index over document_chunks
```

## 📊 Data Pipeline

### Ingestion

- **Source**: 10 AML/CFT policy PDFs (Bank Asia) + synthetic KPI CSVs for Bangladeshi banks
- **Chunking**: PyMuPDF text extraction with token-based chunking
- **Output**: `document_chunks` Delta table (1,339 chunks) in Unity Catalog

### Vector Search

- **Endpoint type**: Delta Sync, HYBRID search
- **Embedding model**: `databricks-gte-large-en`
- **Index**: syncs automatically as `document_chunks` updates

### KPI Tables (Unity Catalog)

| Table | Description |
|---|---|
| `str_filing_kpi` | STR filing volumes, timeliness, SLA breaches |
| `ctr_reporting_kpi` | CTR filings, unreported transactions |
| `sanctions_screening_kpi` | Screening volumes, hit rates, SLA breaches |
| `kyc_edd_kpi` | KYC/EDD completion, high-risk customer counts |
| `training_compliance_kpi` | Staff AML/CFT training completion |

## 🧪 Example Questions

- "What was Bank Asia's STR SLA breach count in Gulshan Branch for March 2025?"
- "What are the KYC requirements for high-risk customers?"
- "Which branch had the most unreported CTRs in Q2 2025, and what do the regulations say about CTR filing?"

## 🛠️ Tech Stack

- **Databricks**: Unity Catalog, Vector Search, Genie, MLflow
- **Embeddings**: `databricks-gte-large-en`
- **PDF processing**: PyMuPDF
- **Data**: Bank Asia AML/CFT policy PDFs + synthetic KPI datasets

## 🔮 Future Enhancements

- [ ] Replace `LIKE` keyword search with native vector similarity search inside Genie's document queries
- [ ] Add citation-level confidence scoring
- [ ] Expand to multi-bank comparative analytics
- [ ] Automate document refresh pipeline for new policy versions

---

Built on Databricks — Unity Catalog, Vector Search, and Genie.
