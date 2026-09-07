# Document Search System

A complete RAG (Retrieval Augmented Generation) system for searching and chatting with PDF and CSV documents using Databricks Vector Search and LangChain.

## 🎯 Project Overview

This project enables semantic search and question-answering over multiple file formats:
- **PDFs**: Extract and search text from policy documents, reports, manuals, etc.
- **CSVs**: Convert tabular data into searchable text for data-driven questions

## 📁 Project Structure

```
Document Search System/
├── README.md                          # This file
├── Data Ingestion.ipynb              # Processes PDFs & CSVs into chunks
└── Vector Search Serving.ipynb       # Deploys RAG chatbot endpoint

Csv_Pdf_knowledge_base/               # Unity Catalog
└── default/
    ├── document_chunks               # Delta table with all chunks
    ├── document_chunks_index         # Vector Search index
    ├── document_search_chatbot       # Registered model
    └── raw_documents/                # Volume for source files
```

## 🚀 Quick Start

### Step 1: Upload Your Documents

1. Navigate to Catalog Explorer
2. Go to `Csv_Pdf_knowledge_base` → `default` → `raw_documents` volume
3. Upload your PDF and CSV files

### Step 2: Run Data Ingestion

1. Open `Data Ingestion.ipynb`
2. Run all cells
3. Verify chunks in the `document_chunks` table

### Step 3: Create Vector Search Index

Before running the serving notebook, create a Vector Search index:

```python
from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()

# Create endpoint (if needed)
vsc.create_endpoint(
    name="document_search_endpoint",
    endpoint_type="STANDARD"
)

# Create index
vsc.create_delta_sync_index(
    endpoint_name="document_search_endpoint",
    index_name="Csv_Pdf_knowledge_base.default.document_chunks_index",
    source_table_name="Csv_Pdf_knowledge_base.default.document_chunks",
    pipeline_type="TRIGGERED",
    primary_key="id",
    embedding_source_column="chunk_text",
    embedding_model_endpoint_name="databricks-gte-large-en"
)
```

### Step 4: Deploy RAG Chatbot

1. Open `Vector Search Serving.ipynb`
2. Update configuration (endpoint names, model names)
3. Run all cells to deploy the chatbot
4. Test via the Review App or REST API

## 📊 Architecture

```
┌─────────────────┐
│  Raw Documents  │
│  (PDF + CSV)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Data Ingestion  │
│  - Extract Text │
│  - Chunk Data   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Delta Table    │
│ document_chunks │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Vector Search   │
│     Index       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   RAG Chain     │
│ (LangChain)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Model Serving   │
│   Endpoint      │
└─────────────────┘
```

## 🔧 Configuration

### Data Ingestion Configuration

- **Catalog**: `Csv_Pdf_knowledge_base`
- **Schema**: `default`
- **Table**: `document_chunks`
- **Volume**: `raw_documents`
- **Max Tokens per Chunk**: 500
- **Tokenizer**: `llama-tokenizer`

### Vector Search Configuration

- **Endpoint**: `document_search_endpoint`
- **Index**: `document_chunks_index`
- **Embedding Model**: `databricks-gte-large-en`
- **Top K Retrieval**: 5 chunks

### RAG Configuration

- **LLM Model**: `databricks-meta-llama-3-3-70b-instruct`
- **Temperature**: 0.1
- **Max Tokens**: 500
- **Registered Model**: `Csv_Pdf_knowledge_base.default.document_search_chatbot`

## 📝 Data Schema

### document_chunks Table

| Column | Type | Description |
|--------|------|-------------|
| `id` | BIGINT | Auto-generated unique identifier |
| `chunk_text` | STRING | The text chunk for embedding |
| `source_file` | STRING | Original filename |
| `file_type` | STRING | 'PDF' or 'CSV' |

## 🧪 Testing

### Sample Questions

- "What documents do you have?"
- "Summarize the main topics in the PDFs"
- "What data is available in the CSV files?"
- "Tell me about [specific topic from your documents]"
- "Compare information between different files"

### Testing via Python

```python
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

response = w.serving_endpoints.query(
    name="document_search_chatbot",
    messages=[
        {"role": "user", "content": "What information do you have?"}
    ]
)

print(response.choices[0].message.content)
```

## 🔄 Updating Documents

To add new documents:

1. Upload new files to the `raw_documents` volume
2. Re-run the `Data Ingestion` notebook
3. The vector search index will automatically sync (if using Delta Sync)
4. Your chatbot immediately has access to the new content!

## 🐛 Troubleshooting

### Issue: "Vector Search index not found"
**Solution**: Create the vector search index before running the serving notebook (see Step 3)

### Issue: "No chunks found in table"
**Solution**: Run the Data Ingestion notebook first to populate the table

### Issue: "Poor answer quality"
**Solutions**:
- Increase `k` (number of retrieved chunks) in the retriever config
- Adjust the prompt template to be more specific
- Check that your source documents contain relevant information
- Verify chunk sizes are appropriate (not too large or small)

### Issue: "Deployment failed"
**Solutions**:
- Check Model Serving endpoint logs
- Verify all resources (vector index, LLM endpoint) are accessible
- Ensure Unity Catalog permissions are set correctly

## 📚 Key Features

✅ **Multi-Format Support**: PDFs and CSVs in one unified system  
✅ **Automatic Chunking**: Intelligent token-based text splitting  
✅ **Metadata Tracking**: Know which document each answer came from  
✅ **Change Data Feed**: Automatic vector index updates  
✅ **Scalable Architecture**: Handles large document collections  
✅ **Production-Ready**: MLflow tracking, Model Serving, scale-to-zero  
✅ **Fully Documented**: Comprehensive inline documentation  

## 🔮 Future Enhancements

- [ ] Support for Microsoft Office formats (Word, Excel, PowerPoint)
- [ ] Incremental ingestion (only process new/modified files)
- [ ] Citation tracking in answers
- [ ] Custom Databricks App UI
- [ ] User feedback collection
- [ ] Multi-user access control
- [ ] Conversation history and context

## 📖 References

- [Databricks Vector Search Documentation](https://docs.databricks.com/en/generative-ai/vector-search.html)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Databricks Model Serving](https://docs.databricks.com/en/machine-learning/model-serving/index.html)

---

**Built with ❤️ using Databricks, Vector Search, and LangChain**