# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Hybrid RAG Architecture Guide
# MAGIC %md
# MAGIC ## 🎯 Hybrid RAG Architecture for Your Assistant
# MAGIC
# MAGIC ### The Problem You Had
# MAGIC Your original approach converted CSV rows into text chunks for vector search. This **cannot handle analytical queries** like:
# MAGIC - "What was Bank Asia's STR SLA breach count?" (needs filtering + aggregation)
# MAGIC - "Which branch had the most unreported CTRs?" (needs grouping + ranking)
# MAGIC - "What's the EDD completion rate?" (needs calculation across columns)
# MAGIC
# MAGIC ### The Solution: Dual-Path Processing
# MAGIC
# MAGIC ```
# MAGIC ┌─────────────────────────┐
# MAGIC │   Raw Documents Volume  │
# MAGIC └───────┬───────────┬──────┘
# MAGIC         │           │
# MAGIC     PDFs │           │ CSVs
# MAGIC         │           │
# MAGIC         ▼           ▼
# MAGIC    Text Chunks   Structured Tables
# MAGIC         │           │
# MAGIC   Vector Search   SQL Queries
# MAGIC         │           │
# MAGIC         └───┬───────┘
# MAGIC             ▼
# MAGIC      Hybrid RAG Assistant
# MAGIC ```
# MAGIC
# MAGIC ### What Your Assistant Should Do
# MAGIC
# MAGIC **1. Intent Classification**
# MAGIC - **Analytical query** (aggregation, filtering, calculations) → Generate SQL
# MAGIC - **Document search** (regulations, guidelines, definitions) → Vector search
# MAGIC
# MAGIC **2. SQL Generation for KPIs**
# MAGIC ```python
# MAGIC # Pseudo-code for your RAG assistant
# MAGIC if user_query contains ["count", "rate", "total", "average", "which branch"]:
# MAGIC     # Use LLM to generate SQL
# MAGIC     sql_query = generate_sql_from_question(user_query)
# MAGIC     result = spark.sql(sql_query).toPandas()
# MAGIC     return format_answer(result)
# MAGIC ```
# MAGIC
# MAGIC **3. Vector Search for Documents**
# MAGIC ```python
# MAGIC else:
# MAGIC     # Use vector search on PDF chunks
# MAGIC     results = vector_search_index.search(user_query, num_results=5)
# MAGIC     return generate_answer_from_chunks(results)
# MAGIC ```
# MAGIC
# MAGIC ### Available KPI Tables
# MAGIC After running the updated ingestion:
# MAGIC - `knowledge_base.default.str_filing_kpi` - STR filing metrics
# MAGIC - `knowledge_base.default.ctr_reporting_kpi` - CTR reporting metrics
# MAGIC - `knowledge_base.default.kyc_edd_kpi` - KYC/EDD metrics
# MAGIC - `knowledge_base.default.sanctions_screening_kpi` - Sanctions screening metrics
# MAGIC - `knowledge_base.default.training_compliance_kpi` - Training metrics
# MAGIC
# MAGIC ### Implementation Steps
# MAGIC 1. ✅ Run the updated Cell 9 to create structured tables
# MAGIC 2. ✅ Test SQL queries (cells below) to verify data
# MAGIC 3. ⚡ Update your RAG assistant to route queries:
# MAGIC    - Detect analytical intent → Generate & execute SQL
# MAGIC    - Detect document search → Use vector search
# MAGIC 4. 🛠️ Consider using Databricks SQL Agent or function calling to automate SQL generation

# COMMAND ----------

# DBTITLE 1,Document Search System - Data Ingestion
# MAGIC %md
# MAGIC # Document Search System - Data Ingestion
# MAGIC
# MAGIC ## Overview
# MAGIC This notebook processes both **PDF and CSV files** from a Unity Catalog volume, converts them into searchable text chunks, and stores them in a Delta table for vector search retrieval.
# MAGIC
# MAGIC ## Architecture
# MAGIC ```
# MAGIC Raw Documents (Volume)
# MAGIC ├── PDFs → Extract Text → Chunk by Tokens
# MAGIC └── CSVs → Row-to-Text → Chunk by Tokens
# MAGIC                     ↓
# MAGIC         Delta Table (with Change Data Feed)
# MAGIC                     ↓
# MAGIC             Vector Search Index
# MAGIC ```
# MAGIC
# MAGIC ## Key Features
# MAGIC * **Multi-Format Support**: Handles PDFs and CSVs in one unified pipeline
# MAGIC * **Token-Based Chunking**: Ensures optimal chunk sizes for embeddings
# MAGIC * **Metadata Tracking**: Preserves source file information
# MAGIC * **Delta Lake Storage**: ACID transactions with Change Data Feed for vector sync
# MAGIC * **Batch Processing**: Automatically processes all files in the volume
# MAGIC
# MAGIC ## Configuration
# MAGIC * **Catalog**: `knowledge_base`
# MAGIC * **Schema**: `default`
# MAGIC * **Table**: `document_chunks`
# MAGIC * **Volume**: `raw_documents`
# MAGIC * **Max Tokens per Chunk**: 500

# COMMAND ----------

# DBTITLE 1,Install Required Packages
# Install required libraries
%pip install pypdf transformers pandas
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Import Libraries
import os
import pypdf
import pandas as pd
from typing import List, Dict, Tuple
from transformers import AutoTokenizer
from pyspark.sql import SparkSession

# COMMAND ----------

# DBTITLE 1,Configuration
# === Configuration ===
CATALOG_NAME = "knowledge_base"
SCHEMA_NAME = "default"
TABLE_NAME = f"{CATALOG_NAME}.{SCHEMA_NAME}.document_chunks"
VOLUME_PATH = f"/Volumes/{CATALOG_NAME}/{SCHEMA_NAME}/raw_documents"

# Tokenizer configuration
TOKENIZER_MODEL = "llama_v3_3_70b_instruct"
TOKENIZER_CACHE = "/tmp/hf_cache"

# Chunking parameters
MAX_TOKENS_PER_CHUNK = 500
SEPARATORS = ["\n\n", "\n", ". ", ", ", " ", ""]

print(f"Target Table: {TABLE_NAME}")
print(f"Source Volume: {VOLUME_PATH}")
print(f"Max Tokens per Chunk: {MAX_TOKENS_PER_CHUNK}")

# COMMAND ----------

# DBTITLE 1,Text Chunking Function
def chunk_text(text: str, tokenizer: AutoTokenizer) -> List[str]:
    """
    Splits text into chunks under the token limit by progressively
    splitting with different separators. Ensures no chunk exceeds MAX_TOKENS_PER_CHUNK.
    
    Args:
        text: The text to chunk
        tokenizer: The tokenizer for counting tokens
    
    Returns:
        List of text chunks, each under MAX_TOKENS_PER_CHUNK tokens
    """
    chunks = [text]
    
    # Progressively split using different separators
    for separator in SEPARATORS:
        new_chunks = []
        for chunk in chunks:
            # Check if the chunk is too large and needs to be split further
            if len(tokenizer.encode(chunk)) > MAX_TOKENS_PER_CHUNK:
                # Split the chunk by the current separator
                parts = chunk.split(separator)
                current_chunk = ""
                
                for part in parts:
                    combined = current_chunk + separator + part if current_chunk else part
                    
                    if len(tokenizer.encode(combined)) <= MAX_TOKENS_PER_CHUNK:
                        current_chunk = combined
                    else:
                        if current_chunk:
                            new_chunks.append(current_chunk.strip())
                        
                        # If part itself is too large, split it into smaller pieces
                        while len(tokenizer.encode(part)) > MAX_TOKENS_PER_CHUNK:
                            tokens = tokenizer.encode(part)
                            sub_part = tokenizer.decode(tokens[:MAX_TOKENS_PER_CHUNK])
                            new_chunks.append(sub_part.strip())
                            part = tokenizer.decode(tokens[MAX_TOKENS_PER_CHUNK:])
                            tokens = tokenizer.encode(part)
                        current_chunk = part.strip()
                
                if current_chunk:
                    new_chunks.append(current_chunk.strip())
            else:
                new_chunks.append(chunk.strip())
        
        chunks = new_chunks
    
    # Final pass: ensure no chunk exceeds MAX_TOKENS_PER_CHUNK
    final_chunks = []
    for chunk in chunks:
        tokens = tokenizer.encode(chunk)
        start = 0
        while start < len(tokens):
            sub_tokens = tokens[start:start+MAX_TOKENS_PER_CHUNK]
            sub_chunk = tokenizer.decode(sub_tokens)
            final_chunks.append(sub_chunk.strip())
            start += MAX_TOKENS_PER_CHUNK

    # Filter out any empty strings
    return [chunk for chunk in final_chunks if chunk]

# COMMAND ----------

# DBTITLE 1,PDF Processing Functions
def get_pdf_text(pdf_path: str) -> str:
    """
    Reads all text from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
    
    Returns:
        Extracted text from all pages
    """
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = pypdf.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
    except Exception as e:
        print(f"  ✗ Error reading PDF file {pdf_path}: {e}")
    return text


def process_pdf_file(pdf_path: str, tokenizer: AutoTokenizer) -> Tuple[List[str], str]:
    """
    Processes a single PDF file and returns chunks with metadata.
    
    Args:
        pdf_path: Path to the PDF file
        tokenizer: The tokenizer for counting tokens
    
    Returns:
        Tuple of (list of chunks, source filename)
    """
    filename = os.path.basename(pdf_path)
    print(f"  Processing PDF: {filename}")
    
    # Extract text from PDF
    pdf_text = get_pdf_text(pdf_path)
    
    if not pdf_text:
        print(f"    ⚠ PDF text is empty")
        return [], filename
    
    # Chunk the text
    chunks = chunk_text(pdf_text, tokenizer)
    print(f"    ✓ Generated {len(chunks)} chunks")
    
    return chunks, filename

# COMMAND ----------

# DBTITLE 1,CSV Processing Functions
def csv_row_to_text(row: pd.Series, column_names: List[str], filename: str) -> str:
    """
    Converts a CSV row into a natural language text description.
    
    Args:
        row: A pandas Series representing one row
        column_names: List of column names
        filename: Name of the source CSV file
    
    Returns:
        A natural language description of the row
    """
    # Create a structured text representation
    text_parts = [f"Source: {filename}"]
    
    for col in column_names:
        value = row[col]
        # Skip null/empty values
        if pd.notna(value) and str(value).strip():
            text_parts.append(f"{col}: {value}")
    
    return "\n".join(text_parts)


def process_csv_file(csv_path: str, tokenizer: AutoTokenizer) -> Tuple[List[str], str]:
    """
    Processes a single CSV file and returns chunks with metadata.
    
    Args:
        csv_path: Path to the CSV file
        tokenizer: The tokenizer for counting tokens
    
    Returns:
        Tuple of (list of chunks, source filename)
    """
    filename = os.path.basename(csv_path)
    print(f"  Processing CSV: {filename}")
    
    try:
        # Read CSV file
        df = pd.read_csv(csv_path)
        print(f"    Found {len(df)} rows and {len(df.columns)} columns")
        
        all_chunks = []
        
        # Process each row
        for idx, row in df.iterrows():
            # Convert row to text
            row_text = csv_row_to_text(row, df.columns.tolist(), filename)
            
            # Chunk the text if it's too long
            row_chunks = chunk_text(row_text, tokenizer)
            all_chunks.extend(row_chunks)
        
        print(f"    ✓ Generated {len(all_chunks)} chunks from {len(df)} rows")
        return all_chunks, filename
        
    except Exception as e:
        print(f"    ✗ Error processing CSV file: {e}")
        return [], filename

# COMMAND ----------

# DBTITLE 1,Insert Chunks to Delta Table
def insert_chunks_to_table(spark, chunks: List[str], source_file: str, file_type: str):
    """
    Inserts text chunks with metadata into the Delta table.
    
    Args:
        spark: SparkSession
        chunks: List of text chunks
        source_file: Name of the source file
        file_type: Type of file ('PDF' or 'CSV')
    """
    if not chunks:
        print(f"    ⚠ No chunks to insert from {source_file}")
        return
    
    try:
        # Create data with metadata
        data_to_insert = [
            (chunk, source_file, file_type) 
            for chunk in chunks
        ]
        
        # Create DataFrame and append to table
        spark.createDataFrame(
            data_to_insert, 
            ['chunk_text', 'source_file', 'file_type']
        ).write \
         .format("delta") \
         .mode("append") \
         .saveAsTable(TABLE_NAME)
        
        print(f"    ✓ Inserted {len(chunks)} chunks")
        
    except Exception as e:
        print(f"    ✗ Error inserting chunks to table: {e}")

# COMMAND ----------

# DBTITLE 1,Load CSVs as Structured Tables
def load_csv_as_table(csv_path: str, spark):
    """
    Loads a CSV file as a structured Delta table for SQL queries.
    
    Args:
        csv_path: Path to the CSV file
        spark: SparkSession
    """
    filename = os.path.basename(csv_path)
    table_name_suffix = filename.replace('.csv', '').replace('-', '_').replace(' ', '_')
    table_name = f"{CATALOG_NAME}.{SCHEMA_NAME}.{table_name_suffix}"
    
    print(f"  Loading CSV as table: {table_name}")
    
    try:
        # Read CSV into Spark DataFrame
        df = spark.read.csv(csv_path, header=True, inferSchema=True)
        
        # Write as Delta table
        df.write \
          .format("delta") \
          .mode("overwrite") \
          .option("overwriteSchema", "true") \
          .saveAsTable(table_name)
        
        row_count = df.count()
        print(f"    ✓ Loaded {row_count} rows into {table_name}")
        print(f"    Columns: {', '.join(df.columns)}")
        
        return table_name, row_count
        
    except Exception as e:
        print(f"    ✗ Error loading CSV as table: {e}")
        return None, 0

# COMMAND ----------

# DBTITLE 1,Main Execution - Process All Documents
# ====================================================================
# MAIN EXECUTION BLOCK
# ====================================================================

print("="*70)
print("  DOCUMENT SEARCH SYSTEM - DATA INGESTION")
print("="*70)

# Step 1: Verify volume exists
print("\n[1/5] Verifying volume...")
if not os.path.isdir(VOLUME_PATH):
    print(f"  ✗ Error: Volume not found at {VOLUME_PATH}")
    print(f"  Please upload your PDF and CSV files to this volume first.")
    raise FileNotFoundError(f"Volume not found: {VOLUME_PATH}")
else:
    print(f"  ✓ Volume found: {VOLUME_PATH}")

# Step 2: Initialize tokenizer
print("\n[2/5] Initializing tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    "hf-internal-testing/llama-tokenizer", 
    cache_dir=TOKENIZER_CACHE
)
print("  ✓ Tokenizer loaded")

# Step 3: Create or verify table
print(f"\n[3/5] Setting up table '{TABLE_NAME}'...")
try:
    spark.sql(f"DESCRIBE TABLE {TABLE_NAME}")
    print("  ✓ Table already exists")
    
    # Ask user about clearing existing data
    print("  ⚠ Table contains existing data. Truncating...")
    spark.sql(f"TRUNCATE TABLE {TABLE_NAME}")
    print("  ✓ Previous data cleared")
    
    # Ensure Change Data Feed is enabled
    spark.sql(f"ALTER TABLE {TABLE_NAME} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
    
except Exception:
    print("  Table not found. Creating new table...")
    create_table_sql = f"""
    CREATE TABLE {TABLE_NAME} (
        id BIGINT GENERATED BY DEFAULT AS IDENTITY,
        chunk_text STRING,
        source_file STRING,
        file_type STRING
    )
    TBLPROPERTIES (delta.enableChangeDataFeed = true)
    """
    spark.sql(create_table_sql)
    print("  ✓ Table created successfully")

print(f"  ✓ Table ready: {TABLE_NAME}")

# Step 4: Discover files
print("\n[4/5] Discovering files in volume...")
all_files = os.listdir(VOLUME_PATH)
pdf_files = [f for f in all_files if f.endswith('.pdf')]
csv_files = [f for f in all_files if f.endswith('.csv')]

total_files = len(pdf_files) + len(csv_files)

if total_files == 0:
    print("  ⚠ No PDF or CSV files found in volume")
    print("  Please upload files and run again.")
else:
    print(f"  Found {len(pdf_files)} PDF(s) and {len(csv_files)} CSV(s)")
    
    # Step 5: Process all files
    print("\n[5/5] Processing documents...")
    print("-"*70)
    
    total_chunks = 0
    successful_files = 0
    
    # Process PDFs
    if pdf_files:
        print(f"\n📄 Processing {len(pdf_files)} PDF file(s):")
        for pdf_filename in pdf_files:
            pdf_path = os.path.join(VOLUME_PATH, pdf_filename)
            chunks, source = process_pdf_file(pdf_path, tokenizer)
            
            if chunks:
                insert_chunks_to_table(spark, chunks, source, 'PDF')
                total_chunks += len(chunks)
                successful_files += 1
            print()
    
    # Process CSVs - Load as STRUCTURED TABLES for SQL queries
    if csv_files:
        print(f"\n📊 Loading {len(csv_files)} CSV file(s) as structured tables:")
        csv_tables = []
        for csv_filename in csv_files:
            csv_path = os.path.join(VOLUME_PATH, csv_filename)
            table_name, row_count = load_csv_as_table(csv_path, spark)
            
            if table_name:
                csv_tables.append(table_name)
                successful_files += 1
            print()
        
        print(f"\n  ✓ Created {len(csv_tables)} structured tables for SQL queries")
    
    # Final Summary
    print("="*70)
    print("  ✅ INGESTION COMPLETE!")
    print("="*70)
    print(f"  Files processed: {successful_files}/{total_files}")
    print(f"\n  PDF Processing:")
    print(f"    - Total chunks stored: {total_chunks}")
    print(f"    - Target table: {TABLE_NAME}")
    print(f"\n  CSV Processing:")
    print(f"    - Structured tables created: {len(csv_tables) if csv_files else 0}")
    if csv_files:
        for tbl in csv_tables:
            print(f"      • {tbl}")
    print("\n  Next steps:")
    print("  1. Run SQL queries on CSV tables (cells below)")
    print("  2. Create vector search index on PDF chunks")
    print("  3. Build hybrid RAG: SQL for analytics + Vector search for documents")
    print("="*70)

# COMMAND ----------

# DBTITLE 1,Verify Ingested Data
# MAGIC %sql
# MAGIC -- View ingested chunks with metadata
# MAGIC SELECT 
# MAGIC   id,
# MAGIC   file_type,
# MAGIC   source_file,
# MAGIC   LEFT(chunk_text, 100) as chunk_preview,
# MAGIC   LENGTH(chunk_text) as chunk_length
# MAGIC FROM knowledge_base.default.document_chunks
# MAGIC ORDER BY id DESC
# MAGIC LIMIT 20;

# COMMAND ----------

# DBTITLE 1,Statistics by File Type
# MAGIC %sql
# MAGIC -- Get statistics grouped by file type
# MAGIC SELECT 
# MAGIC   file_type,
# MAGIC   COUNT(*) as total_chunks,
# MAGIC   COUNT(DISTINCT source_file) as file_count,
# MAGIC   AVG(LENGTH(chunk_text)) as avg_chunk_length,
# MAGIC   MIN(LENGTH(chunk_text)) as min_chunk_length,
# MAGIC   MAX(LENGTH(chunk_text)) as max_chunk_length
# MAGIC FROM knowledge_base.default.document_chunks
# MAGIC GROUP BY file_type
# MAGIC ORDER BY file_type;

# COMMAND ----------

# DBTITLE 1,Query 1: Bank Asia STR SLA Breaches
# MAGIC %sql
# MAGIC -- Example Query: What was Bank Asia's STR SLA breach count in Gulshan Branch for March 2025?
# MAGIC SELECT 
# MAGIC   bank_name,
# MAGIC   branch,
# MAGIC   month,
# MAGIC   str_sla_breaches,
# MAGIC   str_filed_count,
# MAGIC   avg_str_filing_days
# MAGIC FROM knowledge_base.default.str_filing_kpi
# MAGIC WHERE bank_name = 'Bank Asia'
# MAGIC   AND branch = 'Gulshan Branch'
# MAGIC   AND month = '2025-03';

# COMMAND ----------

# DBTITLE 1,Query 2: Branches with Most Unreported CTRs in Q2 2025
# MAGIC %sql
# MAGIC -- Example Query: Which branch had the most unreported CTRs in Q2 2025?
# MAGIC SELECT 
# MAGIC   bank_name,
# MAGIC   branch,
# MAGIC   SUM(ctrs_unreported) as total_unreported_ctrs,
# MAGIC   SUM(ctrs_filed) as total_filed_ctrs
# MAGIC FROM knowledge_base.default.ctr_reporting_kpi
# MAGIC WHERE month IN ('2025-04', '2025-05', '2025-06')
# MAGIC GROUP BY bank_name, branch
# MAGIC ORDER BY total_unreported_ctrs DESC
# MAGIC LIMIT 10;

# COMMAND ----------

# DBTITLE 1,Next Steps - Vector Search Setup
# MAGIC %md
# MAGIC ## ✅ Data Ingestion Complete!
# MAGIC
# MAGIC ### What You Have Now
# MAGIC * **Catalog**: `knowledge_base`
# MAGIC * **Table**: `document_chunks` with Change Data Feed enabled
# MAGIC * **Data**: PDF and CSV chunks ready for vector search
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Next: Create Vector Search Index
# MAGIC
# MAGIC To enable semantic search over your documents, you need to:
# MAGIC
# MAGIC 1. **Create a Vector Search Endpoint** (if you don't have one)
# MAGIC    ```python
# MAGIC    from databricks.vector_search.client import VectorSearchClient
# MAGIC    
# MAGIC    vsc = VectorSearchClient()
# MAGIC    vsc.create_endpoint(
# MAGIC        name="document_search_endpoint",
# MAGIC        endpoint_type="STANDARD"
# MAGIC    )
# MAGIC    ```
# MAGIC
# MAGIC 2. **Create a Delta Sync Index** on your table
# MAGIC    ```python
# MAGIC    vsc.create_delta_sync_index(
# MAGIC        endpoint_name="document_search_endpoint",
# MAGIC        index_name="Csv_Pdf_knowledge_base.default.document_chunks_index",
# MAGIC        source_table_name="Csv_Pdf_knowledge_base.default.document_chunks",
# MAGIC        pipeline_type="TRIGGERED",
# MAGIC        primary_key="id",
# MAGIC        embedding_source_column="chunk_text",
# MAGIC        embedding_model_endpoint_name="databricks-gte-large-en"  # or your preferred model
# MAGIC    )
# MAGIC    ```
# MAGIC
# MAGIC 3. **Move to the Vector Search Serving notebook** to set up the RAG endpoint
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Tips for Better Results
# MAGIC * Use descriptive filenames for your source documents
# MAGIC * Ensure CSVs have meaningful column names
# MAGIC * Monitor chunk sizes - very small or very large chunks reduce retrieval quality
# MAGIC * Test your vector search with sample queries before deploying