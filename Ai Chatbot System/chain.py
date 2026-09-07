from databricks_langchain.vectorstores import DatabricksVectorSearch
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from databricks_langchain.chat_models import ChatDatabricks
from operator import itemgetter
import mlflow
import re

## Enable MLflow Tracing
mlflow.langchain.autolog()

## Load configuration
model_config = mlflow.models.ModelConfig()

## SQL execution uses Databricks SDK Statement Execution API
## (works in Model Serving with the endpoint's built-in auth)

## Vector Search retriever
vector_search_as_retriever = DatabricksVectorSearch(
    index_name=model_config.get("vector_search_index"),
    columns=["id", "chunk_text", "source_file", "file_type"]
).as_retriever(search_kwargs={"k": 10})

## Prompt templates
prompt = ChatPromptTemplate.from_messages(
    [("system", model_config.get("llm_prompt_template")), ("user", "{question}")]
)

sql_prompt_template = model_config.get("sql_prompt_template")

## Foundation model
model = ChatDatabricks(
    endpoint=model_config.get("llm_model_serving_endpoint_name"),
    temperature=0.1,
    max_tokens=500,
)

## Helper functions
def extract_user_query_string(chat_messages_array):
    return chat_messages_array[-1]["content"]

def format_context(docs):
    formatted = []
    for doc in docs:
        source = doc.metadata.get('source_file', 'Unknown')
        file_type = doc.metadata.get('file_type', 'Unknown')
        text = doc.page_content
        formatted.append(f"[{file_type}: {source}]\n{text}")
    return "\n\n---\n\n".join(formatted)

def is_analytical_query(question: str) -> bool:
    """Detect if question requires SQL query for analytics/KPIs"""
    analytical_keywords = [
        "count", "total", "average", "rate", 
        "most", "least", "highest", "lowest",
        "which branch", "how many", "compare"
    ]
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in analytical_keywords)

def has_document_keywords(question: str) -> bool:
    """Detect if question needs document/regulation context"""
    document_keywords = [
        "regulation", "policy", "guideline", "requirement",
        "what does", "explain", "how to", "what are",
        "compliance", "rule", "law", "standard", "aml", "cft"
    ]
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in document_keywords)

def generate_sql_query(question: str) -> str:
    """Use LLM to generate SQL query from natural language question"""
    kpi_tables = model_config.get("kpi_tables")
    catalog = list(kpi_tables.values())[0].split('.')[0] if kpi_tables else "knowledge_base"
    db = list(kpi_tables.values())[0].split('.')[1] if kpi_tables else "default"
    
    sql_system_prompt = sql_prompt_template.replace("{catalog}", catalog)
    sql_system_prompt = sql_system_prompt.replace("{db}", db)
    
    sql_prompt = ChatPromptTemplate.from_messages([
        ("system", sql_system_prompt),
        ("user", "{question}")
    ])
    
    sql_chain = sql_prompt | model | StrOutputParser()
    sql_query = sql_chain.invoke({"question": question})
    
    sql_query = re.sub(r'^```sql\s*', '', sql_query, flags=re.MULTILINE)
    sql_query = re.sub(r'```\s*$', '', sql_query, flags=re.MULTILINE)
    sql_query = sql_query.strip()
    
    return sql_query

def execute_sql_query(sql_query: str) -> str:
    """Execute SQL query using Databricks SDK Statement Execution API"""
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.sql import ExecuteStatementRequestOnWaitTimeout
    
    try:
        if 'limit' not in sql_query.lower():
            sql_query = sql_query.rstrip(';') + ' LIMIT 100'
        
        w = WorkspaceClient()
        warehouse_id = model_config.get("sql_warehouse_http_path").split("/")[-1]
        
        response = w.statement_execution.execute_statement(
            statement=sql_query,
            warehouse_id=warehouse_id,
            wait_timeout="30s",
            on_wait_timeout=ExecuteStatementRequestOnWaitTimeout.CANCEL,
            row_limit=100,
        )
        
        if response.status and response.status.state != "SUCCEEDED":
            error_msg = response.status.error.message if response.status.error else f"State: {response.status.state}"
            return f"Error executing SQL query: {error_msg}\n\nQuery: {sql_query}"
        
        if not response.result or not response.result.data_array:
            return f"SQL Query Executed:\n{sql_query}\n\nNo results found for this query."
        
        columns = [c.name for c in response.manifest.schema.columns]
        rows = response.result.data_array
        
        header = " | ".join(columns)
        separator = "-+-".join(["-" * max(len(c), 3) for c in columns])
        data_lines = [" | ".join(str(v) if v is not None else "NULL" for v in row) for row in rows[:50]]
        
        truncation_note = f"\n(Showing first 50 of {len(rows)} rows)" if len(rows) > 50 else ""
        context = f"SQL Query Executed:\n{sql_query}\n\nResults:\n{header}\n{separator}\n" + "\n".join(data_lines) + truncation_note
        context += f"\n\nTotal rows: {len(rows)}"
        
        return context
        
    except Exception as e:
        return f"Error executing SQL query: {str(e)}\n\nQuery: {sql_query}"

def route_query(question: str) -> str:
    """Route query to SQL, Vector Search, or BOTH based on question type"""
    needs_sql = is_analytical_query(question)
    needs_docs = has_document_keywords(question)
    
    contexts = []
    
    if needs_sql:
        sql_query = generate_sql_query(question)
        sql_context = execute_sql_query(sql_query)
        contexts.append(f"[SQL Query Results]\n{sql_context}")
    
    if needs_docs:
        docs = vector_search_as_retriever.invoke(question)
        doc_context = format_context(docs)
        contexts.append(f"[Document Context]\n{doc_context}")
    
    if not contexts:
        docs = vector_search_as_retriever.invoke(question)
        contexts.append(format_context(docs))
    
    return "\n\n" + "="*70 + "\n\n".join(contexts)

## Hybrid RAG Chain
chain = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_user_query_string),
        "context": itemgetter("messages")
        | RunnableLambda(extract_user_query_string)
        | RunnableLambda(route_query),
    }
    | prompt
    | model
    | StrOutputParser()
)

## Set chain as model
mlflow.models.set_model(model=chain)
