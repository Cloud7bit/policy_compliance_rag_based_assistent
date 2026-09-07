import os
import streamlit as st
from databricks.sdk import WorkspaceClient

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# Set this to the serving endpoint name created in Vector_Search_Serving.ipynb
# (endpoint_name = f"{catalog}_{db}_document_search_chatbot")
SERVING_ENDPOINT = os.getenv("SERVING_ENDPOINT", "knowledge_base_default_document_search_chatbot")

st.set_page_config(
    page_title="AML/CFT Policy Compliance Assistant",
    page_icon="🛡️",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Workspace client — Databricks Apps injects auth automatically when the
# endpoint is added as a resource to the app (App settings > Resources).
# ---------------------------------------------------------------------------
@st.cache_resource
def get_client():
    return WorkspaceClient()


def query_endpoint(messages: list[dict]) -> str:
    """Send full chat history to the serving endpoint and return the reply text."""
    client = get_client()
    # Use a direct REST API call to the /invocations endpoint.
    # The serving_endpoints.query() method is for Foundation Model endpoints
    # only (chat completions format). This endpoint is a custom MLflow model
    # (LangChain chain) that expects {"inputs": {"messages": [...]}} and
    # returns {"predictions": ["answer text"]}.
    host = client.config.host.rstrip("/")
    url = f"{host}/serving-endpoints/{SERVING_ENDPOINT}/invocations"
    response = client.api_client.do(
        method="POST",
        url=url,
        body={"inputs": {"messages": messages}},
        headers={"Content-Type": "application/json"},
    )
    predictions = response.get("predictions", "")
    if isinstance(predictions, list):
        return predictions[0] if predictions else ""
    return predictions or str(response)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🛡️ AML/CFT Policy Compliance Assistant")
st.caption(
    "Ask about STR/CTR/KYC/sanctions/training KPIs, or Bank Asia AML/CFT policy "
    "and regulatory requirements."
)

with st.sidebar:
    st.subheader("About")
    st.markdown(
        "Hybrid RAG assistant combining:\n"
        "- **SQL** over KPI tables (STR, CTR, KYC/EDD, sanctions screening, training)\n"
        "- **Vector search** over Bank Asia AML/CFT policy documents\n\n"
        "Ask an analytical question, a policy question, or both in one query."
    )
    st.divider()
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()
    st.caption(f"Endpoint: `{SERVING_ENDPOINT}`")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask about STR breaches, KYC/EDD, sanctions, or policy requirements..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Checking KPI data and policy documents..."):
            try:
                answer = query_endpoint(st.session_state.messages)
            except Exception as e:
                answer = (
                    "⚠️ I couldn't reach the model serving endpoint. "
                    f"Details: `{e}`"
                )
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

# Example prompts
if not st.session_state.messages:
    st.markdown("**Try asking:**")
    examples = [
        "What was Bank Asia's STR SLA breach count in Gulshan Branch for March 2025?",
        "What are the KYC requirements for high-risk customers?",
        "Which branch had the most unreported CTRs in Q2 2025, and what do the regulations say about CTR filing?",
    ]
    for ex in examples:
        st.markdown(f"- {ex}")
