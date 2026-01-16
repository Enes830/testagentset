import streamlit as st
from rag_system import RAGSystem
import config

with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Keys (can be overridden from config defaults)
    openai_api_key = st.text_input(
        "OpenAI API Key", 
        key="chatbot_api_key", 
        type="password",
        value=config.OPENAI_API_KEY
    )
    agentset_api_key = st.text_input(
        "Agentset API Key", 
        key="chatbot_agentset_api_key", 
        type="password",
        value=config.AGENTSET_API_KEY
    )
    agentset_namespace = st.text_input(
        "Agentset Namespace ID", 
        key="agentset_namespace",
        value=config.AGENTSET_NAMESPACE_ID
    )

st.title("💬 Playground")
st.caption("Agentset RAG chatbot example")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    if not openai_api_key or not agentset_api_key or not agentset_namespace:
        st.info("Please add your API keys and Agentset Namespace ID to continue.")
        st.stop()

    # Initialize RAG system with config settings
    rag = RAGSystem(
        agentset_namespace_id=agentset_namespace,
        agentset_api_token=agentset_api_key,
        openai_api_key=openai_api_key,
        system_prompt=config.SYSTEM_PROMPT,
    )

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.spinner("Thinking..."):
        result = rag.query(prompt, top_k=config.TOP_K, min_score=config.MIN_SCORE)
        msg = result["response"]

    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)

    with st.expander("📚 Retrieved Context"):
        st.text(result["context"])