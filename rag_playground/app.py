import streamlit as st
from rag_playground.rag_system import RAGSystem
from rag_playground.document_ingester import DocumentIngester
from rag_playground import config

# Page configuration
st.set_page_config(
    page_title="RAG Playground",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Hide chat avatars
st.markdown("""
<style>
/* Hide avatar container */
.stChatMessage > div:first-child {
    display: none !important;
    width: 0 !important;
    height: 0 !important;
    visibility: hidden !important;
}
[data-testid="stChatMessageAvatarContainer"] {
    display: none !important;
    width: 0 !important;
}
.stChatMessage svg {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
for key, default in [("messages", []), ("active_tab", "chat")]:
    if key not in st.session_state:
        st.session_state[key] = default


def is_configured():
    """Check if all required API keys are configured."""
    return all(
        [
            st.session_state.get("openai_api_key"),
            st.session_state.get("agentset_api_key"),
            st.session_state.get("agentset_namespace"),
        ]
    )


def get_ingester():
    """Create a DocumentIngester instance with current config."""
    return DocumentIngester(
        agentset_namespace_id=st.session_state.get("agentset_namespace"),
        agentset_api_token=st.session_state.get("agentset_api_key"),
    )


def get_rag_system():
    """Create a RAGSystem instance with current config."""
    return RAGSystem(
        agentset_namespace_id=st.session_state.get("agentset_namespace"),
        agentset_api_token=st.session_state.get("agentset_api_key"),
        openai_api_key=st.session_state.get("openai_api_key"),
        system_prompt=config.SYSTEM_PROMPT,
        model=st.session_state.get("openai_model", config.OPENAI_MODEL),
    )


def metadata_popover(key_prefix, max_fields=5):
    """Render metadata input fields in a popover and return the metadata dict."""
    with st.popover("Add metadata"):
        num = st.number_input("Number of fields", 0, max_fields, 0, key=f"{key_prefix}_meta_count")
        metadata = {}
        for i in range(int(num)):
            c1, c2 = st.columns(2)
            with c1:
                k = st.text_input(f"Key {i + 1}", key=f"{key_prefix}_k_{i}")
            with c2:
                v = st.text_input(f"Value {i + 1}", key=f"{key_prefix}_v_{i}")
            if k and v:
                metadata[k] = v
    return metadata or None


def show_ingest_result(result):
    """Display success or error message from an ingestion result."""
    if result["success"]:
        st.success(f"Ingestion started. Job ID: `{result['job_id']}`")
    else:
        st.error(result["message"])


# Sidebar - Configuration
with st.sidebar:
    st.title("RAG Playground")

    with st.expander("API Configuration", expanded=not is_configured()):
        api_fields = [
            ("OpenAI API Key", "openai_api_key", config.OPENAI_API_KEY, "sk-...", "Required for generating responses", True),
            ("Agentset API Key", "agentset_api_key", config.AGENTSET_API_KEY, "agentset_...", "Required for document retrieval", True),
            ("Namespace ID", "agentset_namespace", config.AGENTSET_NAMESPACE_ID, "ns_...", "Your Agentset namespace identifier", False),
            ("base URL", "agentset_base_url", "https://api.agentset.ai", "https://api.agentset.com", "Custom Agentset API base URL", False),
        ]
        for label, key, default, placeholder, help_text, is_password in api_fields:
            value = st.text_input(label, type="password" if is_password else "default",
                                  value=default or "", placeholder=placeholder, help=help_text)
            if value:
                st.session_state[key] = value

    st.markdown('<p class="sidebar-section">RAG Settings</p>', unsafe_allow_html=True)

    with st.expander("Advanced Settings"):
        # OpenAI Model Selection
        selected_model = st.selectbox(
            "OpenAI Model",
            options=config.AVAILABLE_MODELS,
            index=config.AVAILABLE_MODELS.index(st.session_state.get("openai_model", config.OPENAI_MODEL)),
            help="Select the OpenAI model for generating responses",
        )
        st.session_state["openai_model"] = selected_model

        st.divider()

        top_k = st.slider(
            "Results to retrieve",
            min_value=1,
            max_value=20,
            value=config.TOP_K,
            help="Number of document chunks to retrieve",
        )
        st.session_state["top_k"] = top_k

        min_score = st.slider(
            "Minimum relevance score",
            min_value=0.0,
            max_value=1.0,
            value=config.MIN_SCORE,
            step=0.05,
            help="Filter out results below this threshold",
        )
        st.session_state["min_score"] = min_score

    st.markdown('<p class="sidebar-section">Actions</p>', unsafe_allow_html=True)

    if st.button(
        "Clear Chat History",
        use_container_width=True,
        disabled=len(st.session_state.get("messages", [])) == 0,
    ):
        st.session_state["messages"] = []
        st.rerun()


# Main content area with tabs
tab_chat, tab_ingest = st.tabs(["Chat", "Ingest Documents"])

# Chat Tab
with tab_chat:
    # Create a container for chat messages (keeps chat_input at bottom)
    chat_container = st.container()
    
    # Chat input - placed here so it stays at the bottom
    prompt = st.chat_input("Ask a question about your documents...")
    
    with chat_container:
        # Show welcome message if no messages yet
        if not st.session_state["messages"]:
            st.markdown("""
            ### Welcome to RAG Playground
            
            Ask questions and get answers based on your ingested documents.
            
            **Quick start:**
            1. Configure your API keys in the sidebar
            2. Ingest documents in the "Ingest Documents" tab
            3. Start asking questions below
            """)

        # Display chat messages
        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if msg.get("context"):
                    with st.expander("View retrieved context"):
                        st.text(msg["context"])

    # Process new input
    if prompt:
        if not is_configured():
            st.error("Please configure your API keys in the sidebar to continue.")
            st.stop()

        # Add user message
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"):
                st.write(prompt)

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Searching and generating response..."):
                    rag = get_rag_system()
                    result = rag.query(
                        prompt,
                        top_k=st.session_state.get("top_k", config.TOP_K),
                        min_score=st.session_state.get("min_score", config.MIN_SCORE),
                    )
                    response = result["response"]
                    context = result.get("context", "")

                st.write(response)
                if context:
                    with st.expander("View retrieved context"):
                        st.text(context)

        # Save assistant message with context
        st.session_state["messages"].append(
            {"role": "assistant", "content": response, "context": context}
        )


# Ingest Documents Tab
with tab_ingest:
    if not is_configured():
        st.info("Configure your API keys in the sidebar to enable ingestion.")
    
    st.markdown("Add documents to your knowledge base for retrieval.")

    ingest_type = st.radio(
        "Ingestion method",
        ["Text", "URL", "Upload", "Check Status"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if ingest_type == "Text":
        text_content = st.text_area(
            "Content", height=150,
            placeholder="Paste or type the text content you want to ingest...",
        )
        col1, col2 = st.columns(2)
        with col1:
            file_name = st.text_input("File name (optional)", placeholder="document.txt")
        with col2:
            text_metadata = metadata_popover("text", max_fields=10)

        if st.button("Ingest Text", type="primary", disabled=not text_content or not is_configured()):
            with st.spinner("Ingesting..."):
                result = get_ingester().ingest_text(
                    text_content, file_name or None, text_metadata
                )
            show_ingest_result(result)

    elif ingest_type == "URL":
        document_name = st.text_input("Document name", placeholder="Research Paper")
        file_url = st.text_input("File URL", placeholder="https://example.com/document.pdf")
        url_metadata = metadata_popover("url")

        if st.button("Ingest from URL", type="primary", disabled=not (document_name and file_url) or not is_configured()):
            with st.spinner("Ingesting..."):
                result = get_ingester().ingest_file_from_url(document_name, file_url, url_metadata)
            show_ingest_result(result)

    elif ingest_type == "Upload":
        uploaded_file = st.file_uploader("Choose a file", help="Supported formats: PDF, TXT, DOCX, and more")

        if uploaded_file:
            st.caption(f"Selected: {uploaded_file.name} ({len(uploaded_file.getbuffer()):,} bytes)")
            custom_name = st.text_input("Custom file name (optional)", placeholder=uploaded_file.name)
            upload_metadata = metadata_popover("upload")

            if st.button("Upload and Ingest", type="primary", disabled=not is_configured()):
                import tempfile, os
                with st.spinner("Uploading and ingesting..."):
                    with tempfile.NamedTemporaryFile(delete=False) as tmp:
                        tmp.write(uploaded_file.getbuffer())
                        tmp_path = tmp.name
                    try:
                        result = get_ingester().ingest_local_file(
                            tmp_path, custom_name or uploaded_file.name, upload_metadata
                        )
                        show_ingest_result(result)
                    finally:
                        os.unlink(tmp_path)

    elif ingest_type == "Check Status":
        job_id = st.text_input("Job ID", placeholder="Enter the job ID from a previous ingestion")

        if st.button("Check Status", type="primary", disabled=not job_id or not is_configured()):
            with st.spinner("Checking..."):
                result = get_ingester().get_job_status(job_id)
            (st.info if result["success"] else st.error)(result["message"])
