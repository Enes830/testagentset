import streamlit as st
from rag_system import RAGSystem
from document_ingester import DocumentIngester
import config

# Page configuration
st.set_page_config(
    page_title="RAG Playground",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional styling - clean chat avatars without emojis
st.markdown(
    """
<style>
    /* Replace default chat avatars with simple initials */
    .stChatMessage [data-testid="chatAvatarIcon-user"] {
        display: none;
    }
    .stChatMessage [data-testid="chatAvatarIcon-assistant"] {
        display: none;
    }
    
    /* Style for user messages */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageAvatarContainer"] {
        background: #e5e5e5;
        border-radius: 50%;
        width: 2rem;
        height: 2rem;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageAvatarContainer"]::after {
        content: "U";
        font-weight: 600;
        font-size: 0.875rem;
        color: #666;
    }
    
    /* Style for assistant messages */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageAvatarContainer"] {
        background: #333;
        border-radius: 50%;
        width: 2rem;
        height: 2rem;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageAvatarContainer"]::after {
        content: "A";
        font-weight: 600;
        font-size: 0.875rem;
        color: #fff;
    }
    
    /* Sidebar section headers */
    .sidebar-section { 
        font-size: 0.75rem; 
        font-weight: 600; 
        color: #666; 
        text-transform: uppercase; 
        letter-spacing: 0.5px;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = "chat"


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
    )


# Sidebar - Configuration
with st.sidebar:
    st.title("RAG Playground")

    st.markdown(
        '<p class="sidebar-section">API Configuration</p>', unsafe_allow_html=True
    )

    openai_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=config.OPENAI_API_KEY or "",
        placeholder="sk-...",
        help="Required for generating responses",
    )
    if openai_key:
        st.session_state["openai_api_key"] = openai_key

    agentset_key = st.text_input(
        "Agentset API Key",
        type="password",
        value=config.AGENTSET_API_KEY or "",
        placeholder="Your Agentset API key",
        help="Required for document retrieval",
    )
    if agentset_key:
        st.session_state["agentset_api_key"] = agentset_key

    namespace = st.text_input(
        "Namespace ID",
        value=config.AGENTSET_NAMESPACE_ID or "",
        placeholder="ns_...",
        help="Your Agentset namespace identifier",
    )
    if namespace:
        st.session_state["agentset_namespace"] = namespace

    st.markdown('<p class="sidebar-section">RAG Settings</p>', unsafe_allow_html=True)

    with st.expander("Advanced Settings"):
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

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        if not is_configured():
            st.error("Please configure your API keys in the sidebar to continue.")
            st.stop()

        # Add user message
        st.session_state["messages"].append({"role": "user", "content": prompt})
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
        st.info("Configure your API keys in the sidebar to ingest documents.")
    else:
        st.markdown("Add documents to your knowledge base for retrieval.")

        ingest_type = st.radio(
            "Ingestion method",
            ["Text", "URL", "Upload", "Check Status"],
            horizontal=True,
            label_visibility="collapsed",
        )

        st.divider()

        if ingest_type == "Text":
            st.subheader("Ingest Text Content")

            text_content = st.text_area(
                "Content",
                height=150,
                placeholder="Paste or type the text content you want to ingest...",
            )

            col1, col2 = st.columns(2)
            with col1:
                file_name = st.text_input(
                    "File name (optional)", placeholder="document.txt"
                )
            with col2:
                with st.popover("Add metadata"):
                    meta_key = st.text_input("Key", key="text_meta_key")
                    meta_value = st.text_input("Value", key="text_meta_value")

            if st.button("Ingest Text", type="primary", disabled=not text_content):
                with st.spinner("Ingesting..."):
                    ingester = get_ingester()
                    metadata = (
                        {meta_key: meta_value} if meta_key and meta_value else None
                    )
                    result = ingester.ingest_text(
                        text_content, file_name if file_name else None, metadata
                    )

                if result["success"]:
                    st.success(f"Ingestion started. Job ID: `{result['job_id']}`")
                else:
                    st.error(result["message"])

        elif ingest_type == "URL":
            st.subheader("Ingest File from URL")

            document_name = st.text_input("Document name", placeholder="Research Paper")
            file_url = st.text_input(
                "File URL", placeholder="https://example.com/document.pdf"
            )

            with st.popover("Add metadata"):
                num_meta = st.number_input(
                    "Number of fields", 0, 5, 0, key="url_meta_count"
                )
                url_metadata = {}
                for i in range(int(num_meta)):
                    c1, c2 = st.columns(2)
                    with c1:
                        k = st.text_input(f"Key {i + 1}", key=f"url_k_{i}")
                    with c2:
                        v = st.text_input(f"Value {i + 1}", key=f"url_v_{i}")
                    if k and v:
                        url_metadata[k] = v

            can_submit = document_name and file_url
            if st.button("Ingest from URL", type="primary", disabled=not can_submit):
                with st.spinner("Ingesting..."):
                    ingester = get_ingester()
                    result = ingester.ingest_file_from_url(
                        document_name, file_url, url_metadata if url_metadata else None
                    )

                if result["success"]:
                    st.success(f"Ingestion started. Job ID: `{result['job_id']}`")
                else:
                    st.error(result["message"])

        elif ingest_type == "Upload":
            st.subheader("Upload Local File")

            uploaded_file = st.file_uploader(
                "Choose a file", help="Supported formats: PDF, TXT, DOCX, and more"
            )

            if uploaded_file:
                # Show file info
                st.caption(
                    f"Selected: {uploaded_file.name} ({len(uploaded_file.getbuffer()):,} bytes)"
                )

                custom_name = st.text_input(
                    "Custom file name (optional)", placeholder=uploaded_file.name
                )

                with st.popover("Add metadata"):
                    num_meta = st.number_input(
                        "Number of fields", 0, 5, 0, key="upload_meta_count"
                    )
                    upload_metadata = {}
                    for i in range(int(num_meta)):
                        c1, c2 = st.columns(2)
                        with c1:
                            k = st.text_input(f"Key {i + 1}", key=f"upload_k_{i}")
                        with c2:
                            v = st.text_input(f"Value {i + 1}", key=f"upload_v_{i}")
                        if k and v:
                            upload_metadata[k] = v

                if st.button("Upload and Ingest", type="primary"):
                    import tempfile
                    import os

                    with st.spinner("Uploading and ingesting..."):
                        # Save to temp file
                        with tempfile.NamedTemporaryFile(delete=False) as tmp:
                            tmp.write(uploaded_file.getbuffer())
                            tmp_path = tmp.name

                        try:
                            final_name = (
                                custom_name if custom_name else uploaded_file.name
                            )
                            ingester = get_ingester()
                            result = ingester.ingest_local_file(
                                tmp_path,
                                final_name,
                                upload_metadata if upload_metadata else None,
                            )

                            if result["success"]:
                                st.success(
                                    f"Ingestion started. Job ID: `{result['job_id']}`"
                                )
                            else:
                                st.error(result["message"])
                        finally:
                            os.unlink(tmp_path)

        elif ingest_type == "Check Status":
            st.subheader("Check Job Status")

            job_id = st.text_input(
                "Job ID", placeholder="Enter the job ID from a previous ingestion"
            )

            if st.button("Check Status", type="primary", disabled=not job_id):
                with st.spinner("Checking..."):
                    ingester = get_ingester()
                    result = ingester.get_job_status(job_id)

                if result["success"]:
                    st.info(result["message"])
                else:
                    st.error(result["message"])
