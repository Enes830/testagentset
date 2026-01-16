import streamlit as st
from rag_system import RAGSystem
from document_ingester import DocumentIngester
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
    
    # Document Ingestion Button
    st.divider()
    if st.button("📄 Ingest Document", use_container_width=True):
        st.session_state["show_ingest_ui"] = True

st.title("💬 Playground")
st.caption("Agentset RAG chatbot example")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

if "show_ingest_ui" not in st.session_state:
    st.session_state["show_ingest_ui"] = False

# Show document ingestion UI if button was clicked
if st.session_state.get("show_ingest_ui", False):
    st.header("📄 Document Ingestion")
    
    ingestion_type = st.radio(
        "Select ingestion type:",
        ["Text Content", "File from URL", "Upload Local File", "Check Job Status"]
    )
    
    if ingestion_type == "Text Content":
        st.subheader("Ingest Text Content")
        text_content = st.text_area("Text content to ingest:")
        file_name = st.text_input("File name (optional):", placeholder="e.g., notes.txt")
        
        with st.expander("Add Metadata (optional)"):
            col1, col2 = st.columns(2)
            with col1:
                metadata_key = st.text_input("Metadata key")
            with col2:
                metadata_value = st.text_input("Metadata value")
        
        if st.button("📤 Ingest Text"):
            if not agentset_api_key or not agentset_namespace:
                st.error("Please add your Agentset API Key and Namespace ID.")
            elif not text_content:
                st.error("Please enter text content to ingest.")
            else:
                with st.spinner("Ingesting text..."):
                    ingester = DocumentIngester(
                        agentset_namespace_id=agentset_namespace,
                        agentset_api_token=agentset_api_key
                    )
                    metadata = {metadata_key: metadata_value} if metadata_key and metadata_value else None
                    result = ingester.ingest_text(text_content, file_name if file_name else None, metadata)
                    
                    if result["success"]:
                        st.success(result["message"])
                        st.info(f"Job ID: {result['job_id']}")
                    else:
                        st.error(result["message"])
    
    elif ingestion_type == "File from URL":
        st.subheader("Ingest File from URL")
        document_name = st.text_input("Document name:", placeholder="e.g., 'Research Paper'")
        file_url = st.text_input("File URL:", placeholder="https://example.com/document.pdf")
        
        with st.expander("Add Metadata (optional)"):
            num_metadata = st.number_input("Number of metadata fields:", min_value=0, max_value=5, value=0)
            metadata = {}
            for i in range(num_metadata):
                col1, col2 = st.columns(2)
                with col1:
                    key = st.text_input(f"Key {i+1}", key=f"meta_key_{i}")
                with col2:
                    value = st.text_input(f"Value {i+1}", key=f"meta_val_{i}")
                if key and value:
                    metadata[key] = value
        
        if st.button("📤 Ingest File"):
            if not agentset_api_key or not agentset_namespace:
                st.error("Please add your Agentset API Key and Namespace ID.")
            elif not document_name or not file_url:
                st.error("Please fill in document name and URL.")
            else:
                with st.spinner("Ingesting file..."):
                    ingester = DocumentIngester(
                        agentset_namespace_id=agentset_namespace,
                        agentset_api_token=agentset_api_key
                    )
                    result = ingester.ingest_file_from_url(
                        document_name, 
                        file_url, 
                        metadata if metadata else None
                    )
                    
                    if result["success"]:
                        st.success(result["message"])
                        st.info(f"Job ID: {result['job_id']}")
                    else:
                        st.error(result["message"])
    
    elif ingestion_type == "Upload Local File":
        st.subheader("Upload and Ingest Local File")
        uploaded_file = st.file_uploader("Choose a file to upload")
        custom_file_name = st.text_input("Custom file name (optional):")
        
        with st.expander("Add Metadata (optional)"):
            num_metadata = st.number_input("Number of metadata fields:", min_value=0, max_value=5, value=0, key="local_meta_count")
            metadata = {}
            for i in range(num_metadata):
                col1, col2 = st.columns(2)
                with col1:
                    key = st.text_input(f"Key {i+1}", key=f"local_meta_key_{i}")
                with col2:
                    value = st.text_input(f"Value {i+1}", key=f"local_meta_val_{i}")
                if key and value:
                    metadata[key] = value
        
        if st.button("📤 Upload and Ingest"):
            if not agentset_api_key or not agentset_namespace:
                st.error("Please add your Agentset API Key and Namespace ID.")
            elif not uploaded_file:
                st.error("Please select a file to upload.")
            else:
                with st.spinner("Uploading and ingesting file..."):
                    # Save uploaded file temporarily
                    import tempfile
                    import os
                    with tempfile.NamedTemporaryFile(delete=False) as tmp:
                        tmp.write(uploaded_file.getbuffer())
                        tmp_path = tmp.name
                    
                    # Use custom name or uploaded file name
                    final_file_name = custom_file_name if custom_file_name else uploaded_file.name
                    
                    # Display file info
                    with st.status("Processing file..."):
                        st.write(f"📁 File: {final_file_name}")
                        st.write(f"📊 Size: {len(uploaded_file.getbuffer())} bytes")
                        st.write(f"🏷️ MIME Type: {DocumentIngester._get_content_type(final_file_name)}")
                    
                    ingester = DocumentIngester(
                        agentset_namespace_id=agentset_namespace,
                        agentset_api_token=agentset_api_key
                    )
                    result = ingester.ingest_local_file(
                        tmp_path,
                        final_file_name,
                        metadata if metadata else None
                    )
                    
                    # Clean up temp file
                    os.unlink(tmp_path)
                    
                    if result["success"]:
                        st.success(result["message"])
                        st.info(f"Job ID: {result['job_id']}")
                    else:
                        st.error(result["message"])
    
    elif ingestion_type == "Check Job Status":
        st.subheader("Check Ingestion Job Status")
        job_id = st.text_input("Job ID:", placeholder="Enter the job ID to check")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔍 Check Status"):
                if not agentset_api_key or not agentset_namespace:
                    st.error("Please add your Agentset API Key and Namespace ID.")
                elif not job_id:
                    st.error("Please enter a job ID.")
                else:
                    with st.spinner("Checking job status..."):
                        ingester = DocumentIngester(
                            agentset_namespace_id=agentset_namespace,
                            agentset_api_token=agentset_api_key
                        )
                        result = ingester.get_job_status(job_id)
                        
                        if result["success"]:
                            st.info(result["message"])
                        else:
                            st.error(result["message"])
        
    
    st.divider()
    if st.button("✕ Close Ingestion UI"):
        st.session_state["show_ingest_ui"] = False
        st.rerun()

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