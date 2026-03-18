import streamlit as st

from config import DEFAULT_MODEL, NEO4J_DEFAULT_PASSWORD, NEO4J_DEFAULT_URI, NEO4J_DEFAULT_USER
from extractor import parse_graph_elements_via_llm
from graph_store import Neo4jGraphStore
from pdf_ingest import extract_pdf_chunks
from rag_engine import run_attributed_graph_rag


_STATUS_COLORS = {"good": "#0ca30c", "warning": "#fab219", "critical": "#d03b3b"}


def _support_status(score: float):
    if score >= 0.7:
        return "🟢", "Strong support", _STATUS_COLORS["good"]
    if score >= 0.4:
        return "🟡", "Partial support", _STATUS_COLORS["warning"]
    return "🔴", "Weak support", _STATUS_COLORS["critical"]


def _inject_custom_css():
    st.markdown(
        """
        <style>
        html, body, [class*="css"]  {
            font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
        }
        [data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
        [data-testid="stMetricLabel"] { color: #52514e; }
        hr { border-color: rgba(11, 11, 11, 0.10); }
        .vg-header {
            padding: 1.25rem 1.5rem;
            border-radius: 10px;
            background: #fcfcfb;
            border: 1px solid rgba(11, 11, 11, 0.10);
            margin-bottom: 1.25rem;
        }
        .vg-header h1 { margin: 0 0 0.25rem 0; font-size: 1.6rem; }
        .vg-header p { margin: 0; color: #52514e; }
        .vg-chip-row { display: flex; gap: 0.5rem; margin-top: 0.75rem; flex-wrap: wrap; }
        .vg-chip {
            font-size: 0.8rem;
            padding: 0.25rem 0.65rem;
            border-radius: 999px;
            background: #f0efec;
            color: #52514e;
            border: 1px solid rgba(11, 11, 11, 0.10);
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def main():
    st.set_page_config(
        page_title="VeriGraph — GraphRAG with Citations",
        page_icon="🔍",
        layout="wide"
    )
    _inject_custom_css()

    st.sidebar.markdown("## 🔍 VeriGraph")
    st.sidebar.caption("GraphRAG with verifiable citations")
    st.sidebar.divider()

    with st.sidebar.container(border=True):
        st.markdown("**Graph Configuration**")
        db_uri = st.text_input("Neo4j Database URI", NEO4J_DEFAULT_URI)
        db_user = st.text_input("Neo4j Username", NEO4J_DEFAULT_USER)
        db_pass = st.text_input("Neo4j Password", type="password", value=NEO4J_DEFAULT_PASSWORD)
        model_options = ["llama3.2", "mistral", "phi3"]
        if DEFAULT_MODEL not in model_options:
            model_options.insert(0, DEFAULT_MODEL)
        model_choice = st.selectbox(
            "Local Inference Model", model_options,
            index=model_options.index(DEFAULT_MODEL)
        )

    if 'has_nodes' not in st.session_state:
        st.session_state.has_nodes = False
        st.session_state.processed_docs = []
        try:
            store = Neo4jGraphStore(db_uri, db_user, db_pass)
            persisted_docs = store.list_ingested_documents()
            store.shutdown()
            if persisted_docs:
                st.session_state.has_nodes = True
                st.session_state.processed_docs = persisted_docs
        except Exception:
            pass

    if 'trace_history' not in st.session_state:
        st.session_state.trace_history = []

    def dashboard_page():
        st.markdown(
            """
            <div class="vg-header">
              <h1>🔍 VeriGraph</h1>
              <p>Knowledge Graph-based GraphRAG: multi-hop reasoning with traceable, verifiable citations.</p>
              <div class="vg-chip-row">
                <span class="vg-chip">Multi-hop reasoning</span>
                <span class="vg-chip">Citation provenance</span>
                <span class="vg-chip">Audit trails</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        tab_docs, tab_query, tab_viz = st.tabs(["📄 Document Ingestion", "❓ Ask Assistant", "🔬 Database Status"])

        with tab_docs:
            st.header("1. Extract Knowledge Graph elements")

            document_payload = st.text_area("Input Document Text:", "", height=200, placeholder="Paste any document text here...")
            user_doc_name = st.text_input("Specify Document Label:", "")

            if st.button("🔨 Extract & Ingest Nodes"):
                if not document_payload.strip():
                    st.warning("Please enter some document text before extracting.")
                elif not user_doc_name.strip():
                    st.warning("Please specify a document label.")
                else:
                    with st.spinner("Extracting elements and relationships using LLM..."):
                        try:
                            store = Neo4jGraphStore(db_uri, db_user, db_pass)
                            existing_entities = store.get_all_entity_name_embeddings()
                            extracted_nodes, extracted_edges = parse_graph_elements_via_llm(
                                document_payload, user_doc_name, model_choice, existing_entities=existing_entities
                            )

                            for node in extracted_nodes:
                                store.insert_node(node)

                            for edge in extracted_edges:
                                store.insert_edge(edge)

                            store.shutdown()

                            st.success(f"Successfully loaded {len(extracted_nodes)} entities and {len(extracted_edges)} relationships.")

                            with st.expander("Ingested Entities Details"):
                                for en in extracted_nodes:
                                    st.write(f"🔹 **{en.display_name}** ({en.category}): {en.summary}")

                            with st.expander("Ingested Relationships Details"):
                                for rel in extracted_edges:
                                    st.write(f"🔗 {rel.source_node} --[{rel.edge_type}]--> {rel.target_node}: {rel.context}")

                            st.session_state.has_nodes = True
                            if user_doc_name not in st.session_state.processed_docs:
                                st.session_state.processed_docs.append(user_doc_name)

                        except Exception as ex:
                            st.error(f"Failed to ingest: {ex}")
                            st.info("Ensure Neo4j is online and active, and both the chat and embedding Ollama models (ollama pull llama3.2, ollama pull nomic-embed-text) are downloaded.")

            st.divider()
            st.subheader("Or upload a PDF")

            uploaded_pdf = st.file_uploader("Upload a PDF document", type=["pdf"])

            if uploaded_pdf is not None and st.button("📄 Extract & Ingest PDF"):
                try:
                    pdf_chunks = extract_pdf_chunks(uploaded_pdf)
                except Exception as ex:
                    st.error(f"Failed to read PDF: {ex}")
                    pdf_chunks = []

                if uploaded_pdf is not None and not pdf_chunks:
                    st.warning("No extractable text found in this PDF (it may be scanned/image-only).")

                if pdf_chunks:
                    progress_bar = st.progress(0.0, text=f"Processing 0/{len(pdf_chunks)} chunks...")
                    total_nodes, total_edges = 0, 0
                    store = Neo4jGraphStore(db_uri, db_user, db_pass)
                    try:
                        for idx, (chunk_label, chunk_text) in enumerate(pdf_chunks):
                            doc_label = f"{uploaded_pdf.name} ({chunk_label})"
                            # Re-fetched each chunk so entities inserted by
                            # earlier chunks (this upload or prior ones) are
                            # available for semantic resolution.
                            existing_entities = store.get_all_entity_name_embeddings()
                            extracted_nodes, extracted_edges = parse_graph_elements_via_llm(
                                chunk_text, doc_label, model_choice, existing_entities=existing_entities
                            )

                            for node in extracted_nodes:
                                store.insert_node(node)
                            for edge in extracted_edges:
                                store.insert_edge(edge)

                            total_nodes += len(extracted_nodes)
                            total_edges += len(extracted_edges)
                            progress_bar.progress(
                                (idx + 1) / len(pdf_chunks),
                                text=f"Processing {idx + 1}/{len(pdf_chunks)} chunks..."
                            )

                        st.success(
                            f"Ingested {uploaded_pdf.name}: {len(pdf_chunks)} chunks, "
                            f"{total_nodes} entity mentions, {total_edges} relationships."
                        )
                        st.session_state.has_nodes = True
                        if uploaded_pdf.name not in st.session_state.processed_docs:
                            st.session_state.processed_docs.append(uploaded_pdf.name)
                    except Exception as ex:
                        st.error(f"Failed to ingest PDF: {ex}")
                        st.info("Ensure Neo4j is online and active, and both the chat and embedding Ollama models (ollama pull llama3.2, ollama pull nomic-embed-text) are downloaded.")
                    finally:
                        store.shutdown()

        with tab_query:
            st.header("2. Attributed Query Search")

            if not st.session_state.has_nodes:
                st.warning("Please index documents in the Knowledge Graph before asking questions.")
            else:
                st.info(f"Loaded Indexes: {', '.join(st.session_state.processed_docs)}")

            user_query = st.text_input("Enter Question:", "", placeholder="Ask a question about your ingested documents...")

            if st.button("🔎 Search Attributed Answer"):
                with st.spinner("Traversing graph and resolving query..."):
                    try:
                        # Execute multi-hop Graph-RAG query
                        store = Neo4jGraphStore(db_uri, db_user, db_pass)
                        rag_result = run_attributed_graph_rag(user_query, store, model_choice)
                        store.shutdown()

                        # Store trace for the dedicated Trace Logs route
                        st.session_state.trace_history.append({
                            "query": user_query,
                            "steps": rag_result.execution_steps
                        })
                        st.caption("🔬 Path traversals & trace logs are on the **Trace Logs** page (sidebar).")

                        st.subheader("💬 Generated Answer")
                        st.write(rag_result.content)

                        st.subheader("📚 Source Attributions")
                        if rag_result.sources:
                            for idx, source in enumerate(rag_result.sources):
                                icon, status_label, color = _support_status(source.score)
                                with st.expander(f"{icon} Source [{idx+1}]: {source.doc_source} — {source.score:.0%} support"):
                                    st.write(f"**Cited Claim:** {source.claim_text}")
                                    st.write(f"**Document Name:** {source.doc_source}")
                                    st.write(f"**Context Text:** {source.reference_text}")
                                    st.markdown(
                                        f"**Support Score:** {icon} "
                                        f"<span style='color:{color}; font-weight:600'>{source.score:.0%} — {status_label}</span> "
                                        f"(keyword overlap + semantic similarity with source)",
                                        unsafe_allow_html=True
                                    )
                                    st.write(f"**Reasoning Trace:** {' ➔ '.join(source.path_trace)}")
                        else:
                            st.info("No specific verifiable sources matched this answer.")

                    except Exception as ex:
                        st.error(f"Error querying: {ex}")

        with tab_viz:
            st.header("3. Graph DB Statistics")

            if st.button("📊 Query Database Sizes"):
                try:
                    # Query total size of Nodes and relations in active store
                    store = Neo4jGraphStore(db_uri, db_user, db_pass)
                    with store.db_driver.session() as sess:
                        total_nodes = sess.run("MATCH (n) RETURN count(n) as cnt").single()['cnt']
                        total_edges = sess.run("MATCH ()-[r]->() RETURN count(r) as cnt").single()['cnt']

                    with st.container(border=True):
                        col_left, col_right = st.columns(2)
                        col_left.metric("Total Nodes", total_nodes)
                        col_right.metric("Total Edges", total_edges)
                    store.shutdown()
                except Exception as ex:
                    st.error(f"Could not connect: {ex}")

            if st.button("🗑️ Clear Database Graph"):
                try:
                    # Wipe Neo4j instance
                    store = Neo4jGraphStore(db_uri, db_user, db_pass)
                    store.reset_database()
                    store.shutdown()
                    st.session_state.has_nodes = False
                    st.session_state.processed_docs = []
                    st.success("Successfully reset all database records.")
                except Exception as ex:
                    st.error(f"Failed to clear database: {ex}")

    def trace_logs_page():
        st.title("🔬 Path Traversals & Trace Logs")
        st.divider()

        if not st.session_state.trace_history:
            st.info("No trace logs yet. Run a query from the **Ask Assistant** tab on the Dashboard page.")
            return

        st.caption(f"{len(st.session_state.trace_history)} query run(s) this session — most recent first.")

        if st.button("🗑️ Clear Trace History"):
            st.session_state.trace_history = []
            st.rerun()

        # Most recent run expanded, older runs collapsed
        for run_number, run in enumerate(reversed(st.session_state.trace_history)):
            original_index = len(st.session_state.trace_history) - run_number
            with st.expander(f"Run #{original_index}: {run['query']}", expanded=(run_number == 0)):
                for log_step in run["steps"]:
                    st.text(log_step)

    pages = st.navigation([
        st.Page(dashboard_page, title="Dashboard", icon="🔍", url_path="dashboard", default=True),
        st.Page(trace_logs_page, title="Trace Logs", icon="🔬", url_path="trace-logs"),
    ])
    pages.run()
