import streamlit as st

from config import DEFAULT_MODEL, NEO4J_DEFAULT_PASSWORD, NEO4J_DEFAULT_URI, NEO4J_DEFAULT_USER
from extractor import parse_graph_elements_via_llm
from graph_store import Neo4jGraphStore
from rag_engine import run_attributed_graph_rag


def main():
    st.set_page_config(
        page_title="VeriGraph GraphRAG with Citations 🔍",
        page_icon="🔍",
        layout="wide"
    )

    st.sidebar.divider()
    st.sidebar.header("⚙️ Graph Configuration")

    db_uri = st.sidebar.text_input("Neo4j Database URI", NEO4J_DEFAULT_URI)
    db_user = st.sidebar.text_input("Neo4j Username", NEO4J_DEFAULT_USER)
    db_pass = st.sidebar.text_input("Neo4j Password", type="password", value=NEO4J_DEFAULT_PASSWORD)
    model_options = ["llama3.2", "mistral", "phi3"]
    if DEFAULT_MODEL not in model_options:
        model_options.insert(0, DEFAULT_MODEL)
    model_choice = st.sidebar.selectbox(
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
            # Neo4j offline or unreachable — fall back to empty state silently
            pass

    # History of query runs, consumed by the Trace Logs route
    if 'trace_history' not in st.session_state:
        st.session_state.trace_history = []

    def dashboard_page():
        st.title("🔍  VeriGraph with Citations")
        st.divider()

        st.markdown("""
        This system implements **Knowledge Graph-based Retrieval-Augmented Generation (GraphRAG)**:
        - **Multi-hop reasoning**: Traverses interconnected elements across different source files.
        - **Citations Provenance**: Traces and validates every statement back to source chunks.
        - **Audit Trails**: Clear logs of traversal paths and search steps.
        """)

        tab_docs, tab_query, tab_viz = st.tabs(["📄 Document Ingestion", "❓ Ask Assistant", "🔬 Database Status"])

        with tab_docs:
            st.header("1. Extract Knowledge Graph elements")

            demo_documents = {
                "Helios Platform Specification": """
                Helios is a renewable energy analytics platform developed by Meridian Dynamics. It
                aggregates telemetry from solar farms and wind installations to forecast grid output.
                The platform was architected by Dr. Elena Vasquez, who leads the data science team
                in Austin. Helios integrates with TimescaleDB to store time-series readings and
                publishes daily capacity forecasts to regional grid operators.
                """,
                "Meridian Dynamics Company Overview": """
                Meridian Dynamics was founded in 2019 as a climate technology startup. The company
                is backed by GreenField Ventures, who led a $25 million Series B round. Their flagship
                product is Helios, an energy forecasting engine. The Helios team includes engineers
                previously at the National Renewable Energy Laboratory and Siemens, working under the
                technical leadership of Dr. Elena Vasquez.
                """
            }

            selected_template = st.selectbox("Select document template:", list(demo_documents.keys()))
            document_payload = st.text_area("Input Document Text:", demo_documents[selected_template], height=200)
            user_doc_name = st.text_input("Specify Document Label:", selected_template)

            if st.button("🔨 Extract & Ingest Nodes"):
                with st.spinner("Extracting elements and relationships using LLM..."):
                    try:
                        # Ingest extracted entities and edges to active graph
                        store = Neo4jGraphStore(db_uri, db_user, db_pass)
                        extracted_nodes, extracted_edges = parse_graph_elements_via_llm(document_payload, user_doc_name, model_choice)

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
                        st.info("Ensure Neo4j is online and active, and the Ollama model is downloaded.")

        with tab_query:
            st.header("2. Attributed Query Search")

            if not st.session_state.has_nodes:
                st.warning("Please index documents in the Knowledge Graph before asking questions.")
            else:
                st.info(f"Loaded Indexes: {', '.join(st.session_state.processed_docs)}")

            user_query = st.text_input("Enter Question:", "What are the core components of Helios and who leads the development?")

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
                                with st.expander(f"Source [{idx+1}]: {source.doc_source} (support: {source.score:.0%})"):
                                    st.write(f"**Cited Claim:** {source.claim_text}")
                                    st.write(f"**Document Name:** {source.doc_source}")
                                    st.write(f"**Context Text:** {source.reference_text}")
                                    st.write(f"**Support Score:** {source.score:.0%} keyword overlap with source")
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
