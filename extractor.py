from typing import Dict, Iterable, List, Optional, Tuple

import streamlit as st
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser

from config import get_chat_model
from embeddings import cosine_similarity, embed_text
from models import GraphEdge, GraphNode, compute_node_id

_SEMANTIC_MATCH_THRESHOLD = 0.88


def _semantic_match(candidate_embedding: List[float], pool: List[Dict]) -> Optional[str]:

    best_name, best_score = None, 0.0
    for entity in pool:
        score = cosine_similarity(candidate_embedding, entity["embedding"])
        if score > best_score:
            best_name, best_score = entity["name"], score
    return best_name if best_score >= _SEMANTIC_MATCH_THRESHOLD else None


def parse_graph_elements_via_llm(
    document_content: str,
    document_name: str,
    model_name: str = "llama3.2",
    existing_entities: Optional[Iterable[Dict]] = None
) -> Tuple[List[GraphNode], List[GraphEdge]]:
    existing_pool = list(existing_entities or [])
    extraction_prompt = f"""You are a structural parser. Extract core entities and
explicit relationships from the text.

Return strictly one JSON object in this exact format:
{{
  "nodes": [
    {{"name": "Entity Name", "type": "PERSON/ORGANIZATION/CONCEPT/TECHNOLOGY/EVENT", "summary": "Brief explanation"}}
  ],
  "edges": [
    {{"source": "Source Name", "target": "Target Name", "type": "RELATIONSHIP_TYPE", "context": "How they connect"}}
  ]
}}

Rules:
- Include the document's primary subject as a node.
- Include every explicitly named component, person, organization, technology,
  location, and customer as a node.
- Create an edge for every explicit relationship in the text.
- For a platform and its components, use HAS_COMPONENT.
- For a person who architected a platform, use ARCHITECTED.
- For a person who leads a team, use LEADS.
- For integrations, use INTEGRATES_WITH.
- The "source" and "target" of every edge must exactly match a "name" in the
  nodes list. Do not use shortened names, pronouns, or names not in nodes.
- Return JSON only: no Markdown and no explanation.

Text to analyze:
{document_content}
"""

    try:
        chat_model = get_chat_model(model_name, json_mode=True)
        completion = chat_model.invoke([HumanMessage(content=extraction_prompt)])
        parsed_data = JsonOutputParser().parse(completion.content)

        node_list = []
        local_pool: List[Dict] = []
        skipped_nodes = 0
        for nd in parsed_data.get("nodes", []):
            if not isinstance(nd, dict) or not nd.get("name"):
                skipped_nodes += 1
                continue

            raw_name = nd["name"]
            name_embedding = embed_text(raw_name)
            canonical_name = _semantic_match(name_embedding, existing_pool) or raw_name
            unique_hash = compute_node_id(canonical_name)

            node_list.append(
                GraphNode(
                    node_id=unique_hash,
                    display_name=canonical_name,
                    category=nd.get("type", "CONCEPT"),
                    summary=nd.get("summary") or nd.get("description", ""),
                    origin_document=document_name,
                    text_snippet=document_content[:250] + "..."
                )
            )
            local_pool.append({"name": canonical_name, "embedding": name_embedding})

        edge_list = []
        skipped_edges = 0
        for eg in parsed_data.get("edges", []):
            if not isinstance(eg, dict) or not eg.get("source") or not eg.get("target"):
                skipped_edges += 1
                continue

            edge_list.append(
                GraphEdge(
                    source_node=eg["source"],
                    target_node=eg["target"],
                    edge_type=eg.get("type", "RELATED_TO"),
                    context=eg.get("context", ""),
                    origin_document=document_name
                )
            )

        if skipped_nodes or skipped_edges:
            st.warning(
                f"Skipped {skipped_nodes} malformed node(s) and "
                f"{skipped_edges} malformed edge(s) from LLM output."
            )

        known_names = {n.display_name for n in node_list}
        existing_names = {e["name"] for e in existing_pool}

        def resolve_endpoint(raw_name: str) -> str:
            if raw_name in known_names:
                return raw_name
            endpoint_embedding = embed_text(raw_name)
            return (
                _semantic_match(endpoint_embedding, local_pool)
                or _semantic_match(endpoint_embedding, existing_pool)
                or raw_name
            )

        for edge in edge_list:
            edge.source_node = resolve_endpoint(edge.source_node)
            edge.target_node = resolve_endpoint(edge.target_node)
            for endpoint in (edge.source_node, edge.target_node):
                if endpoint not in known_names and endpoint not in existing_names:
                    stub_hash = compute_node_id(endpoint)
                    node_list.append(
                        GraphNode(
                            node_id=stub_hash,
                            display_name=endpoint,
                            category="CONCEPT",
                            summary=edge.context or f"Referenced in relationship with {edge.source_node}",
                            origin_document=document_name,
                            text_snippet=document_content[:500] + "..."
                        )
                    )
                    known_names.add(endpoint)

        return node_list, edge_list

    except Exception as exc:
        st.warning(f"Error extracting structural elements: {exc}")
        return [], []