import hashlib
import json
from typing import List, Tuple

import streamlit as st

from config import inference_client
from models import GraphEdge, GraphNode


def parse_graph_elements_via_llm(
    document_content: str,
    document_name: str,
    model_name: str = "llama3.2"
) -> Tuple[List[GraphNode], List[GraphEdge]]:

    # Structural extraction prompt formatting
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
        completion = inference_client.chat(
            model=model_name,
            messages=[{"role": "user", "content": extraction_prompt}],
            format="json"
        )

        parsed_data = json.loads(completion["message"]["content"])

        node_list = []
        skipped_nodes = 0
        for nd in parsed_data.get("nodes", []):
            # LLM output is untrusted: fields may be missing or renamed.
            # Require a name; default everything else.
            if not isinstance(nd, dict) or not nd.get("name"):
                skipped_nodes += 1
                continue

            # Generate stable md5 hash as node_id
            unique_hash = hashlib.md5(
                f"{nd['name']}_{document_name}".encode()
            ).hexdigest()[:12]

            node_list.append(
                GraphNode(
                    node_id=unique_hash,
                    display_name=nd["name"],
                    category=nd.get("type", "CONCEPT"),
                    summary=nd.get("summary") or nd.get("description", ""),
                    origin_document=document_name,
                    text_snippet=document_content[:250] + "..."
                )
            )

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

        def resolve_endpoint(raw_name: str) -> str:
            if raw_name in known_names:
                return raw_name
            # Case-insensitive exact match
            for name in known_names:
                if name.lower() == raw_name.lower():
                    return name
            for name in known_names:
                if raw_name.lower() in name.lower() or name.lower() in raw_name.lower():
                    return name
            return raw_name  # No match: keep as-is, stub node created below

        for edge in edge_list:
            edge.source_node = resolve_endpoint(edge.source_node)
            edge.target_node = resolve_endpoint(edge.target_node)

            # Create stub nodes for endpoints the LLM never emitted as nodes
            for endpoint in (edge.source_node, edge.target_node):
                if endpoint not in known_names:
                    stub_hash = hashlib.md5(
                        f"{endpoint}_{document_name}".encode()
                    ).hexdigest()[:12]
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