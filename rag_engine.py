import re
from typing import Dict

from config import inference_client
from graph_store import Neo4jGraphStore
from models import AttributedAnswer, SourceAttribution

# Common words ignored when scoring claim/source overlap
_STOPWORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'to', 'of', 'in', 'on', 'at', 'by', 'for', 'with', 'and', 'or', 'as',
    'it', 'its', 'this', 'that', 'these', 'those', 'from', 'has', 'have',
    'had', 'who', 'which', 'their', 'they', 'led', 'uses', 'using', 'also'
}


def _extract_claim_sentences(answer_text: str) -> Dict[str, str]:
    """Maps each citation key (e.g. '[1]') to the sentence that carries it."""
    claim_map = {}
    # Split answer into sentences, keeping citation markers attached
    sentences = re.split(r'(?<=[.!?])\s+', answer_text)
    for sentence in sentences:
        for citation_num in re.findall(r'\[(\d+)\]', sentence):
            citation_key = f"[{citation_num}]"
            # Keep the first sentence found for each citation
            claim_map.setdefault(citation_key, sentence.strip())
    return claim_map


def _score_claim_support(claim_text: str, source_text: str) -> float:
    """Scores how well the source text supports the claim via keyword overlap."""
    def significant_words(text: str) -> set:
        words = set(re.findall(r'[a-z0-9]+', text.lower()))
        return {w for w in words if len(w) > 2 and w not in _STOPWORDS}

    claim_words = significant_words(re.sub(r'\[\d+\]', '', claim_text))
    source_words = significant_words(source_text)
    if not claim_words:
        return 0.0
    return round(len(claim_words & source_words) / len(claim_words), 2)


def _normalize_citations(answer_text: str, valid_refs: set) -> str:
    def replace_if_valid(match: re.Match) -> str:
        number = match.group('num')
        if f"[{number}]" in valid_refs:
            return f"[{number}]"
        return match.group(0)  # Not a known source: leave as-is

    # Variants, most specific first:
    #   (N 13) / (Source 13) / (Ref 13) / (Citation 13) / [Source 13]
    #   Source 13 (bare, word-prefixed only — bare (13) needs the prefix or
    #   parens to avoid mangling ordinary numbers in prose)
    patterns = [
        r'[\(\[]\s*(?:N|no\.?|source|ref(?:erence)?|citation)\s*(?P<num>\d+)\s*[\)\]]',
        r'\b(?:Source|Ref(?:erence)?|Citation)\s+(?P<num>\d+)\b',
        r'\((?P<num>\d+)\)',
    ]
    for pattern in patterns:
        answer_text = re.sub(pattern, replace_if_valid, answer_text, flags=re.IGNORECASE)
    return answer_text


def run_attributed_graph_rag(
    user_query: str,
    graph_store: Neo4jGraphStore,
    model_name: str = "llama3.2"
) -> AttributedAnswer:

    trace_steps = []
    attributions = []

    trace_steps.append(f"🔎 Querying Neo4j Graph Index for: '{user_query}'")
    query_terms = {
        word.lower()
        for word in re.findall(r"[A-Za-z0-9]+", user_query)
        if len(word) > 2 and word.lower() not in _STOPWORDS
    }
    entity_terms = re.findall(
        r"\b(?:[A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+)*)\b",
        user_query
    )

    initial_matches = []
    seen_node_names = set()

    for term in [*entity_terms, *query_terms]:
        for match in graph_store.query_nodes_by_text(term):
            if match["name"] not in seen_node_names:
                initial_matches.append(match)
                seen_node_names.add(match["name"])

    if not initial_matches:
        return AttributedAnswer(
            content="No matching nodes found in the knowledge graph to answer your query.",
            sources=[],
            execution_steps=trace_steps
        )

    trace_steps.append(f"📊 Identified {len(initial_matches)} potential starting points")

    # Step 2: Multi-hop graph expansion
    combined_contexts = []
    for match in initial_matches[:3]:
        trace_steps.append(f"🔗 Traversing paths from node: {match['name']}")
        neighbors = graph_store.fetch_hops_context(match['name'], depth=2)

        for nb in neighbors:
            combined_contexts.append({
                "node_name": nb['name'],
                "summary": nb['summary'],
                "document": nb['source'],
                "chunk": nb['chunk'],
                "relation_context": nb.get('path_contexts', [])
            })
            trace_steps.append(f"  → Found connected element: {nb['name']}")

    # Compile prompt content context with indexing map
    context_builder = []
    provenance_reference = {}

    for idx, ctx in enumerate(combined_contexts):
        ref_id = f"[{idx+1}]"
        context_builder.append(f"{ref_id} Entity: {ctx['node_name']} - {ctx['summary']}")
        provenance_reference[ref_id] = {
            "file": ctx['document'],
            "snippet": ctx['chunk'],
            "term": ctx['node_name'],
            "summary": ctx['summary']
        }

    compiled_context = "\n".join(context_builder)
    trace_steps.append(f"📝 Compiled context from {len(context_builder)} nodes")

    # Step 3: Run LLM query answering
    prompt = f"""You are a query processor. Answer the user question based on the graph relationships provided below.
    IMPORTANT: You must cite the specific sources using [N] notation for every claim you make.

    RELATIONSHIP CONTEXT:
    {compiled_context}

    QUESTION: {user_query}

    Citation rules:
    - Cite with square brackets and the source number only: [1], [2].
    - Do NOT use (1), (N 1), (Source 1), "Table 1", "Figure 1", or footnotes.
    - Only cite numbers that appear in the RELATIONSHIP CONTEXT above.
    - Every factual sentence must end with at least one citation.

    Example: "Helios was architected by Dr. Elena Vasquez [1], and integrates with TimescaleDB [3]."

    Generate a detailed response backed by inline citations.
    """

    try:
        response = inference_client.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        answer_text = response['message']['content']
        trace_steps.append("✅ Generated attributed text from local LLM")

        # Step 4: Normalize drifted citation formats, then trace and verify
        # inline citations against source snippets
        normalized_text = _normalize_citations(answer_text, set(provenance_reference))
        if normalized_text != answer_text:
            trace_steps.append("🔧 Normalized non-standard citation formats to [N] notation")
            answer_text = normalized_text

        claim_sentences = _extract_claim_sentences(answer_text)
        citations_found = re.findall(r'\[(\d+)\]', answer_text)
        for citation in sorted(set(citations_found), key=int):
            citation_key = f"[{citation}]"
            if citation_key in provenance_reference:
                ref_item = provenance_reference[citation_key]
                claim = claim_sentences.get(citation_key, f"Claim linked to citation {citation_key}")
                # Score the claim against both the summary context and raw snippet
                support_material = f"{ref_item['term']} {ref_item['summary']} {ref_item['snippet']}"
                support_score = _score_claim_support(claim, support_material)
                attributions.append(SourceAttribution(
                    claim_text=claim,
                    doc_source=ref_item['file'],
                    reference_text=ref_item['snippet'],
                    score=support_score,
                    path_trace=[f"GraphNode: {ref_item['term']}"]
                ))
            else:
                # LLM cited a reference number that was never provided
                trace_steps.append(f"⚠️ Hallucinated citation detected: {citation_key} has no source")

        trace_steps.append(f"🔒 Verified {len(attributions)} unique citations")

        return AttributedAnswer(
            content=answer_text,
            sources=attributions,
            execution_steps=trace_steps
        )

    except Exception as error:
        return AttributedAnswer(
            content=f"An error occurred during text generation: {error}",
            sources=[],
            execution_steps=trace_steps
        )
