import hashlib
from dataclasses import dataclass
from typing import List


def compute_node_id(name: str) -> str:
    return hashlib.md5(name.strip().lower().encode()).hexdigest()[:12]


@dataclass
class GraphNode:
    node_id: str            
    display_name: str      
    category: str           
    summary: str            
    origin_document: str    
    text_snippet: str      


@dataclass
class GraphEdge:
    source_node: str       
    target_node: str      
    edge_type: str          
    context: str          
    origin_document: str    


@dataclass
class SourceAttribution:
    claim_text: str         
    doc_source: str         
    reference_text: str     
    score: float           
    path_trace: List[str]   


@dataclass
class AttributedAnswer:

    content: str       
    sources: List[SourceAttribution]
    execution_steps: List[str]      