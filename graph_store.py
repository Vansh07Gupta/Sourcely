from typing import Dict, List
from neo4j import GraphDatabase
from embeddings import embed_text
from models import GraphEdge, GraphNode, compute_node_id


class Neo4jGraphStore:

    def __init__(self, connection_url: str, auth_user: str, auth_pass: str):
        self.db_driver = GraphDatabase.driver(connection_url, auth=(auth_user, auth_pass))

    def shutdown(self):
        self.db_driver.close()

    def reset_database(self):
        with self.db_driver.session() as db_session:
            db_session.run("MATCH (n) DETACH DELETE n")

    def insert_node(self, node: GraphNode):
        embedding = embed_text(f"{node.display_name}. {node.summary}")
        name_embedding = embed_text(node.display_name)
        with self.db_driver.session() as db_session:
            db_session.run(
                """
                MERGE (e:GraphNode {id: $node_id})
                ON CREATE SET e.name = $display_name,
                              e.category = $category,
                              e.summary = $summary,
                              e.embedding = $embedding,
                              e.name_embedding = $name_embedding,
                              e.origin_documents = [$origin_document],
                              e.text_snippets = [$text_snippet]
                ON MATCH SET e.summary = $summary,
                             e.embedding = $embedding,
                             e.name_embedding = $name_embedding,
                             e.origin_documents = CASE
                                 WHEN $origin_document IN e.origin_documents THEN e.origin_documents
                                 ELSE e.origin_documents + $origin_document END,
                             e.text_snippets = CASE
                                 WHEN $text_snippet IN e.text_snippets THEN e.text_snippets
                                 ELSE e.text_snippets + $text_snippet END
                """,
                node_id=node.node_id,
                display_name=node.display_name,
                category=node.category,
                summary=node.summary,
                embedding=embedding,
                name_embedding=name_embedding,
                origin_document=node.origin_document,
                text_snippet=node.text_snippet
            )

    def insert_edge(self, edge: GraphEdge):
        source_id = compute_node_id(edge.source_node)
        target_id = compute_node_id(edge.target_node)
        with self.db_driver.session() as db_session:
            db_session.run(
                """
                MATCH (x:GraphNode {id: $source_id})
                MATCH (y:GraphNode {id: $target_id})
                MERGE (x)-[r:LINKED_TO {type: $edge_type}]->(y)
                SET r.context = $context,
                    r.origin_document = $origin_document
                """,
                source_id=source_id,
                target_id=target_id,
                edge_type=edge.edge_type,
                context=edge.context,
                origin_document=edge.origin_document
            )

    def fetch_hops_context(self, start_node_id: str, depth: int = 2) -> List[Dict]:
        if not isinstance(depth, int) or isinstance(depth, bool):
            raise ValueError(f"depth must be an integer, got {type(depth).__name__}")
        depth = max(1, min(depth, 5))
        with self.db_driver.session() as db_session:
            query_string = f"""
            MATCH (root:GraphNode {{id: $node_id}})
            MATCH path = (root)-[*1..{depth}]-(neighbor:GraphNode)
            RETURN neighbor.name as name,
                   neighbor.summary as summary,
                   neighbor.embedding as embedding,
                   neighbor.origin_documents as sources,
                   neighbor.text_snippets as chunks,
                   [rel in relationships(path) | rel.context] as path_contexts
            LIMIT 25
            """
            cursor = db_session.run(query_string, node_id=start_node_id)
            return [dict(record) for record in cursor]

    def get_all_entity_name_embeddings(self) -> List[Dict]:
        """Name + name-only embedding for every entity, so the extractor can
        semantically resolve a newly extracted name against the graph
        instead of substring-matching raw strings."""
        with self.db_driver.session() as db_session:
            cursor = db_session.run(
                """
                MATCH (n:GraphNode)
                WHERE n.name_embedding IS NOT NULL
                RETURN n.name as name, n.name_embedding as embedding
                """
            )
            return [dict(record) for record in cursor]

    def list_ingested_documents(self) -> List[str]:
        with self.db_driver.session() as db_session:
            cursor = db_session.run(
                """
                MATCH (n:GraphNode)
                UNWIND n.origin_documents as doc
                RETURN DISTINCT doc
                """
            )
            return [record['doc'] for record in cursor]

    def get_all_node_embeddings(self) -> List[Dict]:
        with self.db_driver.session() as db_session:
            cursor = db_session.run(
                """
                MATCH (n:GraphNode)
                WHERE n.embedding IS NOT NULL
                RETURN n.id as node_id,
                       n.name as name,
                       n.summary as summary,
                       n.embedding as embedding,
                       n.origin_documents as sources,
                       n.text_snippets as chunks
                """
            )
            return [dict(record) for record in cursor]
