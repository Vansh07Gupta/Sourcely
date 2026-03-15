from typing import Dict, List
from neo4j import GraphDatabase
from models import GraphEdge, GraphNode


class Neo4jGraphStore:

    def __init__(self, connection_url: str, auth_user: str, auth_pass: str):
        self.db_driver = GraphDatabase.driver(connection_url, auth=(auth_user, auth_pass))

    def shutdown(self):
        self.db_driver.close()

    def reset_database(self):
        with self.db_driver.session() as db_session:
            db_session.run("MATCH (n) DETACH DELETE n")

    def insert_node(self, node: GraphNode):
        with self.db_driver.session() as db_session:
            db_session.run(
                """
                MERGE (e:GraphNode {id: $node_id})
                SET e.name = $display_name,
                    e.category = $category,
                    e.summary = $summary,
                    e.origin_document = $origin_document,
                    e.text_snippet = $text_snippet
                """,
                node_id=node.node_id,
                display_name=node.display_name,
                category=node.category,
                summary=node.summary,
                origin_document=node.origin_document,
                text_snippet=node.text_snippet
            )

    def insert_edge(self, edge: GraphEdge):
        with self.db_driver.session() as db_session:
            db_session.run(
                """
                MATCH (x:GraphNode {name: $source})
                MATCH (y:GraphNode {name: $target})
                MERGE (x)-[r:LINKED_TO {type: $edge_type}]->(y)
                SET r.context = $context,
                    r.origin_document = $origin_document
                """,
                source=edge.source_node,
                target=edge.target_node,
                edge_type=edge.edge_type,
                context=edge.context,
                origin_document=edge.origin_document
            )

    def fetch_hops_context(self, start_node: str, depth: int = 2) -> List[Dict]:
        if not isinstance(depth, int) or isinstance(depth, bool):
            raise ValueError(f"depth must be an integer, got {type(depth).__name__}")
        depth = max(1, min(depth, 5))
        with self.db_driver.session() as db_session:
            query_string = f"""
            MATCH path = (root:GraphNode)-[*1..{depth}]-(neighbor:GraphNode)
            WHERE toLower(root.name) CONTAINS toLower($query_name)
               OR toLower(root.summary) CONTAINS toLower($query_name)
            RETURN neighbor.name as name,
                   neighbor.summary as summary,
                   neighbor.origin_document as source,
                   neighbor.text_snippet as chunk,
                   [rel in relationships(path) | rel.context] as path_contexts
            LIMIT 25
            """
            cursor = db_session.run(query_string, query_name=start_node)
            return [dict(record) for record in cursor]

    def list_ingested_documents(self) -> List[str]:
        with self.db_driver.session() as db_session:
            cursor = db_session.run(
                """
                MATCH (n:GraphNode)
                WHERE n.origin_document IS NOT NULL
                RETURN DISTINCT n.origin_document as doc
                """
            )
            return [record['doc'] for record in cursor]

    def query_nodes_by_text(self, text_query: str) -> List[Dict]:
        with self.db_driver.session() as db_session:
            # Matches node names or summaries containing search term
            cursor = db_session.run(
                """
                MATCH (n:GraphNode)
                WHERE toLower(n.name) CONTAINS toLower($search)
                   OR toLower(n.summary) CONTAINS toLower($search)
                RETURN n.name as name,
                       n.summary as description,
                       n.origin_document as source,
                       n.text_snippet as chunk,
                       n.category as type
                LIMIT 12
                """,
                search=text_query
            )
            return [dict(record) for record in cursor]
