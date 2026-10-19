"""
Local-Agentic-GraphRAG — Neo4j Enterprise Dual-Graph Engine
Provides seamless connection to local Neo4j (bolt://localhost:7687).
Supports Cypher-based multi-hop traversal, batch UNWIND syncing, and automatic schema indexing.
Gracefully handles offline states with zero system crashes.
"""

import os
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger("neo4j_engine")

try:
    from neo4j import GraphDatabase, Driver
    NEO4J_DRIVER_AVAILABLE = True
except ImportError:
    NEO4J_DRIVER_AVAILABLE = False
    Driver = Any


class Neo4jEngine:
    """
    Enterprise Dual-Graph Engine backed by Neo4j with Cypher query execution.
    Acts as the primary graph engine when Neo4j is running, while safely falling
    back to NetworkX / JSON graph when offline.
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password")
        self._driver: Optional[Driver] = None
        self._connected = False
        self._init_connection()

    def _init_connection(self):
        """Attempts to verify connectivity with Neo4j server."""
        if not NEO4J_DRIVER_AVAILABLE:
            self._connected = False
            return

        try:
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                connection_timeout=2.0,
                max_connection_lifetime=300,
            )
            # Lightweight ping
            self._driver.verify_connectivity()
            self._connected = True
            self._ensure_schema()
            logger.info("Connected successfully to Neo4j at %s", self.uri)
        except Exception as e:
            self._connected = False
            self._driver = None
            logger.debug("Neo4j offline (%s). Using local in-memory graph fallback.", str(e))

    def is_connected(self) -> bool:
        """Returns True if live Neo4j instance is connected and accessible."""
        if not self._connected or not self._driver:
            return False
        try:
            self._driver.verify_connectivity()
            return True
        except Exception:
            self._connected = False
            return False

    def _ensure_schema(self):
        """Initializes unique constraints and indexes for high-speed entity matching."""
        if not self._connected or not self._driver:
            return

        try:
            with self._driver.session() as session:
                session.run(
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE"
                )
                session.run(
                    "CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.name)"
                )
        except Exception as e:
            logger.warning("Could not create Neo4j constraints: %s", str(e))

    def sync_triples(self, triples: List[Dict[str, str]]):
        """
        Batch imports entity-relation triples into Neo4j using UNWIND Cypher.
        Triples format: [{"subject": "A", "relation": "rel", "object": "B"}]
        """
        if not self.is_connected() or not triples:
            return

        cypher = """
        UNWIND $batch AS item
        MERGE (s:Entity {name: item.subject})
          ON CREATE SET s.id = apoc.create.uuid(), s.label = item.subject
        MERGE (o:Entity {name: item.object})
          ON CREATE SET o.id = apoc.create.uuid(), o.label = item.object
        MERGE (s)-[r:RELATION {type: item.relation}]->(o)
        """
        # Fallback cypher if APOC is not installed
        clean_cypher = """
        UNWIND $batch AS item
        MERGE (s:Entity {name: item.subject})
          ON CREATE SET s.id = item.subject, s.label = item.subject
        MERGE (o:Entity {name: item.object})
          ON CREATE SET o.id = item.object, o.label = item.object
        MERGE (s)-[r:RELATION {type: item.relation}]->(o)
        """

        try:
            with self._driver.session() as session:
                session.run(clean_cypher, batch=triples)
        except Exception as e:
            logger.error("Error syncing triples to Neo4j: %s", str(e))

    def multi_hop_query(
        self,
        keywords: List[str],
        max_hops: int = 2,
        limit: int = 25,
    ) -> List[Dict[str, Any]]:
        """
        Executes a deep multi-hop Cypher traversal from matching entities.
        Returns a list of structured relationship records:
        [{"source": "A", "relation": "rel", "target": "B", "hops": 1}]
        """
        if not self.is_connected() or not keywords:
            return []

        cypher = f"""
        MATCH (s:Entity)
        WHERE ANY(kw IN $keywords WHERE toLower(s.name) CONTAINS toLower(kw))
        MATCH path = (s)-[r:RELATION*1..{max_hops}]-(t:Entity)
        WITH s, t, relationships(path) AS rels, length(path) AS hops
        RETURN DISTINCT
            s.name AS source,
            [rel in rels | rel.type] AS relations,
            t.name AS target,
            hops
        ORDER BY hops ASC
        LIMIT $limit
        """

        results = []
        try:
            with self._driver.session() as session:
                records = session.run(cypher, keywords=keywords, limit=limit)
                for rec in records:
                    rels = rec["relations"]
                    rel_str = " -> ".join(rels) if rels else "related_to"
                    results.append({
                        "source": rec["source"],
                        "relation": rel_str,
                        "target": rec["target"],
                        "hops": rec["hops"],
                    })
        except Exception as e:
            logger.error("Neo4j multi-hop query failed: %s", str(e))

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Returns node and relationship counts from Neo4j."""
        if not self.is_connected():
            return {"connected": False, "nodes": 0, "edges": 0}

        try:
            with self._driver.session() as session:
                nodes_res = session.run("MATCH (n:Entity) RETURN count(n) AS cnt")
                nodes_cnt = nodes_res.single()["cnt"]
                rels_res = session.run("MATCH ()-[r:RELATION]->() RETURN count(r) AS cnt")
                rels_cnt = rels_res.single()["cnt"]
                return {
                    "connected": True,
                    "nodes": nodes_cnt,
                    "edges": rels_cnt,
                    "uri": self.uri,
                }
        except Exception:
            return {"connected": False, "nodes": 0, "edges": 0}

    def close(self):
        """Closes driver connection pool."""
        if self._driver:
            try:
                self._driver.close()
            except Exception:
                pass


# Global singleton instance
neo4j_engine = Neo4jEngine()
