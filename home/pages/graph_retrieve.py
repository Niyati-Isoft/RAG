# graph_retrieve.py
from typing import List
from neo4j import GraphDatabase
import spacy, os
nlp = spacy.load("en_core_web_sm")

def get_driver(uri, user, password):
    return GraphDatabase.driver(uri, auth=(user, password))

def query_entities(q: str) -> List[str]:
    if not q: return []
    doc = nlp(q)
    ents = [e.text for e in doc.ents]
    return list({x for x in ents if 2 < len(x) < 120}) or [q]

def get_graph_context(driver, entities: List[str], max_edges=60) -> str:
    if not entities: return ""
    with driver.session() as sess:
        data = sess.run("""
        MATCH (e:Entity) WHERE e.name IN $ents
        MATCH p=(e)-[r*1..2]-(n:Entity)
        WITH p, r LIMIT $max
        UNWIND r AS rel
        WITH DISTINCT startNode(rel) AS s, rel AS rel, endNode(rel) AS t
        OPTIONAL MATCH (c:Chunk)-[:MENTIONS]->(s)
        RETURN s.name AS h, type(rel) AS r, t.name AS t,
               rel.source AS source, c.url AS url, c.page AS page, c.slide AS slide
        """, ents=entities, max=max_edges).data()

    lines, seen = [], set()
    for d in data:
        cite = []
        if d.get("source"): cite.append(str(d["source"]))
        if d.get("page"):   cite.append(f"p.{d['page']}")
        if d.get("slide"):  cite.append(f"slide {d['slide']}")
        if d.get("url"):    cite.append(d["url"])
        line = f"- {d['h']} —{d['r']}→ {d['t']}  [{'; '.join(cite)}]".strip()
        if line not in seen:
            seen.add(line); lines.append(line)
    return "\n".join(lines[:200])
