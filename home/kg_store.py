# kg_store.py
import os
from neo4j import GraphDatabase

def get_driver(uri=None, user=None, password=None):
    uri = uri or os.getenv("NEO4J_URI")
    user = user or os.getenv("NEO4J_USER", "neo4j")
    pwd  = password or os.getenv("NEO4J_PASSWORD")
    if not uri or not user or not pwd:
        raise RuntimeError("Neo4j credentials missing (NEO4J_URI/USER/PASSWORD).")
    return GraphDatabase.driver(uri, auth=(user, pwd))

def upsert_triple(tx, h, htype, rel, t, ttype, source, chunk_id, conf=0.7):
    q = f"""
    MERGE (h:Entity {{name:$h}})
      ON CREATE SET h.type=$htype
    MERGE (t:Entity {{name:$t}})
      ON CREATE SET t.type=$ttype
    MERGE (h)-[r:{rel} {{source:$source}}]->(t)
      ON CREATE SET r.evidence_chunk_id=$chunk, r.confidence=$conf
    """
    tx.run(q, h=h, htype=htype, t=t, ttype=ttype, source=source, chunk=chunk_id, conf=conf)

def upsert_mention(tx, chunk_id, text, source, url, page, slide, e):
    q = """
    MERGE (c:Chunk {chunk_id:$cid})
      ON CREATE SET c.text=$text, c.source=$source, c.url=$url, c.page=$page, c.slide=$slide
    MERGE (e:Entity {name:$e})
    MERGE (c)-[:MENTIONS]->(e)
    """
    tx.run(q, cid=chunk_id, text=(text or "")[:2000], source=source, url=url, page=page, slide=slide, e=e)

def write_batch(driver, triples, mentions, meta):
    with driver.session() as sess:
        def run(tx):
            for (h, rel, t) in triples:
                upsert_triple(tx, h, meta['types'].get(h,"Concept"), rel, t, meta['types'].get(t,"Concept"),
                              meta.get('source'), meta.get('chunk_id'), meta.get('conf',0.7))
            for e in mentions:
                upsert_mention(tx, meta.get('chunk_id'), meta.get('text'), meta.get('source'),
                               meta.get('url'), meta.get('page'), meta.get('slide'), e)
        sess.execute_write(run)
