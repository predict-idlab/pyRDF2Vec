from typing import List
from urllib.parse import quote

import rdflib
import requests
from tqdm import tqdm

from pyrdf2vec.graph import KnowledgeGraph, Vertex


def create_kg(
    triples: rdflib.Graph, label_predicates: List[rdflib.URIRef]
) -> KnowledgeGraph:
    """Creates a knowledge graph according to triples and predicates label.

    Args:
        triples: The triples where each item in this list must be an
            iterable (e.g., tuple, list) of three elements.
        label_predicates: The URI's of the predicates that have to be
            excluded from the graph to avoid leakage.

    Returns:
        The knowledge graph.

    """
    kg = KnowledgeGraph()
    for (s, p, o) in tqdm(triples):
        if p not in label_predicates:
            s_v = Vertex(str(s))
            o_v = Vertex(str(o))
            p_v = Vertex(str(p), predicate=True, vprev=s_v, vnext=o_v)
            kg.add_vertex(s_v)
            kg.add_vertex(p_v)
            kg.add_vertex(o_v)
            kg.add_edge(s_v, p_v)
            kg.add_edge(p_v, o_v)
    return kg


def endpoint_to_kg(
    endpoint_url: str = "http://localhost:5820/db/query?query=",
    label_predicates: List[rdflib.URIRef] = [],
) -> KnowledgeGraph:
    """Generates a knowledge graph using a SPARQL endpoint.

    Args:
        endpoint_url: The SPARQL endpoint.
            Defaults to http://localhost:5820/db/query?query=
        label_predicates: The predicates label.
            Defaults to [].

    Returns:
        The knowledge graph.

    """
    session = requests.Session()
    session.mount(
        "http://",
        requests.adapters.HTTPAdapter(pool_connections=100, pool_maxsize=100),
    )

    query = quote("SELECT ?s ?p ?o WHERE { ?s ?p ?o }")
    try:
        r = session.get(
            endpoint_url + query,
            headers={"Accept": "application/sparql-results+json"},
        )
        qres = r.json()["results"]["bindings"]
    except Exception as e:
        print(e)
        print("Could not query the result!")
        qres = []

    triples = [
        (row["s"]["value"], row["p"]["value"], row["o"]["value"])
        for row in qres
    ]
    return create_kg(triples, label_predicates)


def rdflib_to_kg(
    file_name: str,
    file_type: str = None,
    label_predicates: List[rdflib.URIRef] = [],
) -> KnowledgeGraph:
    """Converts a rdflib.Graph type object to a knowledge graph.

    Args:
        file_name: The file name that contains the rdflib.Graph.
        file_type: The format of the knowledge graph.
            Defaults to None.
        label_predicates: The predicates label.
            Defaults to [].

    Returns:
        The knowledge graph.

    """
    kg = rdflib.Graph()
    if file_type is not None:
        kg.parse(file_name, format=file_type)
    else:
        kg.parse(file_name)
    return create_kg(kg, [rdflib.term.URIRef(x) for x in label_predicates])
