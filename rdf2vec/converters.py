import urllib

import rdflib
import requests
from tqdm import tqdm

from rdf2vec.graph import KnowledgeGraph, Vertex


def create_kg(triples, label_predicates):
    """Creates a knowledge graph according to triples and predicates label.

    Args:
        triples (list): The triples where each item in this list must be an
            iterable (e.g., tuple, list) of three elements.
        label_predicates (list): The URI's of the predicates that have to be
            excluded from the graph to avoid leakage.

    Returns:
        graph.KnowledgeGraph: The knowledge graph.

    """
    kg = KnowledgeGraph()
    for (s, p, o) in tqdm(triples):
        if p not in label_predicates:
            s_v = Vertex(str(s))
            o_v = Vertex(str(o))
            p_v = Vertex(str(p), predicate=True, _from=s_v, _to=o_v)
            kg.add_vertex(s_v)
            kg.add_vertex(p_v)
            kg.add_vertex(o_v)
            kg.add_edge(s_v, p_v)
            kg.add_edge(p_v, o_v)
    return kg


def endpoint_to_kg(
    endpoint_url="http://localhost:5820/db/query?query=", label_predicates=[]
):
    """Generates a knowledge graph using a SPARQL endpoint.

    endpoint_url (string): The SPARQL endpoint.
        Defaults to http://localhost:5820/db/query?query=
    label_predicates (list): The predicates label.
        Defaults to [].

    Returns:
        graph.KnowledgeGraph: The knowledge graph.

    """
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=100, pool_maxsize=100
    )
    session.mount("http://", adapter)

    query = urllib.parse.quote("SELECT ?s ?p ?o WHERE { ?s ?p ?o }")
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


def rdflib_to_kg(file_name, filetype=None, label_predicates=[]):
    """Converts a rdflib.Graph to a knowledge graph.

    Args:
        file_name (str): The file name that contains the rdflib.Graph.
        filetype (string): The format of the knowledge graph.
            Defaults to None.
        label_predicates (list): The predicates label.
            Defaults to [].

    Returns:
        graph.KnowledgeGraph: The knowledge graph.

    """
    kg = rdflib.Graph()
    if filetype is not None:
        kg.parse(file_name, format=filetype)
    else:
        kg.parse(file_name)
    return create_kg(kg, [rdflib.term.URIRef(x) for x in label_predicates])
