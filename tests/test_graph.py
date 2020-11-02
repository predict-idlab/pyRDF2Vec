import multiprocessing
import os
import sys
import time

import pytest
import rdflib

from pyrdf2vec.graphs import KG, Vertex
from tests.rdflib_web.lod import serve

a = Vertex("a")
b = Vertex("b")
c = Vertex("c", predicate=True, vprev=a, vnext=b)


class TestVertex:
    def test_eq(self):
        assert a == a

    def test_eq_with_none(self):
        assert a is not None

    def test_id_incremental(self):
        assert b.id == 1

    def test_id_init(self):
        assert a.id == 0

    def test_neq(self):
        assert a != b


SPARQL_ENDPOINT = "http://localhost:5000/sparql"
GRAPH = [
    ["Alice", "knows", "Bob"],
    ["Alice", "knows", "Dean"],
    ["Bob", "knows", "Casper"],
]
URL = "http://pyRDF2Vec"

g = rdflib.Graph()
for t in GRAPH:
    triple: rdflib.URIRef = tuple()
    for entity in t:
        triple = triple + (rdflib.URIRef(f"{URL}#{entity}"),)
    g.add(triple)
g.serialize("tmp.ttl", format="turtle")


@pytest.fixture(autouse=True, scope="session")
def start_server():
    """Hosts a local endpoint."""
    old_stderr, old_stdout = sys.stderr, sys.stdout
    sys.stderr, sys.stdout = open(os.devnull, "w"), open(os.devnull, "w")
    proc = multiprocessing.Process(target=serve, daemon=True, args=(g,))
    proc.start()
    time.sleep(3)
    sys.stdout = old_stdout
    sys.stderr = old_stderr

    yield

    proc.terminate()
    proc.join()


LOCAL_KG = KG(location="tmp.ttl", file_type="turtle")
REMOTE_KG = KG(location=SPARQL_ENDPOINT, is_remote=True)


class TestKG:
    def test_get_neighbors(self):
        for graph in [LOCAL_KG, REMOTE_KG]:
            neighbors = graph.get_hops(f"{URL}#Alice")

            predicates = [neighbor[0] for neighbor in neighbors]
            assert {str(predicate) for predicate in predicates} == {
                f"{URL}#knows"
            }

            objects = [neighbor[1] for neighbor in neighbors]
            assert Vertex(f"{URL}#Bob") in objects
            assert Vertex(f"{URL}#Dean") in objects


# Closing the server and removing the temporary RDF file
os.remove("tmp.ttl")
