import multiprocessing
import os
import sys
import time

import rdflib

from pyrdf2vec.graphs import KG, Vertex
from tests.rdflib_web.lod import serve

# The tests for our Vertex object
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


# Alice -(knows)-> Bob -(knows)-> Casper
#       -(knows)-> Dean
g = rdflib.Graph()
g.add(
    (
        rdflib.URIRef("http://pyRDF2Vec#Alice"),
        rdflib.URIRef("http://pyRDF2Vec#knows"),
        rdflib.URIRef("http://pyRDF2Vec#Bob"),
    )
)
g.add(
    (
        rdflib.URIRef("http://pyRDF2Vec#Alice"),
        rdflib.URIRef("http://pyRDF2Vec#knows"),
        rdflib.URIRef("http://pyRDF2Vec#Dean"),
    )
)
g.add(
    (
        rdflib.URIRef("http://pyRDF2Vec#Bob"),
        rdflib.URIRef("http://pyRDF2Vec#knows"),
        rdflib.URIRef("http://pyRDF2Vec#Casper"),
    )
)
g.serialize("tmp.ttl", format="turtle")

# Host a local endpoint
old_stderr, old_stdout = sys.stderr, sys.stdout
sys.stderr, sys.stdout = open(os.devnull, "w"), open(os.devnull, "w")
proc = multiprocessing.Process(target=serve, daemon=True, args=(g,))
proc.start()
time.sleep(3)
sys.stdout = old_stdout
sys.stderr = old_stderr

# Load a local knowledge graph from a RDF file
LOCAL_KG = KG(location="tmp.ttl", file_type="turtle")
# Load a remote knowledge graph using a SPARQL endpoint
REMOTE_KG = KG(location="http://localhost:5000/sparql", is_remote=True)


class TestKG:
    def test_get_neighbors(self):
        for graph in [LOCAL_KG, REMOTE_KG]:
            neighbors = graph.get_hops("http://pyRDF2Vec#Alice")

            predicates = [x[0] for x in neighbors]
            assert {str(predicate) for predicate in predicates} == {
                "http://pyRDF2Vec#knows"
            }

            objects = [x[1] for x in neighbors]
            assert Vertex("http://pyRDF2Vec#Bob") in objects
            assert Vertex("http://pyRDF2Vec#Dean") in objects


# Closing the server and removing the temporary RDF file
proc.terminate()
proc.join()
os.remove("tmp.ttl")
