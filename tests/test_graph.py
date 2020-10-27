import multiprocessing
import os
import sys
import time

import rdflib

from pyrdf2vec.graphs import KG, Vertex
from rdflib_web.lod import serve

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


# Creating a small, artificial graph in rdflib.Graph
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

# Serialize the graph
g.serialize("tmp.ttl", format="turtle")

# Host a local endpoint
old_stdout = sys.stdout
old_stderr = sys.stderr
sys.stdout = open(os.devnull, "w")
sys.stderr = open(os.devnull, "w")
proc = multiprocessing.Process(target=serve, daemon=True, args=(g,))
proc.start()
time.sleep(3)
sys.stdout = old_stdout
sys.stderr = old_stderr

LOCAL_KNOWLEDGE_GRAPH = KG(location="tmp.ttl", file_type="turtle")

REMOTE_KNOWLEDGE_GRAPH = KG(
    location="http://localhost:5000/sparql", is_remote=True
)


class TestKG:
    # def test_visualise(self):
    #     KNOWLEDGE_GRAPH.visualise()
    #     assert True

    # def test_add_edge(self):
    #     KNOWLEDGE_GRAPH.add_edge(a, c)
    #     assert True

    def test_get_neighbors(self):
        for graph in [LOCAL_KNOWLEDGE_GRAPH, REMOTE_KNOWLEDGE_GRAPH]:
            neighbors = graph.get_hops("http://pyRDF2Vec#Alice")
            predicates = [x[0] for x in neighbors]

            assert {str(x) for x in predicates} == {"http://pyRDF2Vec#knows"}

            objects = [x[1] for x in neighbors]
            assert Vertex("http://pyRDF2Vec#Bob") in objects
            assert Vertex("http://pyRDF2Vec#Dean") in objects

    # def test_inv_get_neighbors(self):
    #     KNOWLEDGE_GRAPH.get_inv_neighbors(a)
    #     assert True

    # def test_remove_edge(self):
    #     KNOWLEDGE_GRAPH.remove_edge(a, c)
    #     assert True


# Closing the server and removing the temporary rdf file
proc.terminate()
proc.join()
os.remove("tmp.ttl")
