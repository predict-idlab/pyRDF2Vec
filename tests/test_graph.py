import os

import pytest
import rdflib

from pyrdf2vec.graphs import KG, Vertex

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

LABEL_PREDICATES = {"http://dl-learner.org/carcinogenesis#isMutagenic"}
LOCAL_KG = KG("tmp.ttl", file_type="turtle")


class TestKG:
    def test_get_neighbors(self):
        # remote_kg = KG("https://dbpedia.org/sparql", is_remote=True)
        for graph in [LOCAL_KG]:
            neighbors = graph.get_hops(f"{URL}#Alice")

            predicates = [neighbor[0] for neighbor in neighbors]
            assert {str(predicate) for predicate in predicates} == {
                f"{URL}#knows"
            }

            objects = [neighbor[1] for neighbor in neighbors]
            assert Vertex(f"{URL}#Bob") in objects
            assert Vertex(f"{URL}#Dean") in objects

    def test_invalid_file(self):
        with pytest.raises(FileNotFoundError):
            KG(
                "foo",
                label_predicates=LABEL_PREDICATES,
            )

        with pytest.raises(FileNotFoundError):
            KG(
                "samples/mutag/",
                label_predicates=LABEL_PREDICATES,
            )

    def test_invalid_url(self):
        with pytest.raises(ValueError):
            KG(
                "foo",
                label_predicates=LABEL_PREDICATES,
                is_remote=True,
            )

    def test_valid_file(self):
        assert KG(
            "samples/mutag/mutag.owl",
            label_predicates=LABEL_PREDICATES,
        )

    def test_valid_url(self):
        KG(
            "https://dbpedia.org/sparql",
            label_predicates=LABEL_PREDICATES,
            is_remote=True,
        )


os.remove("tmp.ttl")
