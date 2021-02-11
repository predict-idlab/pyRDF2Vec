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

SKIP_PREDICATES = {"http://dl-learner.org/carcinogenesis#isMutagenic"}
LOCAL_KG = KG("tmp.ttl", file_type="turtle")


class TestKG:
    def test_get_neighbors(self):
        alice_predicates = [
            neighbor
            for neighbor in LOCAL_KG.get_neighbors(Vertex(f"{URL}#Alice"))
        ]
        assert Vertex(f"{URL}#Alice") == alice_predicates[0].vprev
        assert Vertex(f"{URL}#Dean") == alice_predicates[0].vnext
        assert Vertex(f"{URL}#Bob") == alice_predicates[1].vnext
        assert Vertex(f"{URL}#Alice") == alice_predicates[1].vprev
        assert (
            len(
                [
                    neighbor
                    for neighbor in LOCAL_KG.get_neighbors(
                        Vertex(f"{URL}#Alice"), reverse=True
                    )
                ]
            )
            == 0
        )

        bob_predicates = [
            neighbor
            for neighbor in LOCAL_KG.get_neighbors(Vertex(f"{URL}#Bob"))
        ]
        assert Vertex(f"{URL}#Bob") == bob_predicates[0].vprev
        assert Vertex(f"{URL}#Casper") == bob_predicates[0].vnext

        bob_predicates = [
            neighbor
            for neighbor in LOCAL_KG.get_neighbors(
                Vertex(f"{URL}#Bob"), reverse=True
            )
        ]
        assert Vertex(f"{URL}#Bob") == bob_predicates[0].vnext
        assert Vertex(f"{URL}#Alice") == bob_predicates[0].vprev

        dean_predicates = [
            neighbor
            for neighbor in LOCAL_KG.get_neighbors(
                Vertex(f"{URL}#Dean"), reverse=True
            )
        ]
        assert Vertex(f"{URL}#Dean") == dean_predicates[0].vnext
        assert Vertex(f"{URL}#Alice") == dean_predicates[0].vprev
        assert (
            len(
                [
                    neighbor
                    for neighbor in LOCAL_KG.get_neighbors(
                        Vertex(f"{URL}#Dean")
                    )
                ]
            )
            == 0
        )

    def test_get_hops(self):
        for graph in [LOCAL_KG]:
            neighbors = graph.get_hops(Vertex(f"{URL}#Alice"))
            predicates = [neighbor[0] for neighbor in neighbors]
            objects = [neighbor[1] for neighbor in neighbors]

            assert {predicate.name for predicate in predicates} == {
                f"{URL}#knows"
            }
            assert Vertex(f"{URL}#Bob") in objects
            assert Vertex(f"{URL}#Dean") in objects

    def test_invalid_file(self):
        with pytest.raises(FileNotFoundError):
            KG(
                "foo",
                skip_predicates=SKIP_PREDICATES,
            )

        with pytest.raises(FileNotFoundError):
            KG(
                "samples/mutag/",
                skip_predicates=SKIP_PREDICATES,
            )

    def test_invalid_url(self):
        with pytest.raises(ValueError):
            KG(
                "foo",
                skip_predicates=SKIP_PREDICATES,
                is_remote=True,
            )

    def test_valid_file(self):
        assert KG(
            "samples/mutag/mutag.owl",
            skip_predicates=SKIP_PREDICATES,
        )

    def test_valid_url(self):
        KG(
            "https://dbpedia.org/sparql",
            skip_predicates=SKIP_PREDICATES,
            is_remote=True,
        )


os.remove("tmp.ttl")
