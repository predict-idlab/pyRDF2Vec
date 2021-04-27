import numpy as np
import pytest

from pyrdf2vec.graphs import KG, Vertex

a = Vertex("a")
b = Vertex("b")
c = Vertex("c", predicate=True, vprev=a, vnext=b)


class TestVertex:
    def test_eq(self):
        assert a == a
        assert a != 5
        assert b == b
        assert c == c

    def test_eq_with_none(self):
        assert a is not None
        assert b is not None
        assert c is not None

    def test_lt(self):
        assert a < b
        assert b < c
        assert a < c

    def test_neq(self):
        assert a != b
        assert b != c
        assert a != c


GRAPH = [
    ["Alice", "knows", "Bob"],
    ["Alice", "knows", "Dean"],
    ["Bob", "knows", "Casper"],
]
URL = "http://pyRDF2Vec"

LOCAL_KG = KG(cache=None, literals=[[f"{URL}#knows"]])


class TestKG:
    @pytest.fixture(scope="session")
    def setup(self):
        for row in GRAPH:
            subj = Vertex(f"{URL}#{row[0]}")
            obj = Vertex((f"{URL}#{row[2]}"))
            pred = Vertex(
                (f"{URL}#{row[1]}"), predicate=True, vprev=subj, vnext=obj
            )
            LOCAL_KG.add_walk(subj, pred, obj)

    def test_add_walk(self):
        kg = KG(skip_predicates={f"{URL}#predicate"})
        for row in GRAPH:
            subj = Vertex(f"{URL}#{row[0]}")
            obj = Vertex((f"{URL}#{row[2]}"))
            pred = Vertex(
                (f"{URL}#{row[1]}"), predicate=True, vprev=subj, vnext=obj
            )
            assert kg.add_walk(subj, pred, obj) is True
        subj = Vertex(f"{URL}#{GRAPH[0][0]}")
        obj = Vertex(f"{URL}#{GRAPH[0][2]}")
        pred = Vertex(f"{URL}#predicate")
        assert kg.add_walk(subj, pred, obj) is False

    def test_cast_literals(self, setup):
        literals = LOCAL_KG._cast_literals([[]])
        assert len(literals) == 1
        assert np.NaN in literals

        literals = LOCAL_KG._cast_literals([[], []])
        assert len(literals) == 2
        assert literals == [np.nan, np.nan]

        literals = LOCAL_KG._cast_literals([[f"{URL}#Bob"]])
        assert len(literals) == 1
        assert f"{URL}#Bob" in literals

        literals = LOCAL_KG._cast_literals([[f"{URL}#Bob", f"{URL}#Dean"]])
        assert len(literals) == 1
        print(literals)
        assert (f"{URL}#Bob", f"{URL}#Dean") in literals

        literals = LOCAL_KG._cast_literals([["0.42"]])
        assert len(literals) == 1
        assert 0.42 in literals

        literals = LOCAL_KG._cast_literals([["0.42", "0.37"]])
        assert len(literals) == 1
        assert (0.42, 0.37) in literals

    def test_fetch_hops(self):
        assert LOCAL_KG.fetch_hops(Vertex(f"{URL}#Alice")) == []

    def test_get_pliterals(self, setup):
        alice_literals = LOCAL_KG.get_pliterals(
            f"{URL}#Alice", LOCAL_KG.literals[0]
        )
        assert isinstance(alice_literals, list)
        assert len(alice_literals) == 2
        assert f"{URL}#Bob" in alice_literals
        assert f"{URL}#Dean" in alice_literals

        bob_literals = LOCAL_KG.get_pliterals(
            f"{URL}#Bob", LOCAL_KG.literals[0]
        )
        assert isinstance(bob_literals, list)
        assert len(bob_literals) == 1
        assert f"{URL}#Casper" in bob_literals

        dean_literals = LOCAL_KG.get_pliterals(
            f"{URL}#Dean", LOCAL_KG.literals[0]
        )
        assert isinstance(dean_literals, list)
        assert len(dean_literals) == 0
        assert dean_literals == []

    def test_get_hops(self, setup):
        neighbors = LOCAL_KG.get_hops(Vertex(f"{URL}#Alice"))
        predicates = [neighbor[0] for neighbor in neighbors]
        objects = [neighbor[1] for neighbor in neighbors]

        assert len(neighbors) == 2
        assert len(predicates) == 2
        assert len(objects) == 2

        assert {predicate.name for predicate in predicates} == {f"{URL}#knows"}
        assert Vertex(f"{URL}#Bob") in objects
        assert Vertex(f"{URL}#Dean") in objects

        neighbors = LOCAL_KG.get_hops(Vertex(f"{URL}#Bob"), is_reverse=True)
        predicates = [neighbor[0] for neighbor in neighbors]
        objects = [neighbor[1] for neighbor in neighbors]

        assert len(neighbors) == 1
        assert len(predicates) == 1
        assert len(objects) == 1
        assert Vertex(f"{URL}#Alice") in objects

    def test_get_neighbors(self, setup):
        alice_predicates = [
            neighbor
            for neighbor in LOCAL_KG.get_neighbors(Vertex(f"{URL}#Alice"))
        ]
        assert len(alice_predicates) == 2
        assert Vertex(f"{URL}#Alice") == alice_predicates[0].vprev
        assert Vertex(f"{URL}#Bob") and Vertex(f"{URL}#Dean") in {
            alice_predicates[0].vnext,
            alice_predicates[1].vnext,
        }
        assert Vertex(f"{URL}#Alice") == alice_predicates[1].vprev
        assert (
            len(
                [
                    neighbor
                    for neighbor in LOCAL_KG.get_neighbors(
                        Vertex(f"{URL}#Alice"), is_reverse=True
                    )
                ]
            )
            == 0
        )

        bob_predicates = [
            neighbor
            for neighbor in LOCAL_KG.get_neighbors(Vertex(f"{URL}#Bob"))
        ]
        assert len(bob_predicates) == 1
        assert Vertex(f"{URL}#Bob") == bob_predicates[0].vprev
        assert Vertex(f"{URL}#Casper") == bob_predicates[0].vnext

        bob_predicates = [
            neighbor
            for neighbor in LOCAL_KG.get_neighbors(
                Vertex(f"{URL}#Bob"), is_reverse=True
            )
        ]
        assert len(bob_predicates) == 1
        assert Vertex(f"{URL}#Bob") == bob_predicates[0].vnext
        assert Vertex(f"{URL}#Alice") == bob_predicates[0].vprev

        dean_predicates = [
            neighbor
            for neighbor in LOCAL_KG.get_neighbors(
                Vertex(f"{URL}#Dean"), is_reverse=True
            )
        ]
        assert len(dean_predicates) == 1
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

    def test_init_local_kg(self):
        kg = KG("samples/mutag/mutag.owl")
        assert len(kg._transition_matrix) > 0
        assert len(kg._inv_transition_matrix) > 0

    def test_invalid_file(self):
        with pytest.raises(FileNotFoundError):
            KG("foo")

        with pytest.raises(FileNotFoundError):
            KG("samples/mutag/")

    def test_invalid_url(self):
        with pytest.raises(ValueError):
            KG("http://foo")

    def test_is_exist(self, setup):
        assert LOCAL_KG.is_exist([f"{URL}#Alice", "foo"]) is False
        assert (
            LOCAL_KG.is_exist(
                [
                    f"{URL}#Alice",
                    f"{URL}#Bob",
                    f"{URL}#Casper",
                    f"{URL}#Dean",
                ]
            )
            is True
        )

    def test_remove_edge(self, setup):
        vtx_alice = Vertex(f"{URL}#Alice")

        neighbors = LOCAL_KG.get_hops(vtx_alice)
        assert len(LOCAL_KG.get_hops(vtx_alice)) == 2

        predicates = [
            vertex
            for hops in neighbors
            for vertex in hops
            if vertex.predicate is True
        ]

        with pytest.raises(ValueError):
            LOCAL_KG._is_remote = True
            LOCAL_KG.remove_edge(vtx_alice, predicates[0])
        LOCAL_KG._is_remote = False

        assert LOCAL_KG.remove_edge(vtx_alice, predicates[0]) is True
        assert len(LOCAL_KG.get_hops(vtx_alice)) == 1

        assert LOCAL_KG.remove_edge(vtx_alice, predicates[1]) is True
        assert len(LOCAL_KG.get_hops(Vertex(f"{URL}#Alice"))) == 0

        assert (
            LOCAL_KG.remove_edge(vtx_alice, Vertex(f"{URL}#Unknown")) is False
        )

    def test_res2hops(self, setup):
        subj = Vertex("subj")
        obj = Vertex("obj")
        hops = LOCAL_KG._res2hops(
            subj,
            [
                {
                    "o": {
                        "type": "literal",
                        "xml:lang": "en",
                        "value": "obj",
                    },
                    "p": {
                        "type": "literal",
                        "xml:lang": "en",
                        "value": "pred",
                    },
                },
            ],
        )
        assert isinstance(hops, list)
        assert len(hops) == 1
        pred = Vertex("pred", predicate=True, vprev=subj, vnext=obj)
        assert hops == [(pred, obj)]

    def test_valid_url(self):
        KG("https://dbpedia.org/sparql")
