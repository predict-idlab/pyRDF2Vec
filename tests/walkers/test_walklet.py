import itertools

import pytest

from pyrdf2vec.graphs import KG, Vertex
from pyrdf2vec.walkers import WalkletWalker

LOOP = [
    ["Alice", "knows", "Bob"],
    ["Alice", "knows", "Dean"],
    ["Bob", "knows", "Dean"],
    ["Dean", "loves", "Alice"],
]
LONG_CHAIN = [
    ["Alice", "knows", "Bob"],
    ["Alice", "knows", "Dean"],
    ["Bob", "knows", "Mathilde"],
    ["Mathilde", "knows", "Alfy"],
    ["Alfy", "knows", "Stephane"],
    ["Stephane", "knows", "Alfred"],
    ["Alfred", "knows", "Emma"],
    ["Emma", "knows", "Julio"],
]
URL = "http://pyRDF2Vec"

KG_LOOP = KG()
KG_CHAIN = KG()


class TestWalkletWalker:
    @pytest.fixture(scope="session")
    def setup(self):
        for i, graph in enumerate([LOOP, LONG_CHAIN]):
            for row in graph:
                subj = Vertex(f"{URL}#{row[0]}")
                obj = Vertex((f"{URL}#{row[2]}"))
                pred = Vertex(
                    (f"{URL}#{row[1]}"), predicate=True, vprev=subj, vnext=obj
                )
                if i == 0:
                    KG_LOOP.add_walk(subj, pred, obj)
                else:
                    KG_CHAIN.add_walk(subj, pred, obj)

    @pytest.mark.parametrize(
        "kg, root, depth, max_walks, with_reverse",
        list(
            itertools.product(
                (KG_LOOP, KG_CHAIN),
                (f"{URL}#Alice", f"{URL}#Bob", f"{URL}#Dean"),
                range(15),
                (None, 0, 1, 2, 3, 4, 5),
                (False, True),
            )
        ),
    )
    def test_extract(self, setup, kg, root, depth, max_walks, with_reverse):
        walks = WalkletWalker(
            depth, max_walks, with_reverse=with_reverse
        )._extract(kg, Vertex(root))[root]
        for walk in walks:
            assert len(walk) == 1 or len(walk) == 2
            if not with_reverse:
                assert walk[0] == root
