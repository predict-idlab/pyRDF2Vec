import itertools

import pytest

from pyrdf2vec.graphs import KG, Vertex
from pyrdf2vec.walkers import WLWalker

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

MAX_DEPTHS = range(5)
KGS = [KG_LOOP, KG_CHAIN]
MAX_WALKS = [None, 0, 1, 2, 3, 4, 5]
ROOTS_WITHOUT_URL = ["Alice", "Bob", "Dean"]
WITH_REVERSE = [False, True]
WL_ITERATIONS = range(5)


class TestWLWalker:
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
        "kg, root, max_depth, max_walks, with_reverse, wl_iterations",
        list(
            itertools.product(
                KGS,
                ROOTS_WITHOUT_URL,
                MAX_DEPTHS,
                MAX_WALKS,
                WITH_REVERSE,
                WL_ITERATIONS,
            )
        ),
    )
    def test_extract(
        self,
        setup,
        kg,
        root,
        max_depth,
        max_walks,
        with_reverse,
        wl_iterations,
    ):
        root = f"{URL}#{root}"
        walker = WLWalker(
            max_depth,
            max_walks,
            with_reverse=with_reverse,
            random_state=42,
            wl_iterations=wl_iterations,
        )
        walker._weisfeiler_lehman(kg)
        walks = walker._extract(kg, Vertex(root))[root]
        if max_walks is not None:
            if not with_reverse:
                assert len(walks) <= (max_walks * wl_iterations) + max(
                    max_depth, max_walks
                )
        for walk in walks:
            if not with_reverse:
                assert walk[0] == root
