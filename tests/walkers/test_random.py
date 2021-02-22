import itertools

import pytest

from pyrdf2vec.graphs import KG, Vertex
from pyrdf2vec.walkers import RandomWalker

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


class TestRandomWalker:
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
        "kg, root, depth, is_reverse",
        list(
            itertools.product(
                (KG_LOOP, KG_CHAIN),
                (f"{URL}#Alice", f"{URL}#Bob", f"{URL}#Dean"),
                (range(6)),
                (False, True),
            )
        ),
    )
    def test_bfs(self, setup, kg, depth, root, is_reverse):
        walks = RandomWalker(depth, None, random_state=42)._bfs(
            kg, Vertex(root), is_reverse
        )
        d = (depth * 2) + 1
        for walk in walks:
            assert len(walk) <= d
            if is_reverse:
                assert walk[-1].name == root
            else:
                assert walk[0].name == root

    @pytest.mark.parametrize(
        "kg, root, depth, max_walks, is_reverse",
        list(
            itertools.product(
                (KG_LOOP, KG_CHAIN),
                (f"{URL}#Alice", f"{URL}#Bob", f"{URL}#Dean"),
                (0, 2, 3, 4, 5, 10, 15, 20),
                (0, 1, 2, 3, 4, 5),
                (False, True),
            )
        ),
    )
    def test_dfs(self, setup, kg, root, depth, max_walks, is_reverse):
        walks = RandomWalker(depth, max_walks, random_state=42)._dfs(
            kg, Vertex(root), is_reverse
        )
        d = (depth * 2) + 1
        for walk in walks:
            assert len(walk) <= d

            if is_reverse:
                assert walk[-1].name == root
            else:
                assert walk[0].name == root
