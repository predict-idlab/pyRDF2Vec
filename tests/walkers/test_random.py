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
        "kg, depth, root",
        list(
            itertools.product(
                (KG_LOOP, KG_CHAIN),
                (range(5)),
                (f"{URL}#Alice", f"{URL}#Bob", f"{URL}#Dean"),
            )
        ),
    )
    def test_bfs(self, setup, kg, depth, root):
        walks = RandomWalker(depth, None, random_state=42)._bfs(
            kg, Vertex(root)
        )
        for walk in walks:
            assert len(walk) == (depth * 2) + 1
            assert walk[0].name == root

    @pytest.mark.parametrize(
        "kg, depth, max_walks, root",
        list(
            itertools.product(
                (KG_LOOP, KG_CHAIN),
                (0, 2, 3, 4, 5, 10, 15, 20),
                (0, 1, 2, 3, 4, 5),
                (f"{URL}#Alice", f"{URL}#Bob", f"{URL}#Dean"),
            )
        ),
    )
    def test_dfs(self, setup, kg, depth, max_walks, root):
        walks = RandomWalker(depth, max_walks, random_state=42)._dfs(
            kg, Vertex(root)
        )
        if depth == 0:
            if max_walks == 0:
                assert len(walks) == 0
            else:
                assert len(walks) == 1
        else:
            assert len(walks) == max_walks

        depth1 = (depth * 2) - 1
        depth2 = (depth * 2) + 1
        for walk in walks:
            assert len(walk) == depth1 or depth2
            assert walk[0].name == root
