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


MAX_DEPTHS = range(15)
KGS = [KG_LOOP, KG_CHAIN]
MAX_WALKS = [0, 1, 2, 3, 4, 5]
ROOTS_WITHOUT_URL = ["Alice", "Bob", "Dean"]
WITH_REVERSE = [False, True]


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
        "kg, root, max_depth, is_reverse",
        list(
            itertools.product(KGS, ROOTS_WITHOUT_URL, MAX_DEPTHS, WITH_REVERSE)
        ),
    )
    def test_bfs(self, setup, kg, root, max_depth, is_reverse):
        root = f"{URL}#{root}"
        walks = RandomWalker(max_depth, None, random_state=42)._bfs(
            kg, Vertex(root), is_reverse
        )
        for walk in walks:
            assert len(walk) <= (max_depth * 2) + 1
            if is_reverse:
                assert walk[-1].name == root
            else:
                assert walk[0].name == root

    @pytest.mark.parametrize(
        "kg, root, max_depth, max_walks, is_reverse",
        list(
            itertools.product(
                KGS, ROOTS_WITHOUT_URL, MAX_DEPTHS, MAX_WALKS, WITH_REVERSE
            ),
        ),
    )
    def test_dfs(self, setup, kg, root, max_depth, max_walks, is_reverse):
        root = f"{URL}#{root}"
        for walk in RandomWalker(max_depth, max_walks, random_state=42)._dfs(
            kg, Vertex(root), is_reverse
        ):
            assert len(walk) <= (max_depth * 2) + 1
            if is_reverse:
                assert walk[-1].name == root
            else:
                assert walk[0].name == root

    @pytest.mark.parametrize(
        "kg, root, max_depth, max_walks, with_reverse",
        list(
            itertools.product(
                KGS, ROOTS_WITHOUT_URL, MAX_DEPTHS, MAX_WALKS, WITH_REVERSE
            )
        ),
    )
    def test_extract(
        self, setup, kg, root, max_depth, max_walks, with_reverse
    ):
        root = f"{URL}#{root}"
        walker = RandomWalker(
            max_depth, max_walks, with_reverse=with_reverse, random_state=42
        )
        walks = walker.extract(kg, [root])[0]
        if max_walks is not None:
            if with_reverse:
                assert len(walks) <= max_walks * max_walks
            else:
                assert len(walks) <= max_walks
        for walk in walks:
            for obj in walk[2::2]:
                if obj not in walker._entities:
                    assert obj.startswith("b'")
            if not with_reverse:
                assert walk[0] == root
                assert len(walk) <= (max_depth * 2) + 1
            else:
                assert len(walk) <= ((max_depth * 2) + 1) * 2

    def test_inverse_extract(self, setup):
        walker = RandomWalker(1, None, with_reverse=True, random_state=42)
        walks = walker.extract(KG_LOOP, [f"{URL}#Bob", f"{URL}#Alice"])
        assert any(
            walk[0] == f"{URL}#Alice" and walk[2] == f"{URL}#Bob"
            for walk in walks[0] + walks[1]
        )
