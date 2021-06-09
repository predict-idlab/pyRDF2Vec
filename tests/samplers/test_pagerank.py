import itertools

import pytest

from pyrdf2vec.graphs import KG, Vertex
from pyrdf2vec.samplers import PageRankSampler

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

ALPHA = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
IS_INVERSE = [False, True]
IS_REVERSE = [False, True]
IS_SPLIT = [False, True]
KGS = [KG_LOOP, KG_CHAIN]
ROOTS_WITHOUT_URL = ["Alice", "Bob", "Dean"]


class TestPageRankSampler:
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

    def test_invalid_weight(self):
        with pytest.raises(ValueError):
            PageRankSampler().get_weight(None)

    @pytest.mark.parametrize("kg", list((KG_LOOP, KG_CHAIN)))
    def test_fit(self, setup, kg):
        sampler = PageRankSampler()
        assert len(sampler._pageranks) == 0
        sampler.fit(kg)
        assert len(sampler._pageranks) > 0

    @pytest.mark.parametrize(
        "kg, root, is_reverse, alpha, is_inverse, is_split",
        list(
            itertools.product(
                KGS,
                ROOTS_WITHOUT_URL,
                IS_REVERSE,
                ALPHA,
                IS_INVERSE,
                IS_SPLIT,
            )
        ),
    )
    def test_weight(
        self, setup, kg, root, is_reverse, alpha, is_inverse, is_split
    ):
        sampler = PageRankSampler(
            alpha=alpha, inverse=is_inverse, split=is_split
        )
        sampler.fit(kg)
        for hop in kg.get_hops(Vertex(f"{URL}#{root}"), is_reverse=is_reverse):
            assert sampler.get_weight(hop) <= alpha
