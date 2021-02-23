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


class TestHalkWalker:
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
        "kg, root, is_reverse, alpha, inverse, split",
        list(
            itertools.product(
                (KG_LOOP, KG_CHAIN),
                (f"{URL}#Alice", f"{URL}#Bob", f"{URL}#Dean"),
                (False, True),
                (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                (False, True),
                (False, True),
            )
        ),
    )
    def test_weight(self, setup, kg, root, is_reverse, alpha, inverse, split):
        sampler = PageRankSampler(alpha=alpha, inverse=inverse, split=split)
        sampler.fit(kg)
        print(alpha)
        for hop in kg.get_hops(Vertex(root), is_reverse=is_reverse):
            assert sampler.get_weight(hop) <= alpha
