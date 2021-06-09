import itertools

import pytest

from pyrdf2vec.graphs import KG, Vertex
from pyrdf2vec.samplers import WideSampler

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

KGS = [KG_LOOP, KG_CHAIN]
ROOTS_WITHOUT_URL = ["Alice", "Bob", "Dean"]


class TestWideSampler:
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
            WideSampler().get_weight(None)

    @pytest.mark.parametrize("kg", list((KG_LOOP, KG_CHAIN)))
    def test_fit(self, setup, kg):
        sampler = WideSampler()
        assert len(sampler._pred_degs) == 0
        assert len(sampler._obj_degs) == 0
        assert len(sampler._neighbor_counts) == 0

        sampler.fit(kg)
        assert len(sampler._pred_degs) > 0
        assert len(sampler._obj_degs) > 0
        assert len(sampler._neighbor_counts) > 0

    @pytest.mark.parametrize(
        "kg, root",
        list(itertools.product(KGS, ROOTS_WITHOUT_URL)),
    )
    def test_weight(self, setup, kg, root):
        sampler = WideSampler()
        sampler.fit(kg)
        for hop in kg.get_hops(Vertex(f"{URL}#{root}")):
            weight = sampler.get_weight(hop)
            assert weight > 0
            assert isinstance(weight, float)
