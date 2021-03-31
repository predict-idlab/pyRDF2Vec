import itertools

import pytest

from pyrdf2vec.graphs import KG, Vertex
from pyrdf2vec.samplers import UniformSampler

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

IS_REVERSE = [False, True]
KGS = [KG_LOOP, KG_CHAIN]
ROOTS_WITHOUT_URL = ["Alice", "Bob", "Dean"]


class TestUniformSampler:
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

    def test_fit(self):
        UniformSampler().fit(None)

    @pytest.mark.parametrize(
        "kg, root, is_reverse",
        list(itertools.product(KGS, ROOTS_WITHOUT_URL, IS_REVERSE)),
    )
    def test_weight(self, setup, kg, root, is_reverse):
        sampler = UniformSampler()
        for hop in kg.get_hops(Vertex(f"{URL}#{root}"), is_reverse=is_reverse):
            assert sampler.get_weight(hop) == 1
