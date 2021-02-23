import itertools

import pytest

from pyrdf2vec.graphs import KG, Vertex
from pyrdf2vec.samplers import *

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


class TestSampler:
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
        "kg, root, sampler, is_reverse",
        list(
            itertools.product(
                (KG_LOOP, KG_CHAIN),
                (f"{URL}#Alice", f"{URL}#Bob", f"{URL}#Dean"),
                (
                    ObjFreqSampler(),
                    ObjFreqSampler(inverse=True),
                    ObjFreqSampler(split=True),
                    ObjFreqSampler(inverse=True, split=True),
                    ObjPredFreqSampler(),
                    ObjPredFreqSampler(inverse=True),
                    ObjPredFreqSampler(split=True),
                    ObjPredFreqSampler(inverse=True, split=True),
                    PredFreqSampler(),
                    PredFreqSampler(inverse=True),
                    PredFreqSampler(split=True),
                    PredFreqSampler(inverse=True, split=True),
                    UniformSampler(),
                    PageRankSampler(),
                    PageRankSampler(inverse=True),
                    PageRankSampler(split=True),
                    PageRankSampler(inverse=True, split=True),
                ),
                (False, True),
            ),
        ),
    )
    def test_get_weights(self, setup, kg, root, sampler, is_reverse):
        sampler.fit(kg)
        weights = sampler.get_weights(
            kg.get_hops(Vertex(root), is_reverse=is_reverse)
        )
        assert isinstance(weights, list)
        if len(weights) > 0:
            for weight in weights:
                assert weight <= 1
