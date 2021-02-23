import itertools
from collections import defaultdict

import pytest

from pyrdf2vec.graphs import KG, Vertex

from pyrdf2vec.samplers import (  # isort: skip
    ObjFreqSampler,
    ObjPredFreqSampler,
    PredFreqSampler,
)

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


class TestFreqSampler:
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
        "freq_sampler",
        list(
            (
                ObjFreqSampler(),
                ObjFreqSampler(inverse=True),
                ObjFreqSampler(inverse=True, split=True),
                ObjPredFreqSampler(),
                ObjPredFreqSampler(inverse=True),
                ObjPredFreqSampler(inverse=True, split=True),
                PredFreqSampler(),
                PredFreqSampler(inverse=True),
                PredFreqSampler(inverse=True, split=True),
            )
        ),
    )
    def test_invalid_weight(self, freq_sampler):
        with pytest.raises(ValueError):
            freq_sampler.get_weight(None)

    @pytest.mark.parametrize(
        "kg, freq_sampler",
        list(
            itertools.product(
                (KG_LOOP, KG_CHAIN),
                (
                    ObjFreqSampler(),
                    ObjFreqSampler(inverse=True),
                    ObjFreqSampler(inverse=True, split=True),
                    ObjPredFreqSampler(),
                    ObjPredFreqSampler(inverse=True),
                    ObjPredFreqSampler(inverse=True, split=True),
                    PredFreqSampler(),
                    PredFreqSampler(inverse=True),
                    PredFreqSampler(inverse=True, split=True),
                ),
            )
        ),
    )
    def test_fit(self, setup, kg, freq_sampler):
        sampler = freq_sampler
        # To bypass the fact that the objects tested with pytest are the same.
        freq_sampler._counts = defaultdict(dict)
        assert len(sampler._counts) == 0
        sampler.fit(kg)
        if isinstance(sampler, ObjFreqSampler):
            assert len(sampler._counts) == 9
        elif isinstance(sampler, ObjPredFreqSampler):
            if kg == KG_LOOP:
                assert len(sampler._counts) == 3
            else:
                assert len(sampler._counts) == 8
        else:
            assert len(sampler._counts) == 2

    @pytest.mark.parametrize(
        "kg, root, is_reverse, freq_sampler",
        list(
            itertools.product(
                (KG_LOOP, KG_CHAIN),
                (f"{URL}#Alice", f"{URL}#Bob", f"{URL}#Dean"),
                (False, True),
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
                ),
            )
        ),
    )
    def test_weight(self, setup, kg, root, is_reverse, freq_sampler):
        sampler = freq_sampler
        sampler.fit(kg)
        for hop in kg.get_hops(Vertex(root), is_reverse=is_reverse):
            if isinstance(sampler, ObjFreqSampler):
                assert sampler.get_weight(hop) <= 2
