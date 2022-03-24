import os
import pickle

import numpy as np
import pytest

from pyrdf2vec.graphs import KG, Vertex
from pyrdf2vec.rdf2vec import RDF2VecTransformer
from pyrdf2vec.walkers import RandomWalker, WLWalker

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


class TestRDF2VecTransformer:
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

    def test_fail_load_transformer(self):
        pickle.dump([0, 1, 2], open("tmp", "wb"))
        with pytest.raises(ValueError):
            RDF2VecTransformer.load("tmp")
        os.remove("tmp")

    @pytest.mark.parametrize("kg", KGS)
    def test_get_walks(self, setup, kg):
        transformer = RDF2VecTransformer(verbose=2)
        assert len(transformer._walks) == 0
        with pytest.raises(ValueError):
            transformer.get_walks(kg, ["does", "not", "exist"])
        transformer.get_walks(
            kg, [f"{URL}#{entity}" for entity in ROOTS_WITHOUT_URL]
        )
        assert len(transformer._walks) > 0

    @pytest.mark.parametrize("kg", KGS)
    def test_fit_transform(self, kg):
        entities = [f"{URL}#{entity}" for entity in ROOTS_WITHOUT_URL]
        transformer = RDF2VecTransformer()
        np.testing.assert_array_equal(
            RDF2VecTransformer().fit_transform(kg, entities)[0],
            transformer.fit(kg, entities).transform(kg, entities)[0],
        )

    def test_load_save_transformer(self):
        RDF2VecTransformer(
            walkers=[
                RandomWalker(2, None, random_state=42),
                WLWalker(2, 2, random_state=42),
            ]
        ).save()
        transformer = RDF2VecTransformer.load()
        assert len(transformer.walkers) == 2
        assert isinstance(transformer.walkers[0], RandomWalker)
        assert isinstance(transformer.walkers[1], WLWalker)
        os.remove("transformer_data")

    @pytest.mark.parametrize("kg", KGS)
    def test_transform(self, setup, kg):
        entities = [f"{URL}#{entity}" for entity in ROOTS_WITHOUT_URL]
        assert (
            type(
                RDF2VecTransformer(verbose=2)
                .fit(kg, entities)
                .transform(kg, entities)
            )
            == tuple
        )
