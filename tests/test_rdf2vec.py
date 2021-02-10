import os
import pickle

import numpy as np
import pandas as pd
import pytest
import rdflib

from pyrdf2vec.graphs import KG
from pyrdf2vec.rdf2vec import RDF2VecTransformer
from pyrdf2vec.walkers import RandomWalker, WeisfeilerLehmanWalker

KNOWLEDGE_GRAPH = KG(
    "samples/mutag/mutag.owl",
    skip_predicates={"http://dl-learner.org/carcinogenesis#isMutagenic"},
)

TRAIN_DF = pd.read_csv("samples/mutag/train.tsv", sep="\t", header=0)

ENTITIES = [rdflib.URIRef(x) for x in TRAIN_DF["bond"]]
ENTITIES_SUBSET = ENTITIES[:5]


class TestRDF2VecTransformer:
    def test_fail_load_transformer(self):
        pickle.dump([0, 1, 2], open("tmp", "wb"))
        with pytest.raises(ValueError):
            RDF2VecTransformer.load("tmp")
        os.remove("tmp")

    def test_fit(self):
        transformer = RDF2VecTransformer()
        with pytest.raises(ValueError):
            transformer.fit(KNOWLEDGE_GRAPH, ["does", "not", "exist"])
        transformer.fit(KNOWLEDGE_GRAPH, ENTITIES_SUBSET)

    def test_fit_transform(self):
        np.testing.assert_array_equal(
            RDF2VecTransformer().fit_transform(
                KNOWLEDGE_GRAPH, ENTITIES_SUBSET
            ),
            RDF2VecTransformer()
            .fit(KNOWLEDGE_GRAPH, ENTITIES_SUBSET)
            .transform(ENTITIES_SUBSET),
        )

    def test_load_save_transformer(self):
        RDF2VecTransformer(
            walkers=[
                RandomWalker(2, None, seed=42),
                WeisfeilerLehmanWalker(2, 2, seed=42),
            ]
        ).save()
        transformer = RDF2VecTransformer.load()
        assert len(transformer.walkers) == 2
        assert isinstance(transformer.walkers[0], RandomWalker)
        assert isinstance(transformer.walkers[1], WeisfeilerLehmanWalker)
        os.remove("transformer_data")

    def test_transform(self):
        assert (
            type(
                RDF2VecTransformer()
                .fit(KNOWLEDGE_GRAPH, ENTITIES_SUBSET)
                .transform(ENTITIES_SUBSET)
            )
            == list
        )
