import random
from collections import defaultdict

import numpy as np
import pandas as pd
import pytest
import rdflib
from sklearn.exceptions import NotFittedError

from pyrdf2vec.graphs import RDFLoader
from pyrdf2vec.rdf2vec import RDF2VecTransformer
from pyrdf2vec.samplers import UniformSampler
from pyrdf2vec.walkers import RandomWalker

np.random.seed(42)
random.seed(42)

KG = RDFLoader(
    "samples/mutag/mutag.owl",
    label_predicates=["http://dl-learner.org/carcinogenesis#isMutagenic"],
)

TRAIN_DF = pd.read_csv("samples/mutag/train.tsv", sep="\t", header=0)

ENTITIES = [rdflib.URIRef(x) for x in TRAIN_DF["bond"]]
ENTITIES_SUBSET = ENTITIES[:5]

WALKS = defaultdict(list)


class TestRDF2VecTransformer:
    def test_fit(self):
        transformer = RDF2VecTransformer(
            500, [RandomWalker(2, 5, UniformSampler(inverse=False))]
        )

        # The provided entities to fit() should be in the KG
        with pytest.raises(ValueError):
            non_existing_entities = ["does", "not", "exist"]
            transformer.fit(KG, non_existing_entities)
        transformer.fit(KG, ENTITIES_SUBSET)
        for walks in transformer.walks_:
            WALKS[walks[0]].append(walks)
        assert len(WALKS["http://dl-learner.org/carcinogenesis#d133"]) == 5

    def test_fit_transform(self):
        transformer = RDF2VecTransformer()
        np.testing.assert_array_equal(
            transformer.fit_transform(KG, ENTITIES_SUBSET),
            transformer.fit(KG, ENTITIES_SUBSET).transform(ENTITIES_SUBSET),
        )

    def test_transform(self):
        transformer = RDF2VecTransformer()
        with pytest.raises(NotFittedError):
            transformer.transform(ENTITIES_SUBSET)
        transformer.fit(KG, ENTITIES_SUBSET)
        features_vectors = transformer.transform(ENTITIES_SUBSET)
        assert type(features_vectors) == list
