import random

import numpy as np
import pandas as pd
import pytest
import rdflib
from sklearn.exceptions import NotFittedError

from pyrdf2vec.graphs import KG
from pyrdf2vec.rdf2vec import RDF2VecTransformer

np.random.seed(42)
random.seed(42)

TEST_KG = KG(
    "samples/mutag/mutag.owl",
    label_predicates=["http://dl-learner.org/carcinogenesis#isMutagenic"],
)
TRAIN_DF = pd.read_csv("samples/mutag/train.tsv", sep="\t", header=0)

ENTITIES = [rdflib.URIRef(x) for x in TRAIN_DF["bond"]]
ENTITIES_SUBSET = ENTITIES[:5]


class TestRDF2VecTransformer:
    def test_fit(self):
        transformer = RDF2VecTransformer()
        # The provided entities to fit() should be in the KG
        with pytest.raises(ValueError):
            non_existing_entities = ["does", "not", "exist"]
            transformer.fit(TEST_KG, non_existing_entities)
        transformer.fit(TEST_KG, ENTITIES_SUBSET)
        assert True

    def test_fit_transform(self):
        transformer = RDF2VecTransformer()
        np.testing.assert_array_equal(
            transformer.fit_transform(TEST_KG, ENTITIES_SUBSET),
            transformer.fit(TEST_KG, ENTITIES_SUBSET).transform(
                ENTITIES_SUBSET
            ),
        )

    def test_transform(self):
        transformer = RDF2VecTransformer()
        with pytest.raises(NotFittedError):
            transformer.transform(ENTITIES_SUBSET)
        transformer.fit(TEST_KG, ENTITIES_SUBSET)
        features_vectors = transformer.transform(ENTITIES_SUBSET)
        assert type(features_vectors) == list
