import random
from collections import defaultdict
from typing import DefaultDict

import numpy as np
import pandas as pd
import pytest
import rdflib
from sklearn.exceptions import NotFittedError

from pyrdf2vec.graphs import KG
from pyrdf2vec.rdf2vec import RDF2VecTransformer

np.random.seed(42)
random.seed(42)

KNOWLEDGE_GRAPH = KG(
    "samples/mutag/mutag.owl",
    label_predicates=["http://dl-learner.org/carcinogenesis#isMutagenic"],
)

TRAIN_DF = pd.read_csv("samples/mutag/train.tsv", sep="\t", header=0)

ENTITIES = [rdflib.URIRef(x) for x in TRAIN_DF["bond"]]
ENTITIES_SUBSET = ENTITIES[:5]

WALKS: DefaultDict[rdflib.URIRef, rdflib.URIRef] = defaultdict(list)


class TestRDF2VecTransformer:
    def test_fit(self):
        transformer = RDF2VecTransformer()
        with pytest.raises(ValueError):
            transformer.fit(KNOWLEDGE_GRAPH, ["does", "not", "exist"])
        transformer.fit(KNOWLEDGE_GRAPH, ENTITIES_SUBSET)

    def test_fit_transform(self):
        transformer = RDF2VecTransformer()
        np.testing.assert_array_equal(
            transformer.fit_transform(KNOWLEDGE_GRAPH, ENTITIES_SUBSET),
            transformer.fit(KNOWLEDGE_GRAPH, ENTITIES_SUBSET).transform(
                ENTITIES_SUBSET
            ),
        )

    def test_transform(self):
        transformer = RDF2VecTransformer()
        with pytest.raises(NotFittedError):
            transformer.transform(ENTITIES_SUBSET)
        transformer.fit(KNOWLEDGE_GRAPH, ENTITIES_SUBSET)
        features = transformer.transform(ENTITIES_SUBSET)
        assert type(features) == list
