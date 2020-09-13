import random
from typing import List

import pytest
import rdflib

from pyrdf2vec.graphs import KnowledgeGraph
from pyrdf2vec.samplers import UniformSampler
from pyrdf2vec.walkers import Walker

LABEL_PREDICATE = "http://dl-learner.org/carcinogenesis#isMutagenic"
KG = KnowledgeGraph(
    "samples/mutag/mutag.owl", label_predicates=[LABEL_PREDICATE]
)


def generate_entities() -> List[rdflib.URIRef]:
    return [
        rdflib.URIRef(
            f"{LABEL_PREDICATE.split('#')[0] + '#'}{random.randint(0, 335)}"
        )
        for _ in range(random.randint(0, 200))
    ]


class TestWalker:
    def test_extract_not_implemented(self):
        walker = Walker(2, 5, UniformSampler())
        with pytest.raises(NotImplementedError):
            walker.extract(KG, generate_entities())
