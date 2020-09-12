import random

import rdflib

from pyrdf2vec.graphs import KnowledgeGraph
from pyrdf2vec.samplers import UniformSampler
from pyrdf2vec.walkers import HalkWalker

LABEL_PREDICATE = "http://dl-learner.org/carcinogenesis#isMutagenic"
KG = KnowledgeGraph(
    "samples/mutag/mutag.owl", label_predicates=[LABEL_PREDICATE]
)


def generate_entities():
    return [
        rdflib.URIRef(
            f"{LABEL_PREDICATE.split('#')[0] + '#'}{random.randint(0, 335)}"
        )
        for _ in range(random.randint(0, 200))
    ]


class TestHalkWalker:
    def test_extract(self):
        canonical_walks = HalkWalker(2, 5, UniformSampler()).extract(
            KG, str(generate_entities())
        )
        assert type(canonical_walks) == set
