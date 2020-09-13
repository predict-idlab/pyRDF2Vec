import random
from typing import List

import rdflib

from pyrdf2vec.graphs import KnowledgeGraph
from pyrdf2vec.samplers import UniformSampler
from pyrdf2vec.walkers import WeisfeilerLehmanWalker

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


class TestWeisfeilerLehmanWalker:
    def test_extract(self):
        canonical_walks = WeisfeilerLehmanWalker(
            2, 5, UniformSampler()
        ).extract(KG, generate_entities())
        assert type(canonical_walks) == set

    def test_weisfeiler_lehman(self):
        WeisfeilerLehmanWalker(2, 5, UniformSampler())._weisfeiler_lehman(KG)
        assert True
