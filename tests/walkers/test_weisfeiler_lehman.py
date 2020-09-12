import random

import rdflib

from pyrdf2vec.graphs import KnowledgeGraph, Vertex
from pyrdf2vec.samplers import UniformSampler
from pyrdf2vec.walkers import WeisfeilerLehmanWalker

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


class TestWeisfeilerLehmanWalker:
    def test_extract(self):
        canonical_walks = WeisfeilerLehmanWalker(
            2, 5, UniformSampler()
        ).extract(KG, generate_entities())
        assert type(canonical_walks) == set
        pass

    def test_create_label(self):
        WeisfeilerLehmanWalker(2, 5, UniformSampler())._create_label(
            KG, Vertex("a"), 0
        )
        assert True
        pass

    def test_weisfeiler_lehman(self):
        WeisfeilerLehmanWalker(2, 5, UniformSampler())._weisfeiler_lehman(KG)
        assert True
