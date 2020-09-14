import random
from typing import List

import rdflib

from pyrdf2vec.graphs import KG
from pyrdf2vec.samplers import UniformSampler
from pyrdf2vec.walkers import RandomWalker

LABEL_PREDICATE = "http://dl-learner.org/carcinogenesis#isMutagenic"
KNOWLEDGE_GRAPH = KG(
    "samples/mutag/mutag.owl", label_predicates=[LABEL_PREDICATE]
)


def generate_entities() -> List[rdflib.URIRef]:
    return [
        rdflib.URIRef(
            f"{LABEL_PREDICATE.split('#')[0] + '#'}{random.randint(0, 335)}"
        )
        for _ in range(random.randint(0, 200))
    ]


class TestRandomWalker:
    def test_extract_random_walks(self):
        walks = RandomWalker(2, 5, UniformSampler()).extract_random_walks(
            KNOWLEDGE_GRAPH, str(generate_entities())
        )
        assert type(walks) == list

    def test_extract(self):
        canonical_walks = RandomWalker(2, 5, UniformSampler()).extract(
            KNOWLEDGE_GRAPH, str(generate_entities())
        )
        assert type(canonical_walks) == set
