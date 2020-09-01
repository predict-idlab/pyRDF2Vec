import rdflib

from pyrdf2vec.converters import create_kg, endpoint_to_kg, rdflib_to_kg
from pyrdf2vec.graph import KnowledgeGraph

label_predicates = ["http://dl-learner.org/carcinogenesis#isMutagenic"]


def test_create_kg():
    kg = create_kg(rdflib.Graph(), label_predicates=label_predicates)
    assert type(kg) == KnowledgeGraph


def test_endpoint_to_kg():
    kg = endpoint_to_kg(
        "http://localhost:5820/db/query?query=",
        label_predicates=label_predicates,
    )
    assert type(kg) == KnowledgeGraph


def test_rdflib_to_kg():
    kg = rdflib_to_kg("samples/mutag.owl", label_predicates=label_predicates)
    assert type(kg) == KnowledgeGraph
