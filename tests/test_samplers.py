import inspect
import os
import pkgutil
from operator import itemgetter
from typing import Any, List, Tuple, TypeVar

import pandas as pd
import pytest
import rdflib

import pyrdf2vec
from pyrdf2vec.graphs import KG
from pyrdf2vec.samplers import Sampler
from pyrdf2vec.walkers import RandomWalker

KNOWLEDGE_GRAPH = KG(
    "samples/mutag/mutag.owl",
    skip_predicates={"http://dl-learner.org/carcinogenesis#isMutagenic"},
)

LEAKY_KG = KG("samples/mutag/mutag.owl", skip_predicates=set())
TRAIN_DF = pd.read_csv("samples/mutag/train.tsv", sep="\t", header=0)

ENTITIES = [rdflib.URIRef(x) for x in TRAIN_DF["bond"]]
ENTITIES_SUBSET = ENTITIES[:5]

T = TypeVar("T")


def _get_classes() -> List[Tuple[str, T]]:
    """Gets the classes from a package.

    Returns:
        The classes from a package.

    """
    classes = []
    base_path = [os.path.dirname(pyrdf2vec.__file__)]
    for _, name, _ in pkgutil.walk_packages(
        path=base_path, prefix="pyrdf2vec."
    ):
        module = __import__(name, fromlist="dummy")
        classes.extend(inspect.getmembers(module, inspect.isclass))
    return classes


def _get_samplers() -> List[Tuple[str, T]]:
    """Gets the classes that are not a subclass of `sklearn.BaseEstimator` and
    that are not an abstract class.

    Returns:
        The classes.

    """
    classes = [  # type: ignore
        cls  # type: ignore
        for cls in set(_get_classes())  # type: ignore
        if issubclass(cls[1], Sampler)  # type: ignore
    ]
    classes = filter(lambda c: not is_abstract(c[1]), classes)  # type: ignore
    return sorted(set(classes), key=itemgetter(0))


def check_sampler(Sampler):
    max_walks = 5
    canonical_walks = RandomWalker(
        2, max_walks, Sampler(), random_state=42
    ).extract(KNOWLEDGE_GRAPH, ENTITIES_SUBSET)
    assert type(canonical_walks) == set
    assert len(canonical_walks) <= len(ENTITIES_SUBSET * max_walks)


def is_abstract(cls: Any) -> bool:
    """Tells whether a class is abstract or not.

    Args:
        c: The class has to determine if it is abstract or not.

    Returns:
        True if abstract class, False otherwise.

    """
    return (
        hasattr(cls, "__abstractmethods__")
        and len(cls.__abstractmethods__) != 0
    )


@pytest.mark.parametrize("name, Sampler", _get_samplers())
def test_samplers(name: str, Sampler: T):
    """Tests the samplers.

    Args:
        name: The name of the sampler.
        Walker: The class of the sampler.

    """
    print(f"Testing the Sampler: {name}")
    check_sampler(Sampler)
