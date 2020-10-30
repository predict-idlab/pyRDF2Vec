import inspect
import os
import pkgutil
import random
from operator import itemgetter
from typing import List, Tuple, TypeVar

import numpy as np
import pandas as pd
import pytest
import rdflib

import pyrdf2vec
from pyrdf2vec.graphs import KG
from pyrdf2vec.samplers import UniformSampler
from pyrdf2vec.walkers import Walker

np.random.seed(42)
random.seed(42)

LABEL_PREDICATE = "http://dl-learner.org/carcinogenesis#isMutagenic"
KNOWLEDGE_GRAPH = KG(
    "samples/mutag/mutag.owl", label_predicates=[LABEL_PREDICATE]
)

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


def _get_walkers() -> List[Tuple[str, T]]:
    """Gets the classes that are not a subclass of `sklearn.BaseEstimator` and
    that are not an abstract class.

    Returns:
        The classes.

    """
    classes = [  # type: ignore
        c  # type: ignore
        for c in set(_get_classes())  # type: ignore
        if issubclass(c[1], Walker)  # type: ignore
    ]
    classes = filter(lambda c: not is_abstract(c[1]), classes)  # type: ignore
    return sorted(set(classes), key=itemgetter(0))


def check_walker(Walker, name):
    walks_per_graph = 5
    depth = 2
    canonical_walks = Walker(depth, walks_per_graph, UniformSampler()).extract(
        KNOWLEDGE_GRAPH, ENTITIES_SUBSET
    )
    assert type(canonical_walks) == set
    if name == "WeisfeilerLehmanWalker":
        assert len(canonical_walks) <= len(
            ENTITIES_SUBSET * walks_per_graph * 5
        )
    elif name == "WalkletWalker":
        assert len(canonical_walks) <= len(
            ENTITIES_SUBSET * walks_per_graph * (depth + 1)
        )
    else:
        assert len(canonical_walks) <= len(ENTITIES_SUBSET * walks_per_graph)


def is_abstract(c) -> bool:
    """Tells whether a class is abstract or not.

    Args:
        c: The class has to determine if it is abstract or not.

    Returns:
        True if abstract class, False otherwise.

    """
    return (
        hasattr(c, "__abstractmethods__") and len(c.__abstractmethods__) != 0
    )


@pytest.mark.parametrize("name, Walker", _get_walkers())
def test_walkers(name: str, Walker: T):
    """Tests the walkers."""
    print(f"Testing {name}")
    check_walker(Walker, name)
