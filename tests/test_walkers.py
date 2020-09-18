import pkgutil
import inspect
import pytest
import random
from operator import itemgetter

import numpy as np
import pandas as pd
import rdflib

import pyrdf2vec
from pyrdf2vec.walkers import Walker
from pyrdf2vec.samplers import UniformSampler
from pyrdf2vec.graphs import KG


np.random.seed(42)
random.seed(42)

LABEL_PREDICATE = "http://dl-learner.org/carcinogenesis#isMutagenic"
kg = KG("samples/mutag/mutag.owl", label_predicates=[LABEL_PREDICATE])
LEAKY_KG = KG("samples/mutag/mutag.owl", label_predicates=[])
train_df = pd.read_csv("samples/mutag/train.tsv", sep="\t", header=0)
entities = [rdflib.URIRef(x) for x in train_df["bond"]]
entities_subset = entities[:5]


def is_abstract(c):
    if not(hasattr(c, '__abstractmethods__')):
        return False
    if not len(c.__abstractmethods__):
        return False
    return True


def _get_all_classes():
    # Walk through all the packages from our base_path and
    # add all the classes to a list
    all_classes = []
    base_path = pyrdf2vec.__path__
    for _, name, _ in pkgutil.walk_packages(path=base_path,
                                            prefix='pyrdf2vec.'):
        module = __import__(name, fromlist="dummy")
        all_classes.extend(inspect.getmembers(module, inspect.isclass))
    return all_classes


def _get_walkers():
    all_classes = _get_all_classes()
    # Filter out those that are not a subclass of `sklearn.BaseEstimator`
    all_classes = [c for c in set(all_classes) if issubclass(c[1], Walker)]
    # get rid of abstract base classes
    all_classes = filter(lambda c: not is_abstract(c[1]), all_classes)
    return sorted(set(all_classes), key=itemgetter(0))


def check_walker(Walker):
    canonical_walks = Walker(2, 5, UniformSampler()).extract(
        kg, entities_subset
    )
    assert type(canonical_walks) == set


@pytest.mark.parametrize('name, Walker', _get_walkers())
def test_all_walkers(name, Walker):
    """Test all the estimators in tslearn."""
    print(f'Testing {name}')
    check_walker(Walker)