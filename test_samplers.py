import os
import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rdflib
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC

from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.graphs import KnowledgeGraph
from pyrdf2vec.samplers import (ObjFreqSampler, ObjPredFreqSampler,
                                PageRankSampler, PredFreqSampler,
                                UniformSampler)
from pyrdf2vec.walkers import RandomWalker

warnings.filterwarnings("ignore")

samplers = [
    ("Uniform", UniformSampler()),
    ("Object Frequency", ObjFreqSampler()),
    ("Inverse Object Frequency", ObjFreqSampler(inverse=True)),
    (
        "Inverse Object Frequency Split",
        ObjFreqSampler(inverse=True, split=True),
    ),
    ("Predicate Frequency", PredFreqSampler()),
    ("Inverse Predicate Frequency", PredFreqSampler(inverse=True)),
    ("Predicate + Object Frequency", ObjPredFreqSampler()),
    ("Inverse Predicate + Object Frequency", ObjPredFreqSampler(inverse=True)),
    ("PageRank", PageRankSampler()),
    ("Inverse PageRank", PageRankSampler(inverse=True)),
    ("PageRank Split", PageRankSampler(split=True)),
    ("Inverse PageRank Split", PageRankSampler(inverse=True, split=True)),
]

for name, sampler in samplers:
    print(f"Testing {name}...")
    random_walker = RandomWalker(2, 5, sampler)
    transformer = RDF2VecTransformer(walkers=[random_walker], sg=1)
    walk_embeddings = transformer.fit_transform(kg, all_entities)
