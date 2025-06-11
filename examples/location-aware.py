from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.graphs import KG
from pyrdf2vec.samplers.geo import GeoSampler
from pyrdf2vec.walkers import RandomWalker
import numpy as np

if __name__ == '__main__':
    # Ensure the determinism of this script by initializing a pseudo-random number.
    RANDOM_STATE = 22
    kg = KG(
        "samples/coordinates/countries.ttl",
    )
    entities = [x.name for x in kg._entities]
    embeddings, literals = RDF2VecTransformer(
        # Ensure random determinism for Word2Vec.
        # Must be used with PYTHONHASHSEED.
        Word2Vec(workers=1, epochs=10, vector_size=250),

        # Extract all walks with a maximum depth of 2 for each entity using two
        # processes and use a random state to ensure that the same walks are
        # generated for the entities without hashing as MUTAG is a short KG.
        walkers=[
            RandomWalker(
                10,
                100,
                n_jobs=4,
                random_state=RANDOM_STATE,
                md5_bytes=None,
                sampler=GeoSampler(),
            )
        ],
        verbose=1,
    ).fit_transform(
        kg,
        entities,
    )
