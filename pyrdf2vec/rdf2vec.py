from typing import List, Sequence

import rdflib

from pyrdf2vec.embedders import Embedder, Word2Vec
from pyrdf2vec.graphs import RDFLoader, Vertex
from pyrdf2vec.samplers import UniformSampler
from pyrdf2vec.walkers import RandomWalker, Walker


class RDF2VecTransformer:
    """Transforms nodes in a Knowledge Graph into an embedding.

    Attributes:
        embedder: The embedding technique.
            Defaults to pyrdf2vec.embedders.Word2Vec.
        walkers: The walking strategy.
            Defaults to pyrdf2vec.walkers.RandomWalker(2, None,
            UniformSampler(inverse=False)).

    """

    def __init__(
        self,
        embedder: Embedder = Word2Vec(),
        walkers: Sequence[Walker] = [
            RandomWalker(2, None, UniformSampler(inverse=False))
        ],
    ):
        self.embedder = embedder
        self.walks_ = []
        self.walkers = walkers

    def fit(
        self,
        kg: RDFLoader,
        entities: List[rdflib.URIRef],
        verbose: bool = False,
    ) -> "RDF2VecTransformer":
        """Fits the embedding network based on provided entities.

        Args:
            kg: The Knowledge Graph.
                The graph from which the neighborhoods are extracted for the
                provided entities.
            entities: The entities to create the embedding.
                The test entities should be passed to the fit method as well.

                Due to RDF2Vec being unsupervised, there is no label leakage.

        """
        if isinstance(kg, RDFLoader) and not all(
            [Vertex(str(entity)) in kg._vertices for entity in entities]
        ):
            raise ValueError(
                "The provided entities must be in the Knowledge Graph."
            )

        for walker in self.walkers:
            self.walks_ += list(walker.extract(kg, entities))
        corpus = [list(map(str, x)) for x in self.walks_]

        if verbose:
            print(
                f"Extracted {len(self.walks_)} walks "
                + f"for {len(entities)} entities!"
            )

        if isinstance(self.embedder, Word2Vec):
            self.model_ = self.embedder.fit(corpus)
        return self

    def transform(self, entities: List[rdflib.URIRef]) -> List[str]:
        """Constructs a feature vector for the provided entities.

        Args:
            entities: The entities to create the embedding.
                The test entities should be passed to the fit method as well.

                Due to RDF2Vec being unsupervised, there is no label leakage.

        Returns:
            The embeddings of the provided entities.

        """
        return self.embedder.transform(entities)

    def fit_transform(self, kg, entities: List[rdflib.URIRef]) -> List[str]:
        """Creates a Word2Vec model and generate embeddings for the provided
        entities.

        Args:
            kg: The Knowledge Graph.
                The graph from which we will extract neighborhoods for the
                provided instances.
            entities: The entities to create the embedding.
                The test entities should be passed to the fit method as well.

                Due to RDF2Vec being unsupervised, there is no label leakage.

        Returns:
            The embeddings of the provided entities.

        """
        self.fit(kg, entities)
        return self.transform(entities)
