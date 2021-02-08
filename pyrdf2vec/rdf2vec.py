import pickle
import time
from typing import List, Optional, Sequence

import rdflib

from pyrdf2vec.embedders import Embedder, Word2Vec
from pyrdf2vec.graphs import KG, Vertex
from pyrdf2vec.walkers import RandomWalker, Walker


class RDF2VecTransformer:
    """Transforms nodes in a Knowledge Graph into an embedding.

    Attributes:
        embedder: The embedding technique.
            Defaults to pyrdf2vec.embedders.Word2Vec.
        walkers: The walking strategy.
            Defaults to pyrdf2vec.walkers.RandomWalker(2, None).

    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        walkers: Optional[Sequence[Walker]] = None,
    ):

        self.walks_: List[rdflib.URIRef] = []

        if embedder is not None:
            self.embedder = embedder
        else:
            self.embedder = Word2Vec()

        if walkers is not None:
            self.walkers = walkers
        else:
            self.walkers = [RandomWalker(2)]

        self.entities_ = []

    def fit(
        self,
        kg: KG,
        entities: List[rdflib.URIRef],
        is_update: bool = False,
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
            is_update: If true, the new corpus will be added to old model's
                corpus.
            verbose: If true, display a progress bar for the extraction of the
                walks and display the number of these extracted walks for the
                number of entities with the extraction time.
                Defaults to False.

        Returns:
            The RDF2VecTransformer.

        """
        if not kg.is_remote and not all(
            [Vertex(str(entity)) in kg._vertices for entity in entities]
        ):
            raise ValueError(
                "The provided entities must be in the Knowledge Graph."
            )

        if verbose:
            print(self.walkers[0].info())  # type: ignore

        self.entities_.extend(entities)

        tic = time.perf_counter()
        for walker in self.walkers:  # type: ignore
            self.walks_ += list(walker.extract(kg, entities, verbose))
        toc = time.perf_counter()
        corpus = [list(map(str, walk)) for walk in self.walks_]

        if verbose:
            print(
                f"Extracted {len(self.walks_)} walks "
                + f"for {len(entities)} entities! ({toc - tic:0.4f}s)"
            )
        self.embedder.fit(corpus, is_update=is_update)
        return self

    def transform(self, entities: List[rdflib.URIRef]) -> List[rdflib.URIRef]:
        """Constructs a feature vector for the provided entities.

        Args:
            entities: The entities to create the embeddings.
                The test entities should be passed to the fit method as well.

                Due to RDF2Vec being unsupervised, there is no label leakage.

        Returns:
            The embeddings of the provided entities.

        """
        return self.embedder.transform(entities)

    def fit_transform(
        self,
        kg: KG,
        entities: List[rdflib.URIRef],
        is_update: bool = False,
        verbose: bool = False,
    ) -> List[rdflib.URIRef]:
        """Creates a Word2Vec model and generates embeddings for the provided
        entities.

        Args:
            kg: The Knowledge Graph.
                The graph from which we will extract neighborhoods for the
                provided instances.
            entities: The entities to create the embeddings.
                The test entities should be passed to the fit method as well.

                Due to RDF2Vec being unsupervised, there is no label leakage.
            is_update: If true, the new corpus will be added to old model's
                corpus.
            verbose: If true, display a progress bar for the extraction of the
                walks and display the number of these extracted walks for the
                number of entities with the extraction time.
                Defaults to False.

        Returns:
            The embeddings of the provided entities.

        """
        self.fit(kg, entities, is_update, verbose)
        return self.transform(self.entities_)

    def save(self, filename: str = "transformer_data") -> None:
        """Saves a RDF2VecTransformer object.

        Args:
            filename: The binary file to save the RDF2VecTransformer
            object.

        """
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename: str = "transformer_data") -> "RDF2VecTransformer":
        """Loads a RDF2VecTransformer object.

        Args:
            filename: The binary file to load the RDF2VecTransformer
            object.

        """
        with open(filename, "rb") as f:
            transformer = pickle.load(f)
            if not isinstance(transformer, RDF2VecTransformer):
                raise ValueError(
                    "Failed to load the RDF2VecTransformer object"
                )
            return transformer
