import pickle
import time
from typing import List, Sequence

import attr

from pyrdf2vec.embedders import Embedder, Word2Vec
from pyrdf2vec.graphs import KG, Vertex
from pyrdf2vec.walkers import RandomWalker, Walker


@attr.s
class RDF2VecTransformer:
    """Transforms nodes in a Knowledge Graph into an embedding.

    Args:
        embedder: The embedding technique.
            Defaults to pyrdf2vec.embedders.Word2Vec.
        walkers: The walking strategy.
            Defaults to pyrdf2vec.walkers.RandomWalker(2, None).
        verbose: If True, display a progress bar for the extraction of the
            walks and display the number of these extracted walks for the
            number of entities with the extraction time.
            Defaults to 0.

    """

    embedder: Embedder = attr.ib(
        factory=lambda: Word2Vec(),
        validator=attr.validators.instance_of(Embedder),  # type: ignore
    )
    walkers: Sequence[Walker] = attr.ib(
        factory=lambda: [RandomWalker(2)],  # type: ignore
        validator=attr.validators.deep_iterable(
            member_validator=attr.validators.instance_of(
                Walker  # type: ignore
            ),
            iterable_validator=attr.validators.instance_of(list),
        ),
    )
    verbose: int = attr.ib(
        kw_only=True, default=0, validator=attr.validators.in_([0, 1, 2])
    )

    _entities: List[str] = attr.ib(init=False, factory=list)
    _walks: List[str] = attr.ib(init=False, factory=list)

    def fit(
        self,
        kg: KG,
        entities: List[str],
        is_update: bool = False,
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
        if not kg._is_remote and not all(
            [Vertex(entity) in kg._vertices for entity in entities]
        ):
            raise ValueError(
                "The provided entities must be in the Knowledge Graph."
            )

        if self.verbose == 2:
            print(kg)
            print(self.walkers[0])
            print(self.embedder)

        self._entities.extend(entities)

        walks = []
        tic = time.perf_counter()
        for walker in self.walkers:
            walks += list(walker.extract(kg, entities, self.verbose))
        toc = time.perf_counter()

        if self._walks is None:
            self._walks = walks
        else:
            self._walks += walks

        if self.verbose >= 1:
            print(
                f"Extracted {len(walks)} walks "
                + f"for {len(entities)} entities ({toc - tic:0.4f}s)"
            )
            if len(self._walks) != len(walks):
                print(
                    f"> {len(self._walks)} walks extracted "
                    + f"for {len(self._entities)} entities."
                )

        corpus = [list(map(str, walk)) for walk in self._walks]
        self.embedder.fit(corpus, is_update)
        return self

    def transform(self, entities: List[str]) -> List[str]:
        """Constructs a feature vector for the provided entities.

        Args:
            entities: The entities to create the embeddings.
                The test entities should be passed to the fit method as well.

                Due to RDF2Vec being unsupervised, there is no label leakage.

        Returns:
            The embeddings of the provided entities.

        """
        assert self.embedder is not None
        return self.embedder.transform(entities)

    def fit_transform(
        self,
        kg: KG,
        entities: List[str],
        is_update: bool = False,
    ) -> List[str]:
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

        Returns:
            The embeddings of the provided entities.

        """
        self.fit(kg, entities, is_update)
        return self.transform(self._entities)

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
