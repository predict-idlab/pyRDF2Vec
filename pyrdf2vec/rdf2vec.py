from __future__ import annotations

import asyncio
import pickle
import time
from typing import List, Sequence, Tuple

import attr

from pyrdf2vec.embedders import Embedder, Word2Vec
from pyrdf2vec.graphs import KG, Vertex
from pyrdf2vec.typings import Embeddings, Entities, Literals
from pyrdf2vec.walkers import RandomWalker, Walker


@attr.s
class RDF2VecTransformer:
    """Transforms nodes in a Knowledge Graph into an embedding.

    Attributes:
        _embeddings: All the embeddings of the model.
            Defaults to [].
        _entities: All the entities of the model.
            Defaults to [].
        _is_extract_walks_literals: True if the session must be closed after
            the call to the `transform` function. False, otherwise.
            Defaults to False.
        _literals: All the literals of the model.
            Defaults to [].
        _walks: All the walks of the model.
            Defaults to [].
        embedder: The embedding technique.
            Defaults to Word2Vec.
        walkers: The walking strategies.
            Defaults to [RandomWalker(2, None)]
        verbose: The verbosity level.
            0: does not display anything;
            1: display of the progress of extraction and training of walks;
            2: debugging.
            Defaults to 0.

    """

    embedder = attr.ib(
        factory=lambda: Word2Vec(),
        type=Embedder,
        validator=attr.validators.instance_of(Embedder),  # type: ignore
    )

    walkers = attr.ib(
        factory=lambda: [RandomWalker(2)],  # type: ignore
        type=Sequence[Walker],
        validator=attr.validators.deep_iterable(
            member_validator=attr.validators.instance_of(
                Walker  # type: ignore
            ),
            iterable_validator=attr.validators.instance_of(list),
        ),
    )

    verbose = attr.ib(
        kw_only=True,
        default=0,
        type=int,
        validator=attr.validators.in_([0, 1, 2]),
    )

    _is_extract_walks_literals = attr.ib(
        init=False,
        default=False,
        type=bool,
        repr=False,
        validator=attr.validators.instance_of(bool),
    )

    _embeddings = attr.ib(init=False, type=Embeddings, factory=list)
    _entities = attr.ib(init=False, type=Entities, factory=list)
    _literals = attr.ib(init=False, type=Literals, factory=list)
    _walks = attr.ib(init=False, type=List[str], factory=list)

    def fit(
        self, walks: List[str], is_update: bool = False
    ) -> RDF2VecTransformer:
        """Fits the embeddings based on the provided entities.

        Args:
            walks: The walks to fit.
            is_update: True if the new corpus should be added to old model's
                corpus, False otherwise.
                Defaults to False.

        Returns:
            The RDF2VecTransformer.

        """
        if self.verbose == 2:
            print(self.embedder)

        tic = time.perf_counter()
        self.embedder.fit([list(map(str, walk)) for walk in walks], is_update)
        toc = time.perf_counter()

        if self.verbose >= 1:
            print(f"Fitted {len(walks)} walks ({toc - tic:0.4f}s)")
            if len(self._walks) != len(walks):
                print(
                    f"> {len(self._walks)} walks extracted "
                    + f"for {len(self._entities)} entities."
                )
        return self

    def fit_transform(
        self, kg: KG, entities: Entities, is_update: bool = False
    ) -> Tuple[Embeddings, Literals]:
        """Creates a model and generates embeddings and literals for the
        provided entities.

        Args:
            kg: The Knowledge Graph.
            entities: The entities including test entities to create the
                embeddings. Since RDF2Vec is unsupervised, there is no label
                leakage.
            is_update: True if the new corpus should be added to old model's
                corpus, False otherwise.
                Defaults to False.

        Returns:
            The embeddings and the literals of the provided entities.

        """
        self._is_extract_walks_literals = True
        self.fit(self.get_walks(kg, entities), is_update)
        return self.transform(kg, entities)

    def get_walks(self, kg: KG, entities: Entities) -> List[str]:
        """Gets the walks of an entity based on a Knowledge Graph and a
        list of walkers

        Args:
            kg: The Knowledge Graph.
            entities: The entities including test entities to create the
                embeddings. Since RDF2Vec is unsupervised, there is no label
                leakage.

        Returns:
            The walks for the given entities.

        Raises:
            ValueError: If the provided entities aren't in the Knowledge Graph.

        """
        if not kg._is_remote and not all(
            [Vertex(entity) in kg._vertices for entity in entities]
        ):
            raise ValueError(
                "The provided entities must be in the Knowledge Graph."
            )

        is_new_entities = False
        if not all(entity in self._entities for entity in entities):
            self._entities.extend(entities)
            is_new_entities = True

        if self.verbose == 2:
            print(kg)
            print(self.walkers[0])

        walks: List[str] = []
        tic = time.perf_counter()
        for walker in self.walkers:
            walks += walker.extract(kg, entities, self.verbose)
        toc = time.perf_counter()

        if self._walks is None:
            self._walks = walks
        elif is_new_entities:
            self._walks += walks

        if self.verbose >= 1:
            print(
                f"Extracted {len(walks)} walks "
                + f"for {len(entities)} entities ({toc - tic:0.4f}s)"
            )
        if (
            kg._is_remote
            and kg.mul_req
            and not self._is_extract_walks_literals
        ):
            asyncio.run(kg.connector.close())
        return walks

    def transform(
        self, kg: KG, entities: Entities
    ) -> Tuple[Embeddings, Literals]:
        """Transforms the provided entities into embeddings and literals.

        Args:
            kg: The Knowledge Graph.
            entities: The entities including test entities to create the
                embeddings. Since RDF2Vec is unsupervised, there is no label
                leakage.

        Returns:
            The embeddings and the literals of the provided entities.

        """
        assert self.embedder is not None
        embeddings = self.embedder.transform(entities)
        self._embeddings += embeddings

        tic = time.perf_counter()
        literals = kg.get_literals(entities, self.verbose)
        toc = time.perf_counter()

        if not all(entity in self._entities for entity in entities):
            self._literals += literals

        if kg._is_remote and kg.mul_req:
            self._is_extract_walks_literals = False
            asyncio.run(kg.connector.close())

        if self.verbose >= 1 and len(literals) > 0:
            print(
                f"Extracted {len(literals)} literals for {len(entities)} "
                + f"entities ({toc - tic:0.4f}s)"
            )
        return embeddings, literals

    def save(self, filename: str = "transformer_data") -> None:
        """Saves a RDF2VecTransformer object.

        Args:
            filename: The binary file to save the RDF2VecTransformer object.

        """
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename: str = "transformer_data") -> RDF2VecTransformer:
        """Loads a RDF2VecTransformer object.

        Args:
            filename: The binary file to load the RDF2VecTransformer object.

        Returns:
            The loaded RDF2VecTransformer.

        """

        with open(filename, "rb") as f:
            transformer = pickle.load(f)
            if not isinstance(transformer, RDF2VecTransformer):
                raise ValueError(
                    "Failed to load the RDF2VecTransformer object"
                )
            return transformer
