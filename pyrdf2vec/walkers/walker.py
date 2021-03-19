import multiprocessing
import warnings
from abc import ABC, abstractmethod
from typing import List, Optional

import attr
from tqdm import tqdm

from pyrdf2vec.graphs import KG, Vertex
from pyrdf2vec.samplers import Sampler, UniformSampler
from pyrdf2vec.typings import Entities, EntityWalks

from pyrdf2vec.utils.validation import (  # isort: skip
    _check_max_depth,
    _check_jobs,
    _check_max_walks,
)


class WalkerNotSupported(Exception):
    """Base exception class for the lack of support of a walking strategy for
    the extraction of walks via a SPARQL endpoint server.

    """

    pass


@attr.s
class Walker(ABC):
    """Base class of the walking strategies."""

    kg: Optional[KG] = None
    """Global KG used later on for the worker process."""

    max_depth: int = attr.ib(
        validator=[attr.validators.instance_of(int), _check_max_depth]
    )
    """The maximum depth of one walk."""

    max_walks: Optional[int] = attr.ib(  # type: ignore
        default=None,
        validator=[
            attr.validators.optional(attr.validators.instance_of(int)),
            _check_max_walks,
        ],
    )
    """The maximum number of walks per entity."""

    sampler: Sampler = attr.ib(
        factory=lambda: UniformSampler(),
        validator=attr.validators.instance_of(Sampler),  # type: ignore
    )
    """The sampling strategy."""

    n_jobs: Optional[int] = attr.ib(  # type: ignore
        default=None,
        validator=[
            attr.validators.optional(attr.validators.instance_of(int)),
            _check_jobs,
        ],
    )
    """The number of CPU cores used when parallelizing.
    None means 1. -1 means using all processors.
    """

    with_reverse: Optional[bool] = attr.ib(
        kw_only=True,
        default=False,
        validator=attr.validators.instance_of(bool),
    )
    """True to extracts children's and parents' walks from the root,
    creating (max_walks * max_walks) more walks of 2 * depth, False otherwise.
    """

    random_state: Optional[int] = attr.ib(
        kw_only=True,
        default=None,
        validator=attr.validators.optional(attr.validators.instance_of(int)),
    )
    """The random state to use to keep random determinism with the walking
    strategy.
    """

    _is_support_remote: bool = attr.ib(init=False, repr=False, default=True)
    """True if the walking strategy can be used with a remote Knowledge Graph,
    False Otherwise.
    """

    def __attrs_post_init__(self):
        if self.n_jobs == -1:
            self.n_jobs = multiprocessing.cpu_count()
        self.sampler.random_state = self.random_state

    def extract(
        self, kg: KG, entities: Entities, verbose: int = 0
    ) -> List[str]:
        """Fits the provided sampling strategy and then calls the
        private _extract method that is implemented for each of the
        walking strategies.

        Args:
            kg: The Knowledge Graph.
            entities: The entities to be extracted from the Knowledge Graph.
            verbose: The verbosity level.
                0: does not display anything;
                1: display of the progress of extraction and training of walks;
                2: debugging.
                Defaults to 0.

        Returns:
            The 2D matrix with its number of rows equal to the number of
            provided entities; number of column equal to the embedding size.

        Raises:
            WalkerNotSupported: If there is an attempt to use an invalid
                walking strategy to a remote Knowledge Graph.

        """
        if kg._is_remote and not self._is_support_remote:
            raise WalkerNotSupported(
                "Invalid walking strategy. Please, choose a walking strategy "
                + "that can fetch walks via a SPARQL endpoint server."
            )
        self.sampler.fit(kg)

        process = self.n_jobs if self.n_jobs is not None else 1
        if (kg._is_remote and kg.mul_req) and process >= 2:
            warnings.warn(
                "Using 'mul_req=True' and/or 'n_jobs>=2' speed up the "
                + "extraction of entity's walks, but may violate the policy "
                + "of some SPARQL endpoint servers.",
                category=RuntimeWarning,
                stacklevel=2,
            )

        if kg._is_remote and kg.mul_req:
            kg._fill_hops(entities)

        with multiprocessing.Pool(process, self._init_worker, [kg]) as pool:
            res = list(
                tqdm(
                    pool.imap_unordered(self._proc, entities),
                    total=len(entities),
                    disable=True if verbose == 0 else False,
                )
            )

        entity_walks = {
            entity: walks for elm in res for entity, walks in elm.items()
        }

        canonical_walks = set()
        for entity in entities:
            canonical_walks.update(entity_walks[entity])
        return list(canonical_walks)

    @abstractmethod
    def _extract(self, kg: KG, entity: Vertex) -> EntityWalks:
        """Extracts walks rooted at the provided entities which are then each
        transformed into a numerical representation.

        Args:
            kg: The Knowledge Graph.
            entity: The entity to be extracted from the Knowledge Graph.

        Returns:
            The 2D matrix with its number of rows equal to the number of
            provided entities; number of column equal to the embedding size.

        Raises:
            NotImplementedError: If this method is called, without having
                provided an implementation.

        """
        raise NotImplementedError("This must be implemented!")

    def _init_worker(self, init_kg: KG) -> None:
        """Initializes each worker process.

        Args:
            init_kg: The Knowledge Graph to provide to each worker process.

        """
        global kg
        kg = init_kg  # type: ignore

    def _proc(self, entity: str) -> EntityWalks:
        """Executed by each process.

        Args:
            entity: The entity to be extracted from the Knowledge Graph.

        Returns:
            The extraction of walk by the process.

        """
        global kg
        return self._extract(kg, Vertex(entity))  # type: ignore
