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
    _check_depth,
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
    """Base class of the walking strategies.

    Args:
        depth: The depth per entity.
        max_walks: The maximum number of walks per entity.
        sampler: The sampling strategy.
            Defaults to UniformSampler().
        n_jobs: The number of CPU cores used when parallelizing. None means 1.
            -1 means using all processors.
            Defaults to 1.
        with_reverse: extracts children's and parents' walks from the root,
            creating (max_walks * max_walks) more walks of 2 * depth.
            Defaults to False.
        random_state: The random state to use to ensure ensure random
            determinism to generate the same walks for entities.
            Defaults to None.

    """

    # Global KG used later on for the worker process.
    kg: Optional[KG] = None

    depth: int = attr.ib(
        validator=[attr.validators.instance_of(int), _check_depth]
    )
    max_walks: Optional[int] = attr.ib(  # type: ignore
        default=None,
        validator=[
            attr.validators.optional(attr.validators.instance_of(int)),
            _check_max_walks,
        ],
    )
    sampler: Sampler = attr.ib(
        factory=lambda: UniformSampler(),
        validator=attr.validators.instance_of(Sampler),  # type: ignore
    )
    n_jobs: Optional[int] = attr.ib(  # type: ignore
        default=None,
        validator=[
            attr.validators.optional(attr.validators.instance_of(int)),
            _check_jobs,
        ],
    )
    with_reverse: Optional[bool] = attr.ib(
        kw_only=True,
        default=False,
        validator=attr.validators.instance_of(bool),
    )
    random_state: Optional[int] = attr.ib(
        kw_only=True,
        default=None,
        validator=attr.validators.optional(attr.validators.instance_of(int)),
    )

    _is_support_remote: bool = attr.ib(init=False, repr=False, default=True)

    def __attrs_post_init__(self):
        if self.n_jobs == -1:
            self.n_jobs = multiprocessing.cpu_count()
        self.sampler.random_state = self.random_state

    def extract(
        self, kg: KG, instances: Entities, verbose: int = 0
    ) -> List[str]:
        """Fits the provided sampling strategy and then calls the
        private _extract method that is implemented for each of the
        walking strategies.

        Args:
            kg: The Knowledge Graph.
            instances: The instances to be extracted from the Knowledge Graph.
            verbose: The verbosity level.
                0: does not display anything;
                1: display of the progress of extraction and training of walks;
                2: debugging.
                Defaults to 0.

        Returns:
            The 2D matrix with its number of rows equal to the number of
            provided instances; number of column equal to the embedding size.

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
            kg._fill_hops(instances)

        with multiprocessing.Pool(process, self._init_worker, [kg]) as pool:
            res = list(
                tqdm(
                    pool.imap_unordered(self._proc, instances),
                    total=len(instances),
                    disable=True if verbose == 0 else False,
                )
            )

        instance_walks = {
            instance: walks for elm in res for instance, walks in elm.items()
        }

        canonical_walks = set()
        for instance in instances:
            canonical_walks.update(instance_walks[instance])
        return list(canonical_walks)

    @abstractmethod
    def _extract(self, kg: KG, instance: Vertex) -> EntityWalks:
        """Extracts walks rooted at the provided instances which are then each
        transformed into a numerical representation.

        Args:
            kg: The Knowledge Graph.
            instance: The instance to be extracted from the Knowledge Graph.

        Returns:
            The 2D matrix with its number of rows equal to the number of
            provided instances; number of column equal to the embedding size.

        """
        raise NotImplementedError("This must be implemented!")

    def _init_worker(self, init_kg: KG) -> None:
        """Initializes each worker process.

        Args:
            init_kg: The Knowledge Graph to provide to each worker process.

        """
        global kg
        kg = init_kg  # type: ignore

    def _proc(self, instance: str) -> EntityWalks:
        """Executed by each process.

        Args:
            instance: The instance to be extracted from the Knowledge Graph.

        Returns:
            The extraction of walk by the process.

        """
        global kg
        return self._extract(kg, Vertex(instance))  # type: ignore
