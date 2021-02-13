import asyncio
import multiprocessing
from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Optional, Tuple

import attr
from tqdm import tqdm

from pyrdf2vec.graphs import KG, Vertex
from pyrdf2vec.samplers import Sampler, UniformSampler


class RemoteNotSupported(Exception):
    """Base exception class for the lack of support of a walking strategy for
    the extraction of walks via a SPARQL endpoint server.

    """

    pass


@attr.s
class Walker(ABC):
    """Base class for the walking strategies.

    Attributes:
        depth: The depth per entity.
        max_walks: The maximum number of walks per entity.
        sampler: The sampling strategy.
            Defaults to UniformSampler().
        n_jobs: The number of processes to use for multiprocessing. Use -1 to
            allocate as many processes as there are CPU cores available in the
            machine.
            Defaults to None.
        random_state: The random state to use to ensure ensure random
            determinism to generate the same walks for entities.
            Defaults to None.

    """

    # Global KG used later on for the worker process.
    kg: Optional[KG] = None

    depth: int = attr.ib(attr.validators.instance_of(int))  # type: ignore
    max_walks: Optional[int] = attr.ib(
        default=None,
        validator=attr.validators.optional(attr.validators.instance_of(int)),
    )
    sampler: Sampler = attr.ib(
        factory=lambda: UniformSampler(),
        validator=attr.validators.instance_of(Sampler),  # type: ignore
    )
    n_jobs: Optional[int] = attr.ib(
        default=None,
        validator=attr.validators.optional(attr.validators.instance_of(int)),
    )
    random_state: Optional[int] = attr.ib(
        kw_only=True,
        default=None,
        validator=attr.validators.optional(attr.validators.instance_of(int)),
    )

    _is_support_remote: bool = attr.ib(init=False, repr=False, default=True)

    @depth.validator
    def _check_depth(self, attribute, value):
        if value < 0:
            raise ValueError(f"'depth' must be >= 0 (got {value})")

    @max_walks.validator
    def _check_max_walks(self, attribute, value):
        if value is not None and value < 0:
            raise ValueError(f"'max_walks' must be None or > 0 (got {value})")

    @n_jobs.validator
    def _check_jobs(self, attribute, value):
        if value is not None and value < -1:
            raise ValueError(
                f"'n_jobs' must be None, or equal to -1, or > 0 (got {value})"
            )

    def __attrs_post_init__(self):
        if self.n_jobs == -1:
            self.n_jobs = multiprocessing.cpu_count()
        self.sampler.random_state = self.random_state

    def extract(
        self, kg: KG, instances: List[str], verbose: int = 0
    ) -> Iterable[str]:
        """Fits the provided sampling strategy and then calls the
        private _extract method that is implemented for each of the
        walking strategies.

        Args:
            kg: The Knowledge Graph.

                The graph from which the neighborhoods are extracted for the
                provided instances.
            instances: The instances to be extracted from the Knowledge Graph.
            verbose: If equal to 1 or 2, display a progress bar for the
                extraction of the walks.

        Returns:
            The 2D matrix with its number of rows equal to the number of
            provided instances; number of column equal to the embedding size.

        """
        if kg._is_remote and not self._is_support_remote:
            raise RemoteNotSupported(
                "Invalid walking strategy. Please, choose a walking strategy "
                + "that can fetch walks via a SPARQL endpoint server."
            )
        self.sampler.fit(kg)

        # To avoid circular imports
        if "CommunityWalker" in str(self):
            self._community_detection(kg)  # type: ignore

        if kg._is_remote and kg.is_mul_req:
            asyncio.run(kg._fill_entity_hops(instances))  # type: ignore

        with multiprocessing.Pool(
            self.n_jobs, self._init_worker, [kg]
        ) as pool:
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
        return canonical_walks

    @abstractmethod
    def _extract(
        self, kg: KG, instance: Vertex
    ) -> Dict[str, Tuple[Tuple[str, ...], ...]]:
        """Extracts walks rooted at the provided instances which are then each
        transformed into a numerical representation.

        Args:
            kg: The Knowledge Graph.

                The graph from which the neighborhoods are extracted for the
                provided instances.
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

    def _proc(self, instance: str) -> Dict[str, Tuple[Tuple[str, ...], ...]]:
        """Executed by each process.

        Args:
            instance: The instance to be extracted from the Knowledge Graph.

        Returns:
            The extraction of walk by the process.

        """
        global kg
        return self._extract(kg, Vertex(instance))  # type: ignore
