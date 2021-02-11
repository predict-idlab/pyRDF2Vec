import abc
import asyncio
import multiprocessing
from typing import Dict, List, Optional, Set, Tuple

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
class Walker(metaclass=abc.ABCMeta):
    """Base class for the walking strategies.

    Attributes:
        depth: The depth per entity.
        max_walks: The maximum number of walks per entity.
        sampler: The sampling strategy.
            Defaults to UniformSampler().
        n_jobs: The number of processes to use for multiprocessing. Use -1 to
            allocate as many processes as there are CPU cores available in the
            machine.
            Defaults to 1.
        seed: The seed to use to ensure ensure random determinism to generate
            the same walks for entities.
            Defaults to None.

    """

    # Global KG used later on for the worker process.
    kg: Optional[KG] = None

    depth: int = attr.ib()
    max_walks: Optional[int] = attr.ib(default=None)
    sampler: Sampler = attr.ib(factory=UniformSampler)
    n_jobs: int = attr.ib(default=1)
    seed: Optional[int] = attr.ib(kw_only=True, default=None)
    _is_support_remote: bool = attr.ib(init=False, repr=False, default=True)

    def __attrs_post_init__(self):
        if self.n_jobs == -1:
            self.n_jobs = multiprocessing.cpu_count()
        self.sampler = UniformSampler(seed=self.seed)

    def extract(
        self, kg: KG, instances: List[str], verbose: bool = False
    ) -> Set[Tuple[str, str, str]]:
        """Fits the provided sampling strategy and then calls the
        private _extract method that is implemented for each of the
        walking strategies.

        Args:
            kg: The Knowledge Graph.

                The graph from which the neighborhoods are extracted for the
                provided instances.
            instances: The instances to be extracted from the Knowledge Graph.
            verbose: If true, display a progress bar for the extraction of the
                walks.

        Returns:
            The 2D matrix with its number of rows equal to the number of
            provided instances; number of column equal to the embedding size.

        """
        if kg.is_remote and not self._is_support_remote:
            raise RemoteNotSupported(
                "Invalid walking strategy. Please, choose a walking strategy "
                + "that can fetch walks via a SPARQL endpoint server."
            )
        self.sampler.fit(kg)

        # To avoid circular imports
        if "CommunityWalker" in str(self):
            self._community_detection(kg)  # type: ignore

        if kg.is_remote and kg.is_mul_req:
            asyncio.run(kg._fill_entity_hops(instances))  # type: ignore

        with multiprocessing.Pool(
            self.n_jobs, self._init_worker, [kg]
        ) as pool:
            res = list(
                tqdm(
                    pool.imap_unordered(self._proc, instances),
                    total=len(instances),
                    disable=not verbose,
                )
            )
        instance_walks = {
            instance: walks for elm in res for instance, walks in elm.items()
        }
        canonical_walks = set()
        for instance in instances:
            canonical_walks.update(instance_walks[instance])
        return canonical_walks

    @abc.abstractmethod
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
