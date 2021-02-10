import abc
import asyncio
import multiprocessing
from typing import Any, Dict, List, Optional, Set, Tuple

import attr
import rdflib
from tqdm import tqdm

from pyrdf2vec.graphs import KG
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
    kg = None

    depth: int = attr.ib()
    max_walks: Optional[int] = attr.ib(default=None)
    sampler: Optional[Sampler] = attr.ib(default=UniformSampler())
    n_jobs: int = attr.ib(default=1)
    seed: Optional[int] = attr.ib(kw_only=True, default=None)
    _is_support_remote: bool = attr.ib(init=False, default=True)

    def __attrs_post_init__(self):
        if self.n_jobs == -1:
            self.n_jobs = multiprocessing.cpu_count()
        self.sampler = UniformSampler(seed=self.seed)

    def extract(
        self, kg: KG, instances: List[rdflib.URIRef], verbose=False
    ) -> Set[Tuple[Any, ...]]:
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
                + "that can retrieve walks via a SPARQL endpoint server."
            )
        self.sampler.fit(kg)
        canonical_walks = set()

        # To avoid circular imports
        if "CommunityWalker" in str(self):
            self._community_detection(kg)  # type: ignore

        if kg.is_remote:
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
        res = {k: v for elm in res for k, v in elm.items()}  # type: ignore

        for instance in instances:
            canonical_walks.update(res[instance])
        return canonical_walks

    @abc.abstractmethod
    def _extract(
        self, kg: KG, instance: rdflib.URIRef
    ) -> Dict[Any, Tuple[Tuple[str, ...], ...]]:
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

    def _init_worker(self, init_kg):
        """Initializes each worker process.

        Args:
            init_kg: The Knowledge Graph to provide to each worker process.

        """
        global kg
        kg = init_kg

    def print_walks(
        self,
        kg: KG,
        instances: List[rdflib.URIRef],
        filename: str,
    ) -> None:
        """Prints the walks of a Knowledge Graph.

        Args:
            kg: The Knowledge Graph.

                The graph from which the neighborhoods are extracted for the
                provided instances.
            instances: The instances to be extracted from the Knowledge Graph.
            filename: The filename that contains the rdflib.Graph

        """
        walks = self.extract(kg, instances)
        walk_strs = []
        for _, walk in enumerate(walks):
            s = ""
            for i in range(len(walk)):
                s += f"{walk[i]} "
                if i < len(walk) - 1:
                    s += "--> "
            walk_strs.append(s)

        with open(filename, "w+") as f:
            for s in walk_strs:
                f.write(s)
                f.write("\n\n")

    def _proc(
        self, instance: rdflib.URIRef
    ) -> Dict[Any, Tuple[Tuple[str, ...], ...]]:
        """Executed by each process.

        Args:
            instance: The instance to be extracted from the Knowledge Graph.

        Returns:
            The extraction of walk by the process.

        """
        global kg
        return self._extract(kg, instance)  # type:ignore
