import abc
import multiprocessing
from typing import Any, Dict, List, Optional, Set, Tuple

import rdflib
from tqdm import tqdm

from pyrdf2vec.graphs import KG
from pyrdf2vec.samplers import Sampler, UniformSampler


class RemoteNotSupported(Exception):
    """Base exception class for the lack of support of a walking strategy for
    the extraction of walks via a SPARQL endpoint server.

    """

    pass


class Walker(metaclass=abc.ABCMeta):
    """Base class for the walking strategies.

    Attributes:
        depth: The depth per entity.
        walks_per_graph: The maximum number of walks per entity.
        sampler: The sampling strategy.
            Defaults to UniformSampler().
        n_jobs: The number of processes to use for multiprocessing. Use -1 to
            allocate as many processes as there are CPU cores available in the
            machine.
            Defaults to 1.
        is_support_remote: If true, indicate that the walking strategy can be
            used to retrieve walks via a SPARQL endpoint server.
            Defaults to False.

    """

    # Global KG used later on for the worker process.
    kg = None

    def __init__(
        self,
        depth: int,
        walks_per_graph: Optional[int] = None,
        sampler: Sampler = UniformSampler(),
        n_jobs: int = 1,
        is_support_remote: bool = True,
    ):
        self.depth = depth
        self.is_support_remote = is_support_remote
        if n_jobs == -1:
            self.n_jobs = multiprocessing.cpu_count()
        else:
            self.n_jobs = n_jobs
        self.sampler = sampler
        self.walks_per_graph = walks_per_graph

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
        if kg.is_remote and not self.is_support_remote:
            raise RemoteNotSupported(
                "Invalid walking strategy. Please, choose a walking strategy "
                + "that can retrieve walks via a SPARQL endpoint server."
            )
        self.sampler.fit(kg)
        canonical_walks = set()

        # To avoid circular imports
        if "CommunityWalker" in str(self):
            self._community_detection(kg)  # type: ignore

        with multiprocessing.Pool(
            self.n_jobs, self._init_worker, [kg]
        ) as pool:
            res = list(
                tqdm(
                    # chunkfile = 10?
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

    def info(self):
        """Gets informations related to a Walker.

        Returns:
            A friendly display of the Walker.

        """
        return (
            f"{type(self).__name__}(depth={self.depth},"
            + f"walks_per_graph={self.walks_per_graph},"
            + f"sampler={type(self.sampler).__name__},"
            + f"n_jobs={self.n_jobs},"
            + f"is_support_remote={self.is_support_remote})"
        )

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
