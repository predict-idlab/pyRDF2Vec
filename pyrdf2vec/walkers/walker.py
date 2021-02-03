import abc
import multiprocessing
from typing import Any, Dict, List, Set, Tuple

import rdflib
from tqdm import tqdm

from pyrdf2vec.graphs import KG
from pyrdf2vec.samplers import Sampler, UniformSampler


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

    """

    def __init__(
        self,
        depth: int,
        walks_per_graph: float,
        sampler: Sampler = UniformSampler(),
        n_jobs: int = 1,
    ):
        self.depth = depth
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
            seq: The sequence composed of the Knowledge Graph and instances,
            given to each process.
            verbose: If true, display a progress bar for the extraction of the
                walks.


        Returns:
            The 2D matrix with its number of rows equal to the number of
            provided instances; number of column equal to the embedding size.

        """
        self.sampler.fit(kg)
        canonical_walks = set()
        seq = [(kg, instance) for _, instance in enumerate(instances)]
        with multiprocessing.Pool(self.n_jobs) as pool:
            res = list(
                tqdm(
                    pool.imap_unordered(self._proc, seq),
                    total=len(seq),
                    disable=not verbose,
                )
            )
        res = {
            k: v for element in res for k, v in element.items()
        }  # type: ignore

        for instance in instances:
            canonical_walks.update(res[instance])
        return canonical_walks

    @abc.abstractmethod
    def _extract(
        self, seq: Tuple[KG, rdflib.URIRef]
    ) -> Dict[Any, Tuple[Tuple[str, ...], ...]]:
        """Extracts walks rooted at the provided instances which are then each
        transformed into a numerical representation.

        Args:
            seq: The sequence composed of the Knowledge Graph and instances,
            given to each process.

        Returns:
            The 2D matrix with its number of rows equal to the number of
            provided instances; number of column equal to the embedding size.

        """
        raise NotImplementedError("This must be implemented!")

    def print_walks(
        self,
        kg: KG,
        instances: List[rdflib.URIRef],
        file_name: str,
    ) -> None:
        """Prints the walks of a Knowledge Graph.

        Args:
            kg: The Knowledge Graph.

                The graph from which the neighborhoods are extracted for the
                provided instances.
            instances: The instances to be extracted from the Knowledge Graph.
            file_name: The filename that contains the rdflib.Graph

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

        with open(file_name, "w+") as f:
            for s in walk_strs:
                f.write(s)
                f.write("\n\n")

    def _proc(
        self, seq: Tuple[KG, rdflib.URIRef]
    ) -> Dict[Any, Tuple[Tuple[str, ...], ...]]:
        """Executed by each process.

        Args:
            seq: The sequence composed of the Knowledge Graph and instances,
            given to each process.

        Returns:
            The extraction of walk by the process.

        """
        return self._extract(seq)
