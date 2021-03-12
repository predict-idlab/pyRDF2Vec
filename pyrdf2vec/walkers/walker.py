import asyncio
import multiprocessing
import warnings
from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Optional, Tuple

import attr
import numpy as np
from tqdm import tqdm

from pyrdf2vec.graphs import KG, Vertex
from pyrdf2vec.samplers import Sampler, UniformSampler

from pyrdf2vec.utils.validation import (  # isort: skip
    _check_depth,
    _check_jobs,
    _check_max_walks,
)


class RemoteNotSupported(Exception):
    """Base exception class for the lack of support of a walking strategy for
    the extraction of walks via a SPARQL endpoint server.

    """

    pass


@attr.s
class Walker(ABC):
    """Base class for the walking strategies.

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
    )  # type: ignore
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

    async def extract(
        self,
        kg: KG,
        instances: List[str],
        verbose: int = 0,
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

        process = self.n_jobs if self.n_jobs is not None else 1
        if (kg._is_remote and kg.mul_req) and process >= 2:
            warnings.warn(
                "Using 'mul_req=True' and/or 'n_jobs>=2' speed up the "
                + "extraction of entity's walks, but may violate the policy "
                + "of some SPARQL endpoint servers.",
                category=RuntimeWarning,
                stacklevel=2,
            )

        literals = []
        if kg._is_remote and kg.mul_req:
            queries = [
                kg.connector.get_query(vertex)
                for vertex in instances
                if kg._is_remote
            ]
            await asyncio.create_task(kg.connector.afetch(queries))
            literals = await asyncio.create_task(kg.get_literals(instances))

        with multiprocessing.Pool(process, self._init_worker, [kg]) as pool:
            res = list(
                tqdm(
                    pool.imap_unordered(self._proc, instances),
                    total=len(instances),
                    disable=True if verbose == 0 else False,
                )
            )

        instance_walks = {}
        instance_literals = {}
        for elm in res:
            for instance, walks_literals in elm.items():
                instance_walks[instance] = walks_literals[0]
                if len(literals) == 0:
                    instance_literals[instance] = walks_literals[1]

        canonical_walks = set()
        for instance in instances:
            canonical_walks.update(instance_walks[instance])

        if len(literals) == 0:
            for entity, v in instance_literals.items():
                tmp = [entity]
                for k in v:
                    if isinstance(k, list):
                        if len(k) == 0:
                            tmp += [np.NaN]
                        else:
                            tmp2 = []
                            for literal in k:
                                try:
                                    tmp2.append(float(literal))
                                except:
                                    tmp2.append(literal)
                            tmp += [tuple(tmp2)]
                    else:
                        try:
                            tmp += [float(k)]
                        except:
                            tmp += [k]
                literals.append(tmp)
        return [canonical_walks, literals]

    @abstractmethod
    async def _extract(
        self,
        kg: KG,
        instance: Vertex,
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
        return asyncio.run(self._extract(kg, Vertex(instance)))
