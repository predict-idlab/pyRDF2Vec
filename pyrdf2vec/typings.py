from typing import TYPE_CHECKING, Dict, List, Tuple, Union

if TYPE_CHECKING:
    from pyrdf2vec.graphs import Vertex  # noqa: F401

Hop = Tuple["Vertex", "Vertex"]

SWalk = Tuple[str, ...]
Walk = Tuple["Vertex", ...]

Embeddings = List[str]

EntityWalks = Dict[str, Tuple[SWalk, ...]]
Entities = List[str]

Literal = Union[float, str]
Literals = List[List[Union[Literal, Tuple[Literal, ...]]]]

Response = List[Dict[str, Dict[str, str]]]
