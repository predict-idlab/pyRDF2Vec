from typing import TYPE_CHECKING, Dict, List, Set, Tuple, Union

if TYPE_CHECKING:
    from pyrdf2vec.graphs import Vertex

Hop = Tuple["Vertex", "Vertex"]
SWalk = Tuple[str, ...]
Walk = Tuple["Vertex", ...]

Embeddings = List[str]
EntityWalks = Dict[str, Set[SWalk]]
Entities = List[str]
Literal = Union[float, str]
Literals = List[List[Union[Literal, Tuple[Literal, ...]]]]

Response = List[Dict[str, Dict[str, str]]]
