import nest_asyncio

from .rdf2vec import RDF2VecTransformer

# bypass the asyncio.run error for the Notebooks.
nest_asyncio.apply()

__all__ = [
    "RDF2VecTransformer",
]
__version__ = "0.2.3"
